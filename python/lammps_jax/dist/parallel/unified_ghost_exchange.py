from collections import defaultdict
from functools import partial
from typing import Callable, NamedTuple, Optional
import numpy as np
import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
from jax import lax
try:
    from jax import shard_map
except ImportError:
    from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from nequix.model import bessel_basis, polynomial_cutoff
Array = jnp.ndarray
class GhostExchangeConfig(NamedTuple):
    mesh: Mesh
    n_domains: int
    n_atoms_per_domain: int
    n_atoms: int
    axis_name: str
def create_config(n_domains: int, n_atoms: int, mesh: Optional[Mesh]=None, axis_name: str='i') -> GhostExchangeConfig:
    assert n_atoms % n_domains == 0
    if mesh is None:
        devices = jax.devices()[:n_domains]
        mesh = Mesh(devices, axis_names=(axis_name,))
    return GhostExchangeConfig(mesh=mesh, n_domains=n_domains, n_atoms_per_domain=n_atoms // n_domains, n_atoms=n_atoms, axis_name=axis_name)
def _make_exchange(n_atoms, n_per, n_domains, axis_name):
    if n_domains == 1:
        return lambda features_owned, atom_start: features_owned
    @jax.custom_vjp
    def exchange(features_owned, atom_start):
        full = jnp.zeros((n_atoms, features_owned.shape[-1]), dtype=features_owned.dtype)
        full = lax.dynamic_update_slice(full, features_owned, (atom_start, 0))
        return lax.psum(full, axis_name=axis_name)
    def _fwd(features_owned, atom_start):
        return (exchange(features_owned, atom_start), atom_start)
    def _bwd(atom_start, g):
        total_g = lax.psum(g, axis_name=axis_name)
        return (lax.dynamic_slice_in_dim(total_g, atom_start, n_per, axis=0), None)
    exchange.defvjp(_fwd, _bwd)
    return exchange
def _bfs_halo(adj, owned, n_hops, n_atoms):
    active = set(owned)
    frontier = set(owned)
    for _ in range(n_hops):
        new_frontier = set()
        for node in frontier:
            for nb in adj.get(node, ()):
                if nb not in active and nb < n_atoms:
                    new_frontier.add(nb)
                    active.add(nb)
        frontier = new_frontier
    return active
def precompute_subgraphs(senders, receivers, shifts, edge_mask, n_atoms: int, n_domains: int, n_hops: int, species, *, max_nodes: Optional[int]=None, max_edges: Optional[int]=None):
    s_np = np.asarray(senders)
    r_np = np.asarray(receivers)
    sh_np = np.asarray(shifts)
    mask_np = np.asarray(edge_mask)
    sp_np = np.asarray(species)
    n_per = n_atoms // n_domains
    dim = sh_np.shape[-1]
    adj: dict = defaultdict(set)
    for e in range(len(s_np)):
        if mask_np[e]:
            adj[int(s_np[e])].add(int(r_np[e]))
            adj[int(r_np[e])].add(int(s_np[e]))
    all_nodes, all_edges = ([], [])
    n_active_list, n_edges_list = ([], [])
    for d in range(n_domains):
        lo, hi = (d * n_per, (d + 1) * n_per)
        owned = list(range(lo, hi))
        active = _bfs_halo(adj, owned, n_hops, n_atoms)
        halo = sorted(active - set(owned))
        node_list = owned + halo
        g2l = {g: l for l, g in enumerate(node_list)}
        local_edges = []
        for e in range(len(s_np)):
            s, r = (int(s_np[e]), int(r_np[e]))
            if mask_np[e] and s in g2l and (r in g2l) and (lo <= r < hi):
                local_edges.append((g2l[s], g2l[r], sh_np[e]))
        all_nodes.append(node_list)
        all_edges.append(local_edges)
        n_active_list.append(len(node_list))
        n_edges_list.append(len(local_edges))
    computed_max_nodes = max(n_active_list) + 1
    computed_max_edges = max(n_edges_list) if n_edges_list else 1
    if max_nodes is None:
        max_nodes = computed_max_nodes
    if max_edges is None:
        max_edges = computed_max_edges
    if computed_max_nodes > max_nodes or computed_max_edges > max_edges:
        raise ValueError(f'Subgraph exceeds pre-allocated capacity: nodes {computed_max_nodes} > {max_nodes} or edges {computed_max_edges} > {max_edges}. Reallocate with larger capacity.')
    pad_node = max_nodes - 1
    dom_nidx = np.zeros((n_domains, max_nodes), dtype=np.int32)
    dom_s = np.full((n_domains, max_edges), pad_node, dtype=np.int32)
    dom_r = np.full((n_domains, max_edges), pad_node, dtype=np.int32)
    dom_sh = np.zeros((n_domains, max_edges, dim), dtype=np.float32)
    dom_sp = np.zeros((n_domains, max_nodes), dtype=np.int32)
    dom_em = np.zeros((n_domains, max_edges), dtype=bool)
    for d in range(n_domains):
        na = n_active_list[d]
        nodes = all_nodes[d]
        edges = all_edges[d]
        dom_nidx[d, :na] = nodes
        dom_sp[d, :na] = sp_np[nodes]
        for i, (s, r, shift) in enumerate(edges):
            dom_s[d, i] = s
            dom_r[d, i] = r
            dom_sh[d, i] = shift
        dom_em[d, :len(edges)] = True
    return (jnp.asarray(dom_nidx), jnp.asarray(dom_s), jnp.asarray(dom_r), jnp.asarray(dom_sh), jnp.asarray(dom_sp), jnp.asarray(dom_em), n_active_list, n_edges_list)
def _node_energies(model, dR, edge_mask, species_owned, local_senders, local_receivers, node_indices, atom_start, n_per, n_atoms, exchange_fn):
    features = e3nn.IrrepsArray(e3nn.Irreps(f'{model.n_species}x0e'), jax.nn.one_hot(species_owned, model.n_species))
    safe_dR = jnp.where(edge_mask[:, None], dR, 1.0)
    sq_r = jnp.sum(safe_dR ** 2, axis=-1)
    sq_r_safe = jnp.where(sq_r == 0.0, 1.0, sq_r)
    r_norm = jnp.where(sq_r == 0.0, 0.0, jnp.sqrt(sq_r_safe))
    radial_basis = bessel_basis(r_norm, model.radial_basis_size, model.cutoff) * polynomial_cutoff(r_norm, model.cutoff, model.radial_polynomial_p)[:, None]
    sh = e3nn.spherical_harmonics(e3nn.s2_irreps(model.lmax), safe_dR, normalize=True, normalization='component')
    for layer in model.layers:
        projected = layer.linear_1(features)
        proj_global = exchange_fn(projected.array, atom_start)
        proj_local = e3nn.IrrepsArray(projected.irreps, proj_global[node_indices])
        radial_msg = jax.vmap(layer.radial_mlp)(radial_basis)
        messages = proj_local[local_senders]
        messages = e3nn.tensor_product(messages, sh, filter_ir_out=layer.tp_irreps)
        messages = messages * radial_msg
        messages = e3nn.IrrepsArray(messages.irreps, jnp.where(edge_mask[:, None], messages.array, 0.0))
        agg = e3nn.scatter_sum(messages, dst=local_receivers, output_size=n_per)
        agg = agg / jnp.sqrt(lax.stop_gradient(layer.avg_n_neighbors))
        if layer.index_weights:
            skip = layer.skip(species_owned, features)
        else:
            skip = layer.skip(features)
        features = layer.linear_2(agg) + skip
        if layer.layer_norm is not None:
            features = layer.layer_norm(features)
        features = e3nn.gate(features, even_act=jax.nn.silu, odd_act=jax.nn.tanh, even_gate_act=jax.nn.silu)
    ne = model.readout(features)
    ne = ne * lax.stop_gradient(model.scale) + lax.stop_gradient(model.shift)
    ne = ne + lax.stop_gradient(model.atom_energies[species_owned, None])
    return ne.array.squeeze(-1)
def _energy_and_edge_grads(model, box, nidx, ls, lr, lsh, lsp, lem, n_per, n_atoms, atom_start, exchange_fn, R_all):
    pos_frac = R_all[nidx]
    pos_real = pos_frac @ box
    dR = pos_real[lr] - (pos_real[ls] + lsh @ box)
    species_owned = lsp[:n_per]
    def energy_from_dR(edge_vectors):
        return jnp.sum(_node_energies(model, edge_vectors, lem, species_owned, ls, lr, nidx, atom_start, n_per, n_atoms, exchange_fn))
    return jax.value_and_grad(energy_from_dR)(dR)
def make_sharded_force(model, box: Array, config: GhostExchangeConfig) -> Callable:
    mesh = config.mesh
    axis_name = config.axis_name
    n_domains = config.n_domains
    n_per = config.n_atoms_per_domain
    n_atoms = config.n_atoms
    sharding_pos = NamedSharding(mesh, P(axis_name))
    sharding_sub = NamedSharding(mesh, P(axis_name))
    exchange_fn = _make_exchange(n_atoms, n_per, n_domains, axis_name)
    if n_domains == 1:
        @jax.jit
        def wrapped(R_flat, nidx, ls, lr, lsh, lsp, lem):
            nidx, ls, lr, lsh, lsp, lem = (x.squeeze(0) if x.ndim > 1 and x.shape[0] == 1 else x for x in (nidx, ls, lr, lsh, lsp, lem))
            _, edge_grads = _energy_and_edge_grads(model, box, nidx, ls, lr, lsh, lsp, lem, n_per, n_atoms, 0, exchange_fn, R_flat)
            forces = jnp.zeros_like(R_flat)
            forces = forces.at[nidx[lr]].add(-edge_grads)
            forces = forces.at[nidx[ls]].add(edge_grads)
            return forces
    else:
        @partial(shard_map, mesh=mesh, in_specs=(P(axis_name),) + (P(axis_name),) * 6, out_specs=P(axis_name), check_vma=False)
        def _sharded_force(R_local, nidx, ls, lr, lsh, lsp, lem):
            R_all = lax.all_gather(R_local, axis_name=axis_name).reshape(-1, 3)
            device_idx = lax.axis_index(axis_name)
            idx_dt = jnp.int64 if bool(getattr(jax.config, 'jax_enable_x64', False)) else jnp.int32
            atom_start = lax.convert_element_type(device_idx * n_per, idx_dt)
            nidx = nidx.squeeze(0)
            ls = ls.squeeze(0)
            lr = lr.squeeze(0)
            lsh = lsh.squeeze(0)
            lsp = lsp.squeeze(0)
            lem = lem.squeeze(0)
            _, edge_grads = _energy_and_edge_grads(model, box, nidx, ls, lr, lsh, lsp, lem, n_per, n_atoms, atom_start, exchange_fn, R_all)
            forces = jnp.zeros((n_atoms, 3), dtype=R_all.dtype)
            forces = forces.at[nidx[lr]].add(-edge_grads)
            forces = forces.at[nidx[ls]].add(edge_grads)
            return lax.psum_scatter(forces, axis_name=axis_name, scatter_dimension=0, tiled=True)
        @jax.jit
        def wrapped(R_flat, nidx, ls, lr, lsh, lsp, lem):
            return _sharded_force(jax.device_put(R_flat.reshape(n_domains, n_per, 3), sharding_pos), jax.device_put(nidx, sharding_sub), jax.device_put(ls, sharding_sub), jax.device_put(lr, sharding_sub), jax.device_put(lsh, sharding_sub), jax.device_put(lsp, sharding_sub), jax.device_put(lem, sharding_sub)).reshape(-1, 3)
    return wrapped
def make_sharded_energy(model, box: Array, config: GhostExchangeConfig) -> Callable:
    mesh = config.mesh
    axis_name = config.axis_name
    n_domains = config.n_domains
    n_per = config.n_atoms_per_domain
    n_atoms = config.n_atoms
    sharding_pos = NamedSharding(mesh, P(axis_name))
    sharding_sub = NamedSharding(mesh, P(axis_name))
    exchange_fn = _make_exchange(n_atoms, n_per, n_domains, axis_name)
    if n_domains == 1:
        @jax.jit
        def wrapped(R_flat, nidx, ls, lr, lsh, lsp, lem):
            nidx, ls, lr, lsh, lsp, lem = (x.squeeze(0) if x.ndim > 1 and x.shape[0] == 1 else x for x in (nidx, ls, lr, lsh, lsp, lem))
            E, _ = _energy_and_edge_grads(model, box, nidx, ls, lr, lsh, lsp, lem, n_per, n_atoms, 0, exchange_fn, R_flat)
            return E
    else:
        @partial(shard_map, mesh=mesh, in_specs=(P(axis_name),) + (P(axis_name),) * 6, out_specs=P(), check_vma=False)
        def _sharded_energy(R_local, nidx, ls, lr, lsh, lsp, lem):
            R_all = lax.all_gather(R_local, axis_name=axis_name).reshape(-1, 3)
            device_idx = lax.axis_index(axis_name)
            idx_dt = jnp.int64 if bool(getattr(jax.config, 'jax_enable_x64', False)) else jnp.int32
            atom_start = lax.convert_element_type(device_idx * n_per, idx_dt)
            nidx = nidx.squeeze(0)
            ls = ls.squeeze(0)
            lr = lr.squeeze(0)
            lsh = lsh.squeeze(0)
            lsp = lsp.squeeze(0)
            lem = lem.squeeze(0)
            E, _ = _energy_and_edge_grads(model, box, nidx, ls, lr, lsh, lsp, lem, n_per, n_atoms, atom_start, exchange_fn, R_all)
            return lax.psum(E, axis_name=axis_name)
        @jax.jit
        def wrapped(R_flat, nidx, ls, lr, lsh, lsp, lem):
            return _sharded_energy(jax.device_put(R_flat.reshape(n_domains, n_per, 3), sharding_pos), jax.device_put(nidx, sharding_sub), jax.device_put(ls, sharding_sub), jax.device_put(lr, sharding_sub), jax.device_put(lsh, sharding_sub), jax.device_put(lsp, sharding_sub), jax.device_put(lem, sharding_sub))
    return wrapped
