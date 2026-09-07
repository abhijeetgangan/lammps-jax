"""Replicated-table per-layer feature exchange for message-passing potentials.

Each device owns a contiguous block of M = N / D atoms and a precomputed
subgraph with its n_hops halo. Every layer scatters the owned (M, F) features
into an (N, F) table and psums it; the custom VJP psums the cotangent and
slices the owned rows. The full table moves every layer regardless of D, so
this is a correctness reference; ghost_energy is the O(halo) path.

References:
    S. Plimpton, "Fast Parallel Algorithms for Short-Range Molecular
    Dynamics", J. Comput. Phys. 117, 1-19 (1995),
    doi:10.1006/jcph.1995.1039.
    A. P. Thompson et al., "LAMMPS - a flexible simulation tool for
    particle-based materials modeling at the atomic, meso, and continuum
    scales", Comput. Phys. Commun. 271, 108171 (2022),
    doi:10.1016/j.cpc.2021.108171.
"""

from collections import defaultdict
from functools import partial
from typing import Callable, Optional

import numpy as onp
import jax
import jax.numpy as jnp
from jax import lax, shard_map
from jax.sharding import NamedSharding, PartitionSpec as P
from jax_md import space

from lammps_jax.dist.parallel import ghost_exchange
from lammps_jax.dist.parallel.replicate_data import DomainConfig, owned_start

Array = jnp.ndarray


def grid_order(R_frac, grid):
    """Permutation grouping (N, 3) fractional positions into (px, py, pz) tiles.

    Returns:
        (perm, inverse) int64 (N,) arrays; ordered = data[perm], original = ordered[inverse].
    """
    frac = onp.asarray(R_frac) % 1.0
    px, py, pz = (int(g) for g in grid)
    cx = onp.clip((frac[:, 0] * px).astype(onp.int64), 0, px - 1)
    cy = onp.clip((frac[:, 1] * py).astype(onp.int64), 0, py - 1)
    cz = onp.clip((frac[:, 2] * pz).astype(onp.int64), 0, pz - 1)
    rank = (cx * py + cy) * pz + cz
    perm = onp.argsort(rank, kind='stable')
    inverse = onp.argsort(perm)
    return (perm, inverse)


def make_exchange(n_atoms, n_per, n_domains, axis_name):
    """Builds the per-layer feature exchange with its transpose as custom VJP.

    Returns:
        ``exchange(features_owned (M, F), atom_start) -> (N, F)``; the identity when D = 1.
    """
    if n_domains == 1:
        return lambda features_owned, atom_start: features_owned

    @jax.custom_vjp
    def exchange(features_owned, atom_start):
        full = jnp.zeros((n_atoms, features_owned.shape[-1]), dtype=features_owned.dtype)
        full = lax.dynamic_update_slice(full, features_owned, (atom_start, 0))
        return lax.psum(full, axis_name=axis_name)

    def exchange_fwd(features_owned, atom_start):
        return (exchange(features_owned, atom_start), atom_start)

    def exchange_bwd(atom_start, g):
        total_g = lax.psum(g, axis_name=axis_name)
        return (lax.dynamic_slice_in_dim(total_g, atom_start, n_per, axis=0), None)

    exchange.defvjp(exchange_fwd, exchange_bwd)
    return exchange


def precompute_subgraphs(senders, receivers, shifts, edge_mask, n_atoms: int,
                         n_domains: int, n_hops: int, species, *,
                         max_nodes: Optional[int] = None,
                         max_edges: Optional[int] = None):
    """Host-side extraction of one padded halo subgraph per domain.

    Args:
        senders: (E,) global sender indices.
        shifts: (E, 3) periodic shifts.
        edge_mask: (E,) validity mask.
        n_atoms: N; atom i belongs to domain i // M.
        n_hops: Halo width in edges; edges end at owned receivers only, so 1 suffices with exchange_fn.
        species: (N,) species indices.
        max_nodes: Node capacity, default max count + 1; slot max_nodes - 1 is the pad node.
        max_edges: Edge capacity, default max count.

    Returns:
        (node_idx (D, max_nodes), senders, receivers (D, max_edges), shifts (D, max_edges, 3),
        species (D, max_nodes), edge_mask (D, max_edges), n_active_list, n_edges_list).

    Raises:
        ValueError: N % D != 0, or a domain's node or edge count exceeds its capacity.
    """
    if n_atoms % n_domains:
        raise ValueError(f'n_atoms {n_atoms} is not divisible by n_domains {n_domains}')
    senders = onp.asarray(senders)
    receivers = onp.asarray(receivers)
    shifts = onp.asarray(shifts)
    edge_mask = onp.asarray(edge_mask)
    species = onp.asarray(species)
    n_per = n_atoms // n_domains
    dim = shifts.shape[-1]
    adjacency: dict = defaultdict(set)
    for edge in range(len(senders)):
        if edge_mask[edge]:
            adjacency[int(senders[edge])].add(int(receivers[edge]))
            adjacency[int(receivers[edge])].add(int(senders[edge]))
    all_nodes, all_edges = ([], [])
    n_active_list, n_edges_list = ([], [])
    for domain in range(n_domains):
        lo, hi = (domain * n_per, (domain + 1) * n_per)
        owned = list(range(lo, hi))
        active = set(owned)
        frontier = set(owned)
        for hop in range(n_hops):
            frontier = {neighbor for node in frontier for neighbor in adjacency.get(node, ())
                        if neighbor not in active and neighbor < n_atoms}
            active |= frontier
        halo = sorted(active - set(owned))
        node_list = owned + halo
        global_to_local = {node: local for local, node in enumerate(node_list)}
        local_edges = []
        for edge in range(len(senders)):
            sender, receiver = (int(senders[edge]), int(receivers[edge]))
            if edge_mask[edge] and lo <= receiver < hi and sender in global_to_local:
                local_edges.append((global_to_local[sender], global_to_local[receiver], shifts[edge]))
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
        raise ValueError(f'Subgraph exceeds pre-allocated capacity: nodes {computed_max_nodes} > {max_nodes} '
                         f'or edges {computed_max_edges} > {max_edges}. Reallocate with larger capacity.')
    pad_node = max_nodes - 1
    domain_node_idx = onp.zeros((n_domains, max_nodes), dtype=onp.int32)
    domain_senders = onp.full((n_domains, max_edges), pad_node, dtype=onp.int32)
    domain_receivers = onp.full((n_domains, max_edges), pad_node, dtype=onp.int32)
    domain_shifts = onp.zeros((n_domains, max_edges, dim), dtype=shifts.dtype)
    domain_species = onp.zeros((n_domains, max_nodes), dtype=onp.int32)
    domain_edge_mask = onp.zeros((n_domains, max_edges), dtype=bool)
    for domain in range(n_domains):
        n_active = n_active_list[domain]
        nodes = all_nodes[domain]
        edges = all_edges[domain]
        domain_node_idx[domain, :n_active] = nodes
        domain_species[domain, :n_active] = species[nodes]
        for i, (sender, receiver, shift) in enumerate(edges):
            domain_senders[domain, i] = sender
            domain_receivers[domain, i] = receiver
            domain_shifts[domain, i] = shift
        domain_edge_mask[domain, :len(edges)] = True
    return (jnp.asarray(domain_node_idx), jnp.asarray(domain_senders),
            jnp.asarray(domain_receivers), jnp.asarray(domain_shifts),
            jnp.asarray(domain_species), jnp.asarray(domain_edge_mask),
            n_active_list, n_edges_list)


def node_energies(model, dR, edge_mask, species_owned, local_senders,
                  local_receivers, exchange_fn):
    """Per-atom energies (M,) of the owned rows through the nequix layers.

    Args:
        dR: (max_edges, 3) real-space edge vectors.
        species_owned: (M,) species of the owned rows.
        local_receivers: (max_edges,) node indices, below M on valid edges.
        exchange_fn: ``(M, F) owned features -> (nodes, F)`` in the order senders index.
    """
    import e3nn_jax as e3nn
    from nequix.model import bessel_basis, polynomial_cutoff

    features = e3nn.IrrepsArray(e3nn.Irreps(f'{model.n_species}x0e'),
                                jax.nn.one_hot(species_owned, model.n_species))
    safe_dR = jnp.where(edge_mask[:, None], dR, 1.0)
    r_squared = jnp.sum(safe_dR ** 2, axis=-1)
    r_squared_safe = jnp.where(r_squared == 0.0, 1.0, r_squared)
    r_norm = jnp.where(r_squared == 0.0, 0.0, jnp.sqrt(r_squared_safe))
    radial_basis = (bessel_basis(r_norm, model.radial_basis_size, model.cutoff)
                    * polynomial_cutoff(r_norm, model.cutoff, model.radial_polynomial_p)[:, None])
    edge_harmonics = e3nn.spherical_harmonics(e3nn.s2_irreps(model.lmax), safe_dR,
                                              normalize=True, normalization='component')
    for layer in model.layers:
        projected = layer.linear_1(features)
        projected_nodes = e3nn.IrrepsArray(projected.irreps, exchange_fn(projected.array))
        radial_weights = jax.vmap(layer.radial_mlp)(radial_basis)
        messages = projected_nodes[local_senders]
        messages = e3nn.tensor_product(messages, edge_harmonics, filter_ir_out=layer.tp_irreps)
        messages = messages * radial_weights
        messages = e3nn.IrrepsArray(messages.irreps, jnp.where(edge_mask[:, None], messages.array, 0.0))
        aggregated = e3nn.scatter_sum(messages, dst=local_receivers, output_size=species_owned.shape[0],
                                      mode='drop')
        aggregated = aggregated / jnp.sqrt(lax.stop_gradient(layer.avg_n_neighbors))
        if layer.index_weights:
            skip = layer.skip(species_owned, features)
        else:
            skip = layer.skip(features)
        features = layer.linear_2(aggregated) + skip
        if layer.layer_norm is not None:
            features = layer.layer_norm(features)
        features = e3nn.gate(features, even_act=jax.nn.silu, odd_act=jax.nn.tanh,
                             even_gate_act=jax.nn.silu)
    energies = model.readout(features)
    energies = energies * lax.stop_gradient(model.scale) + lax.stop_gradient(model.shift)
    energies = energies + lax.stop_gradient(model.atom_energies[species_owned, None])
    return energies.array.squeeze(-1)


def energy_and_edge_grads(model, box, node_idx, local_senders, local_receivers,
                          local_shifts, local_species, edge_mask, n_per,
                          atom_start, exchange_fn, R_all):
    """Energy and gradient in dR = pos[receiver] - (pos[sender] + transform_box(box, shift)).

    Args:
        box: (3, 3) lattice vectors as columns.
        local_shifts: (max_edges, 3) fractional periodic shifts.
        local_species: (max_nodes,) species indices; the first M rows are owned.
        R_all: (N, 3) replicated fractional positions.

    Returns:
        energy, gradient (max_edges, 3) with respect to dR.
    """
    pos_real = R_all[node_idx] @ box.T
    dR = pos_real[local_receivers] - (pos_real[local_senders] + local_shifts @ box.T)
    species_owned = local_species[:n_per]
    exchange_nodes = lambda features: exchange_fn(features, atom_start)[node_idx]
    return jax.value_and_grad(lambda edge_vectors: jnp.sum(node_energies(
        model, edge_vectors, edge_mask, species_owned, local_senders, local_receivers,
        exchange_nodes)))(dR)


def ghost_energy(node_energy_fn, owned_pos, n_owned, species_owned, senders, receivers,
                 edge_mask, plan, box, config, axis_name):
    """Energy of the live owned rows with ghosts and per-layer features from exchange_apply.

    Args:
        node_energy_fn: Per-row energies (rows,); arguments as node_energies without the model.
        owned_pos: (rows, 3) fractional positions; through space.transform every derivative in them
            is real-space, and the gradient is the reverse communication.
        species_owned: (rows,) species indices.
        senders: (E,) node indices from ghost_exchange_subgraph.
        receivers: (E,) node indices below n_owned on valid edges.
        edge_mask: (E,) edge validity out to cutoff + skin, masked here at the cutoff.
        plan: Stages from ghost_exchange on owned_pos.
        box: (3, 3) lattice vectors as columns.
        config: The GhostExchangeConfig of the plan.
    """
    pos_nodes = ghost_exchange.exchange_apply(plan, owned_pos, config, axis_name, unwrap=True)
    wrap = jnp.asarray([1.0 if ranks == 1 else 0.0 for ranks in config.grid], dtype=owned_pos.dtype)
    dr_frac = pos_nodes[receivers] - pos_nodes[senders]
    dr_frac = dr_frac - jnp.round(dr_frac) * wrap
    dR = space.transform(box, dr_frac)
    edge_mask = edge_mask & (jnp.sum(dR ** 2, axis=-1) < config.cutoff ** 2)
    exchange_nodes = lambda features: ghost_exchange.exchange_apply(plan, features, config, axis_name)
    energies = node_energy_fn(dR, edge_mask, species_owned, senders, receivers, exchange_nodes)
    return jnp.sum(jnp.where(jnp.arange(species_owned.shape[0]) < n_owned, energies, 0.0))


def make_sharded_force(model, box: Array, config: DomainConfig) -> Callable:
    """Sharded forces for a nequix model on the ``precompute_subgraphs`` arrays.

    Args:
        box: (3, 3) lattice vectors as columns.
        config: Mesh and block sizes with N = D * M.

    Returns:
        ``wrapped(R_flat (N, 3) fractional, node_idx, local_senders, local_receivers,
        local_shifts, local_species, edge_mask) -> (N, 3)`` real-space forces.
    """
    mesh = config.mesh
    axis_name = config.axis_name
    n_domains = config.n_domains
    n_per = config.n_atoms_per_domain
    n_atoms = n_domains * n_per
    sharding = NamedSharding(mesh, P(axis_name))
    exchange_fn = make_exchange(n_atoms, n_per, n_domains, axis_name)

    if n_domains == 1:
        @jax.jit
        def wrapped(R_flat, node_idx, local_senders, local_receivers, local_shifts,
                    local_species, edge_mask):
            subgraph = [x.squeeze(0) if x.ndim > 1 and x.shape[0] == 1 else x
                        for x in (node_idx, local_senders, local_receivers, local_shifts, local_species, edge_mask)]
            energy, edge_grads = energy_and_edge_grads(model, box, *subgraph, n_per, 0,
                                                       exchange_fn, R_flat)
            node_idx, local_senders, local_receivers = subgraph[:3]
            forces = jnp.zeros((n_atoms, 3), dtype=R_flat.dtype)
            forces = forces.at[node_idx[local_receivers]].add(-edge_grads)
            return forces.at[node_idx[local_senders]].add(edge_grads)

        return wrapped

    @partial(shard_map, mesh=mesh, in_specs=(P(axis_name),) * 7,
             out_specs=P(axis_name), check_vma=False)
    def sharded_force(R_local, node_idx, local_senders, local_receivers, local_shifts,
                      local_species, edge_mask):
        R_all = lax.all_gather(R_local, axis_name=axis_name).reshape(-1, 3)
        atom_start = owned_start(axis_name, n_per)
        subgraph = [x.squeeze(0) for x in (node_idx, local_senders, local_receivers,
                                           local_shifts, local_species, edge_mask)]
        energy, edge_grads = energy_and_edge_grads(model, box, *subgraph, n_per, atom_start,
                                                   exchange_fn, R_all)
        node_idx, local_senders, local_receivers = subgraph[:3]
        forces = jnp.zeros((n_atoms, 3), dtype=R_all.dtype)
        forces = forces.at[node_idx[local_receivers]].add(-edge_grads)
        forces = forces.at[node_idx[local_senders]].add(edge_grads)
        return lax.psum_scatter(forces, axis_name=axis_name, scatter_dimension=0, tiled=True)

    @jax.jit
    def wrapped(R_flat, node_idx, local_senders, local_receivers, local_shifts,
                local_species, edge_mask):
        R_on_mesh = jax.device_put(R_flat.reshape(n_domains, n_per, 3), sharding)
        subgraph = [jax.device_put(x, sharding) for x in (node_idx, local_senders, local_receivers,
                                                          local_shifts, local_species, edge_mask)]
        return sharded_force(R_on_mesh, *subgraph).reshape(-1, 3)

    return wrapped

