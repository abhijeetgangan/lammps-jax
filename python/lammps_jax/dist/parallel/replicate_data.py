import inspect
from functools import partial
from typing import Callable, NamedTuple, Optional, Union
import jax
import jax.numpy as jnp
import numpy as onp
from jax import grad, lax
from jax.ops import segment_sum
try:
    from jax import shard_map
except ImportError:
    from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax_md import space
Array = jnp.ndarray
BoxLike = Union[float, Array]
class DomainConfig(NamedTuple):
    mesh: Mesh
    n_domains: int
    n_atoms_per_domain: int
    box_size: Array
    r_cutoff: float
    axis_name: str
def create_config(n_domains: int, n_atoms_per_domain: int, box_size: BoxLike, r_cutoff: float, mesh: Optional[Mesh]=None, axis_name: str='i') -> DomainConfig:
    if mesh is None:
        devices = jax.devices()[:n_domains]
        mesh = Mesh(devices, axis_names=(axis_name,))
    return DomainConfig(mesh=mesh, n_domains=n_domains, n_atoms_per_domain=n_atoms_per_domain, box_size=jnp.asarray(box_size), r_cutoff=r_cutoff, axis_name=axis_name)
def _place_on_mesh(R_flat: Array, n_domains: int, n_per_domain: int, sharding: NamedSharding) -> Array:
    expected = (n_domains, n_per_domain, 3)
    if R_flat.shape == expected:
        if getattr(R_flat, 'sharding', None) == sharding:
            return R_flat
        return jax.device_put(R_flat, sharding)
    return jax.device_put(R_flat.reshape(*expected), sharding)
def make_sharded_energy(energy_fn: Callable, config: DomainConfig) -> Callable:
    mesh = config.mesh
    axis_name = config.axis_name
    n_domains = config.n_domains
    n_per_domain = config.n_atoms_per_domain
    sharding = NamedSharding(mesh, P(axis_name))
    @partial(shard_map, mesh=mesh, in_specs=(P(axis_name),), out_specs=P(), check_vma=False)
    def _sharded_energy(R_local):
        device_idx = lax.axis_index(axis_name)
        R_all = lax.all_gather(R_local, axis_name=axis_name).reshape(-1, 3)
        start = device_idx * n_per_domain
        E_local = energy_fn(R_all, start, n_per_domain)
        return lax.psum(E_local, axis_name=axis_name)
    @jax.jit
    def wrapped(R_flat):
        R_on_mesh = _place_on_mesh(R_flat, n_domains, n_per_domain, sharding)
        return _sharded_energy(R_on_mesh)
    return wrapped
def make_sharded_force(energy_fn: Callable, config: DomainConfig, use_reduce_scatter: bool=False) -> Callable:
    mesh = config.mesh
    axis_name = config.axis_name
    n_domains = config.n_domains
    n_per_domain = config.n_atoms_per_domain
    sharding = NamedSharding(mesh, P(axis_name))
    @partial(shard_map, mesh=mesh, in_specs=(P(axis_name),), out_specs=P(axis_name), check_vma=False)
    def _sharded_force(R_local):
        device_idx = lax.axis_index(axis_name)
        R_all = lax.all_gather(R_local, axis_name=axis_name).reshape(-1, 3)
        start = device_idx * n_per_domain
        index_dtype = jnp.int64 if bool(getattr(jax.config, 'jax_enable_x64', False)) else jnp.int32
        start = lax.convert_element_type(start, index_dtype)
        grad_all = grad(lambda r: energy_fn(r, start, n_per_domain))(R_all)
        if use_reduce_scatter:
            return -lax.psum_scatter(grad_all, axis_name=axis_name, scatter_dimension=0, tiled=True)
        grad_total = lax.psum(grad_all, axis_name=axis_name)
        return -lax.dynamic_slice_in_dim(grad_total, start, n_per_domain, axis=0)
    @jax.jit
    def wrapped(R_flat):
        R_on_mesh = _place_on_mesh(R_flat, n_domains, n_per_domain, sharding)
        return _sharded_force(R_on_mesh).reshape(-1, 3)
    return wrapped
def make_sharded_energy_force(energy_fn: Callable, config: DomainConfig, use_reduce_scatter: bool=False) -> Callable:
    mesh = config.mesh
    axis_name = config.axis_name
    n_domains = config.n_domains
    n_per_domain = config.n_atoms_per_domain
    sharding = NamedSharding(mesh, P(axis_name))
    @partial(shard_map, mesh=mesh, in_specs=(P(axis_name),), out_specs=(P(), P(axis_name)), check_vma=False)
    def _sharded_energy_force(R_local):
        device_idx = lax.axis_index(axis_name)
        R_all = lax.all_gather(R_local, axis_name=axis_name).reshape(-1, 3)
        start = device_idx * n_per_domain
        index_dtype = jnp.int64 if bool(getattr(jax.config, 'jax_enable_x64', False)) else jnp.int32
        start = lax.convert_element_type(start, index_dtype)
        E_local, grad_all = jax.value_and_grad(lambda r: energy_fn(r, start, n_per_domain))(R_all)
        E_total = lax.psum(E_local, axis_name=axis_name)
        if use_reduce_scatter:
            F_local = -lax.psum_scatter(grad_all, axis_name=axis_name, scatter_dimension=0, tiled=True)
        else:
            grad_total = lax.psum(grad_all, axis_name=axis_name)
            F_local = -lax.dynamic_slice_in_dim(grad_total, start, n_per_domain, axis=0)
        return (E_total, F_local)
    @jax.jit
    def wrapped(R_flat):
        R_on_mesh = _place_on_mesh(R_flat, n_domains, n_per_domain, sharding)
        E, F = _sharded_energy_force(R_on_mesh)
        return (E, F.reshape(-1, 3))
    return wrapped
def domain_energy_from_pair(pair_energy_fn: Callable, displacement_fn: Callable) -> Callable:
    metric = space.metric(displacement_fn)
    compute_distances = lambda R_local, R_all: jax.vmap(lambda r_i: jax.vmap(lambda r_j: metric(r_i, r_j))(R_all))(R_local)
    def domain_energy(R_all, local_start, n_local):
        n_all = R_all.shape[0]
        start = lax.convert_element_type(local_start, jnp.int32)
        R_local = lax.dynamic_slice_in_dim(R_all, start, n_local, axis=0)
        dr = compute_distances(R_local, R_all)
        global_idx = jnp.arange(n_all, dtype=jnp.int32)
        local_idx = start + jnp.arange(n_local, dtype=jnp.int32)
        diag_mask = local_idx[:, None] == global_idx[None, :]
        safe_dr = jnp.where(diag_mask, jnp.ones_like(dr), dr)
        pair_energies = jnp.where(diag_mask, 0.0, pair_energy_fn(safe_dr))
        return 0.5 * jnp.sum(pair_energies)
    return domain_energy
def domain_energy_from_eam(charge_fn: Callable, embed_fn: Callable, pair_fn: Callable, displacement_fn: Callable, r_cutoff: float) -> Callable:
    _ = r_cutoff
    metric = space.metric(displacement_fn)
    compute_distances = lambda R_local, R_all: jax.vmap(lambda r_i: jax.vmap(lambda r_j: metric(r_i, r_j))(R_all))(R_local)
    def domain_energy(R_all, local_start, n_local):
        n_all = R_all.shape[0]
        start = lax.convert_element_type(local_start, jnp.int32)
        R_local = lax.dynamic_slice_in_dim(R_all, start, n_local, axis=0)
        dr = compute_distances(R_local, R_all)
        rho_local = jnp.sum(charge_fn(dr), axis=1)
        E_embed = jnp.sum(embed_fn(rho_local))
        global_idx = jnp.arange(n_all, dtype=jnp.int32)
        local_idx = start + jnp.arange(n_local, dtype=jnp.int32)
        diag_mask = local_idx[:, None] == global_idx[None, :]
        safe_dr = jnp.where(diag_mask, jnp.ones_like(dr), dr)
        pair_matrix = jnp.where(diag_mask, 0.0, pair_fn(safe_dr))
        E_pair = 0.5 * jnp.sum(pair_matrix)
        return E_embed + E_pair
    return domain_energy
def _energy_fn_accepts_n_real_arg(energy_fn: Callable) -> bool:
    try:
        return len(inspect.signature(energy_fn).parameters) >= 5
    except (TypeError, ValueError):
        return False
def _validate_symmetric_neighbor_list(neighbor_idx: Array) -> None:
    idx = jax.device_get(neighbor_idx)
    if idx.ndim != 2 or idx.shape[0] != 2:
        raise ValueError('neighbor_idx must have shape (2, n_pairs) for sparse pair lists.')
    senders = onp.asarray(idx[0], dtype=onp.int64)
    receivers = onp.asarray(idx[1], dtype=onp.int64)
    valid = (senders >= 0) & (receivers >= 0)
    senders = senders[valid]
    receivers = receivers[valid]
    edge_codes = senders << 32 | receivers & 4294967295
    rev_codes = receivers << 32 | senders & 4294967295
    if not onp.array_equal(onp.sort(edge_codes), onp.sort(rev_codes)):
        raise ValueError('Neighbor energies require a symmetric sparse list with both (i, j) and (j, i) edges.')
def shard_neighbor_idx_by_sender(neighbor_idx: Array, n_domains: int, n_per_domain: int) -> Array:
    idx = onp.asarray(jax.device_get(neighbor_idx))
    if idx.ndim != 2 or idx.shape[0] != 2:
        raise ValueError('neighbor_idx must have shape (2, n_pairs) before sender sharding.')
    senders = idx[0]
    max_sender = n_domains * n_per_domain
    sender_in_range = (senders >= 0) & (senders < max_sender)
    per_domain_edges = []
    max_edges = 0
    for domain in range(n_domains):
        lo = domain * n_per_domain
        hi = lo + n_per_domain
        mask = sender_in_range & (senders >= lo) & (senders < hi)
        local_edges = idx[:, mask]
        per_domain_edges.append(local_edges)
        max_edges = max(max_edges, local_edges.shape[1])
    max_edges = max(max_edges, 1)
    sharded = -onp.ones((n_domains, 2, max_edges), dtype=idx.dtype)
    for domain, local_edges in enumerate(per_domain_edges):
        n_edges = local_edges.shape[1]
        if n_edges > 0:
            sharded[domain, :, :n_edges] = local_edges
    return jnp.asarray(sharded)
def domain_energy_from_pair_with_neighbors(pair_energy_fn: Callable, displacement_fn: Callable) -> Callable:
    metric = space.metric(displacement_fn)
    compute_distances = jax.vmap(metric)
    def domain_energy(R_all, neighbor_idx, local_start, n_local, n_real=None):
        n_all = R_all.shape[0]
        n_real_i = jnp.asarray(n_all if n_real is None else n_real, dtype=jnp.int32)
        senders, receivers = (neighbor_idx[0], neighbor_idx[1])
        dr = compute_distances(R_all[senders], R_all[receivers])
        valid = (senders < n_all) & (receivers < n_all) & (senders < n_real_i) & (receivers < n_real_i) & (senders != receivers)
        safe_dr = jnp.where(valid, dr, jnp.ones_like(dr))
        pair_E = pair_energy_fn(safe_dr) * valid.astype(dr.dtype)
        local_mask = ((senders >= local_start) & (senders < local_start + n_local) & (senders < n_real_i)).astype(dr.dtype)
        return 0.5 * jnp.sum(pair_E * local_mask)
    return domain_energy
def domain_energy_from_eam_with_neighbors(charge_fn: Callable, embed_fn: Callable, pair_fn: Callable, displacement_fn: Callable, r_cutoff: float) -> Callable:
    _ = r_cutoff
    metric = space.metric(displacement_fn)
    compute_distances = jax.vmap(metric)
    def domain_energy(R_all, neighbor_idx, local_start, n_local, n_real=None):
        n_all = R_all.shape[0]
        n_real_i = jnp.asarray(n_all if n_real is None else n_real, dtype=jnp.int32)
        senders, receivers = (neighbor_idx[0], neighbor_idx[1])
        dr = compute_distances(R_all[senders], R_all[receivers])
        valid = (senders < n_all) & (receivers < n_all) & (senders < n_real_i) & (receivers < n_real_i) & (senders != receivers)
        safe_dr = jnp.where(valid, dr, jnp.ones_like(dr))
        rho_contrib = charge_fn(safe_dr) * valid.astype(dr.dtype)
        rho_neighbors = segment_sum(rho_contrib, receivers, n_all)
        real_atom_mask = (jnp.arange(n_all) < n_real_i).astype(rho_neighbors.dtype)
        rho = rho_neighbors + charge_fn(0.0) * real_atom_mask
        idx = jnp.arange(n_all)
        local_mask = ((idx >= local_start) & (idx < local_start + n_local) & (idx < n_real_i)).astype(rho_neighbors.dtype)
        E_embed = jnp.sum(embed_fn(rho) * local_mask)
        pair_E = pair_fn(safe_dr) * valid.astype(dr.dtype)
        local_pair_mask = ((senders >= local_start) & (senders < local_start + n_local) & (senders < n_real_i)).astype(dr.dtype)
        E_pair = 0.5 * jnp.sum(pair_E * local_pair_mask)
        return E_embed + E_pair
    return domain_energy
def make_sharded_energy_with_nbrs(energy_fn: Callable, n_domains: int, n_per_domain: int, mesh: Optional[Mesh]=None, axis_name: str='i', n_real: Optional[int]=None, validate_neighbor_symmetry: bool=True, validate_neighbor_symmetry_once: bool=True, neighbor_sharding: str='replicated') -> Callable:
    if mesh is None:
        devices = jax.devices()[:n_domains]
        mesh = Mesh(devices, axis_names=(axis_name,))
    if neighbor_sharding not in ('replicated', 'sender'):
        raise ValueError("neighbor_sharding must be either 'replicated' or 'sender'.")
    energy_accepts_n_real = _energy_fn_accepts_n_real_arg(energy_fn)
    sharding = NamedSharding(mesh, P(axis_name))
    replicated_sharding = NamedSharding(mesh, P())
    validated_once = False
    @partial(shard_map, mesh=mesh, in_specs=(P(axis_name), P(), P()), out_specs=P(), check_vma=False)
    def _sharded_energy_replicated(R_local, nbrs_idx, n_real_device):
        device_idx = lax.axis_index(axis_name)
        R_all = lax.all_gather(R_local, axis_name=axis_name).reshape(-1, 3)
        start = device_idx * n_per_domain
        if energy_accepts_n_real:
            E_local = energy_fn(R_all, nbrs_idx, start, n_per_domain, n_real_device)
        else:
            E_local = energy_fn(R_all, nbrs_idx, start, n_per_domain)
        return lax.psum(E_local, axis_name=axis_name)
    @partial(shard_map, mesh=mesh, in_specs=(P(axis_name), P(axis_name), P()), out_specs=P(), check_vma=False)
    def _sharded_energy_sender(R_local, nbrs_idx_local, n_real_device):
        device_idx = lax.axis_index(axis_name)
        R_all = lax.all_gather(R_local, axis_name=axis_name).reshape(-1, 3)
        if nbrs_idx_local.ndim == 3:
            nbrs_idx_local = jnp.squeeze(nbrs_idx_local, axis=0)
        start = device_idx * n_per_domain
        if energy_accepts_n_real:
            E_local = energy_fn(R_all, nbrs_idx_local, start, n_per_domain, n_real_device)
        else:
            E_local = energy_fn(R_all, nbrs_idx_local, start, n_per_domain)
        return lax.psum(E_local, axis_name=axis_name)
    def wrapped_energy(R_flat, neighbor_idx, n_real_override=None):
        nonlocal validated_once
        if validate_neighbor_symmetry and (not validate_neighbor_symmetry_once or not validated_once):
            if neighbor_sharding == 'sender' and neighbor_idx.ndim == 3:
                if neighbor_idx.shape[0] != n_domains or neighbor_idx.shape[1] != 2:
                    raise ValueError('Sender-sharded neighbor_idx must have shape (n_domains, 2, max_edges_per_domain).')
                idx_for_validate = jnp.transpose(neighbor_idx, (1, 0, 2)).reshape(2, -1)
                _validate_symmetric_neighbor_list(idx_for_validate)
            else:
                _validate_symmetric_neighbor_list(neighbor_idx)
            validated_once = True
        n_real_value = n_real if n_real is not None else n_real_override
        if n_real_value is None:
            n_real_value = R_flat.shape[0]
        R_on_mesh = _place_on_mesh(R_flat, n_domains, n_per_domain, sharding)
        n_real_on_mesh = jax.device_put(jnp.asarray(n_real_value, dtype=jnp.int32), replicated_sharding)
        if neighbor_sharding == 'replicated':
            if neighbor_idx.ndim != 2:
                raise ValueError('Replicated neighbor_idx must have shape (2, n_pairs).')
            nbrs_on_mesh = jax.device_put(neighbor_idx, replicated_sharding)
            return _sharded_energy_replicated(R_on_mesh, nbrs_on_mesh, n_real_on_mesh)
        if neighbor_idx.ndim == 2:
            neighbor_idx = shard_neighbor_idx_by_sender(neighbor_idx, n_domains=n_domains, n_per_domain=n_per_domain)
        elif neighbor_idx.ndim != 3:
            raise ValueError('Sender-sharded neighbor_idx must have shape (n_domains, 2, max_edges_per_domain).')
        if neighbor_idx.shape[0] != n_domains or neighbor_idx.shape[1] != 2:
            raise ValueError('Sender-sharded neighbor_idx must have shape (n_domains, 2, max_edges_per_domain).')
        nbrs_on_mesh = jax.device_put(neighbor_idx, sharding)
        return _sharded_energy_sender(R_on_mesh, nbrs_on_mesh, n_real_on_mesh)
    return wrapped_energy
def make_sharded_force_with_nbrs(energy_fn: Callable, n_domains: int, n_per_domain: int, mesh: Optional[Mesh]=None, axis_name: str='i', n_real: Optional[int]=None, validate_neighbor_symmetry: bool=True, validate_neighbor_symmetry_once: bool=True, neighbor_sharding: str='replicated', use_reduce_scatter: bool=False) -> Callable:
    if mesh is None:
        devices = jax.devices()[:n_domains]
        mesh = Mesh(devices, axis_names=(axis_name,))
    if neighbor_sharding not in ('replicated', 'sender'):
        raise ValueError("neighbor_sharding must be either 'replicated' or 'sender'.")
    energy_accepts_n_real = _energy_fn_accepts_n_real_arg(energy_fn)
    sharding = NamedSharding(mesh, P(axis_name))
    replicated_sharding = NamedSharding(mesh, P())
    validated_once = False
    @partial(shard_map, mesh=mesh, in_specs=(P(axis_name), P(), P()), out_specs=P(axis_name), check_vma=False)
    def _sharded_force_replicated(R_local, nbrs_idx, n_real_device):
        device_idx = lax.axis_index(axis_name)
        R_all = lax.all_gather(R_local, axis_name=axis_name).reshape(-1, 3)
        start = device_idx * n_per_domain
        index_dtype = jnp.int64 if bool(getattr(jax.config, 'jax_enable_x64', False)) else jnp.int32
        start = lax.convert_element_type(start, index_dtype)
        if energy_accepts_n_real:
            grad_all = grad(lambda r: energy_fn(r, nbrs_idx, start, n_per_domain, n_real_device))(R_all)
        else:
            grad_all = grad(lambda r: energy_fn(r, nbrs_idx, start, n_per_domain))(R_all)
        if use_reduce_scatter:
            return -lax.psum_scatter(grad_all, axis_name=axis_name, scatter_dimension=0, tiled=True)
        grad_total = lax.psum(grad_all, axis_name=axis_name)
        return -lax.dynamic_slice_in_dim(grad_total, start, n_per_domain, axis=0)
    @partial(shard_map, mesh=mesh, in_specs=(P(axis_name), P(axis_name), P()), out_specs=P(axis_name), check_vma=False)
    def _sharded_force_sender(R_local, nbrs_idx_local, n_real_device):
        device_idx = lax.axis_index(axis_name)
        R_all = lax.all_gather(R_local, axis_name=axis_name).reshape(-1, 3)
        if nbrs_idx_local.ndim == 3:
            nbrs_idx_local = jnp.squeeze(nbrs_idx_local, axis=0)
        start = device_idx * n_per_domain
        index_dtype = jnp.int64 if bool(getattr(jax.config, 'jax_enable_x64', False)) else jnp.int32
        start = lax.convert_element_type(start, index_dtype)
        if energy_accepts_n_real:
            grad_all = grad(lambda r: energy_fn(r, nbrs_idx_local, start, n_per_domain, n_real_device))(R_all)
        else:
            grad_all = grad(lambda r: energy_fn(r, nbrs_idx_local, start, n_per_domain))(R_all)
        if use_reduce_scatter:
            return -lax.psum_scatter(grad_all, axis_name=axis_name, scatter_dimension=0, tiled=True)
        grad_total = lax.psum(grad_all, axis_name=axis_name)
        return -lax.dynamic_slice_in_dim(grad_total, start, n_per_domain, axis=0)
    def wrapped_force(R_flat, neighbor_idx, n_real_override=None):
        nonlocal validated_once
        if validate_neighbor_symmetry and (not validate_neighbor_symmetry_once or not validated_once):
            if neighbor_sharding == 'sender' and neighbor_idx.ndim == 3:
                if neighbor_idx.shape[0] != n_domains or neighbor_idx.shape[1] != 2:
                    raise ValueError('Sender-sharded neighbor_idx must have shape (n_domains, 2, max_edges_per_domain).')
                idx_for_validate = jnp.transpose(neighbor_idx, (1, 0, 2)).reshape(2, -1)
                _validate_symmetric_neighbor_list(idx_for_validate)
            else:
                _validate_symmetric_neighbor_list(neighbor_idx)
            validated_once = True
        n_real_value = n_real if n_real is not None else n_real_override
        if n_real_value is None:
            n_real_value = R_flat.shape[0]
        R_on_mesh = _place_on_mesh(R_flat, n_domains, n_per_domain, sharding)
        n_real_on_mesh = jax.device_put(jnp.asarray(n_real_value, dtype=jnp.int32), replicated_sharding)
        if neighbor_sharding == 'replicated':
            if neighbor_idx.ndim != 2:
                raise ValueError('Replicated neighbor_idx must have shape (2, n_pairs).')
            nbrs_on_mesh = jax.device_put(neighbor_idx, replicated_sharding)
            return _sharded_force_replicated(R_on_mesh, nbrs_on_mesh, n_real_on_mesh).reshape(-1, 3)
        if neighbor_idx.ndim == 2:
            neighbor_idx = shard_neighbor_idx_by_sender(neighbor_idx, n_domains=n_domains, n_per_domain=n_per_domain)
        elif neighbor_idx.ndim != 3:
            raise ValueError('Sender-sharded neighbor_idx must have shape (n_domains, 2, max_edges_per_domain).')
        if neighbor_idx.shape[0] != n_domains or neighbor_idx.shape[1] != 2:
            raise ValueError('Sender-sharded neighbor_idx must have shape (n_domains, 2, max_edges_per_domain).')
        nbrs_on_mesh = jax.device_put(neighbor_idx, sharding)
        return _sharded_force_sender(R_on_mesh, nbrs_on_mesh, n_real_on_mesh).reshape(-1, 3)
    return wrapped_force
def _expand_neighborhood(senders, receivers, owned_start, owned_count, n_layers, n_atoms, n_edges):
    atom_idx = jnp.arange(n_atoms)
    in_set = (atom_idx >= owned_start) & (atom_idx < owned_start + owned_count)
    valid_edge = (senders < n_atoms) & (receivers < n_atoms)
    def expand_one_hop(carry, _):
        in_set_ = carry
        sender_in = jnp.take(in_set_, senders, fill_value=False) & valid_edge
        receiver_in = jnp.take(in_set_, receivers, fill_value=False) & valid_edge
        newly_reached = jnp.zeros(n_atoms, dtype=jnp.bool_)
        newly_reached = newly_reached.at[receivers].max(sender_in)
        newly_reached = newly_reached.at[senders].max(receiver_in)
        return (in_set_ | newly_reached, None)
    in_set, _ = lax.scan(expand_one_hop, in_set, None, length=n_layers)
    return in_set
def _extract_subgraph(R_all, species_all, senders, receivers, edge_features, atom_mask, max_atoms, max_edges):
    n_atoms = R_all.shape[0]
    n_edges = senders.shape[0]
    global_to_local = jnp.cumsum(atom_mask.astype(jnp.int32)) - 1
    global_to_local = jnp.where(atom_mask, global_to_local, max_atoms)
    n_sub = jnp.sum(atom_mask.astype(jnp.int32))
    atom_indices = jnp.where(atom_mask, jnp.arange(n_atoms), n_atoms)
    sorted_idx = jnp.argsort(~atom_mask)
    selected = sorted_idx[:max_atoms]
    selected = jnp.where(jnp.arange(max_atoms) < n_sub, selected, 0)
    R_sub = R_all[selected]
    R_sub = jnp.where((jnp.arange(max_atoms) < n_sub)[:, None], R_sub, jnp.zeros(3))
    species_sub = species_all[selected]
    species_sub = jnp.where(jnp.arange(max_atoms) < n_sub, species_sub, 0)
    sender_in = jnp.take(atom_mask, senders, fill_value=False)
    receiver_in = jnp.take(atom_mask, receivers, fill_value=False)
    edge_valid = sender_in & receiver_in & (senders < n_atoms) & (receivers < n_atoms)
    local_senders = jnp.take(global_to_local, senders, fill_value=max_atoms)
    local_receivers = jnp.take(global_to_local, receivers, fill_value=max_atoms)
    edge_order = jnp.argsort(~edge_valid)
    n_valid_edges = jnp.sum(edge_valid.astype(jnp.int32))
    sel_edges = edge_order[:max_edges]
    sel_valid = jnp.arange(max_edges) < n_valid_edges
    senders_sub = jnp.where(sel_valid, local_senders[sel_edges], max_atoms)
    receivers_sub = jnp.where(sel_valid, local_receivers[sel_edges], max_atoms)
    edges_sub = jnp.where(sel_valid[:, None], edge_features[sel_edges], jnp.ones(edge_features.shape[1]))
    return (R_sub, species_sub, senders_sub, receivers_sub, edges_sub, n_sub, global_to_local)
def _expand_neighborhood_host(senders, receivers, owned_start, owned_count, n_layers, n_atoms):
    in_set = onp.zeros(n_atoms, dtype=bool)
    in_set[owned_start:owned_start + owned_count] = True
    valid = (senders < n_atoms) & (receivers < n_atoms)
    s_valid = senders[valid]
    r_valid = receivers[valid]
    for _ in range(n_layers):
        sender_in = in_set[s_valid]
        receiver_in = in_set[r_valid]
        new_atoms = onp.zeros(n_atoms, dtype=bool)
        onp.maximum.at(new_atoms, r_valid, sender_in)
        onp.maximum.at(new_atoms, s_valid, receiver_in)
        in_set |= new_atoms
    return in_set
def _extract_subgraph_host(R_all, species_all, senders, receivers, shifts, atom_mask, max_atoms, max_edges):
    n_atoms = R_all.shape[0]
    dim = shifts.shape[1] if shifts.ndim == 2 else 3
    selected_global = onp.where(atom_mask)[0]
    n_sub = len(selected_global)
    g2l = onp.full(n_atoms, max_atoms, dtype=onp.int32)
    g2l[selected_global] = onp.arange(n_sub, dtype=onp.int32)
    pad_atoms = max_atoms - n_sub
    if pad_atoms > 0:
        selected_padded = onp.concatenate([selected_global, onp.zeros(pad_atoms, dtype=onp.int64)])
    else:
        selected_padded = selected_global[:max_atoms]
    R_np = onp.asarray(R_all)
    R_sub = R_np[selected_padded].copy()
    R_sub[n_sub:] = 0.0
    sp_np = onp.asarray(species_all)
    sp_sub = sp_np[selected_padded].copy()
    sp_sub[n_sub:] = 0
    s_clipped = onp.clip(senders, 0, n_atoms - 1)
    r_clipped = onp.clip(receivers, 0, n_atoms - 1)
    s_in = atom_mask[s_clipped] & (senders < n_atoms)
    r_in = atom_mask[r_clipped] & (receivers < n_atoms)
    edge_valid = s_in & r_in
    valid_indices = onp.where(edge_valid)[0]
    n_valid = len(valid_indices)
    if n_valid > max_edges:
        valid_indices = valid_indices[:max_edges]
        n_valid = max_edges
    pad_edges = max_edges - n_valid
    if pad_edges > 0:
        sel_edges = onp.concatenate([valid_indices, onp.zeros(pad_edges, dtype=onp.int64)])
    else:
        sel_edges = valid_indices
    arange_e = onp.arange(max_edges)
    valid_mask = arange_e < n_valid
    s_sub = onp.where(valid_mask, g2l[senders[sel_edges]], max_atoms)
    r_sub = onp.where(valid_mask, g2l[receivers[sel_edges]], max_atoms)
    shifts_np = onp.asarray(shifts)
    shifts_sub = onp.where(valid_mask[:, None], shifts_np[sel_edges], 0)
    return (jnp.asarray(R_sub), jnp.asarray(sp_sub), jnp.asarray(s_sub, dtype=jnp.int32), jnp.asarray(r_sub, dtype=jnp.int32), jnp.asarray(shifts_sub), n_sub, jnp.asarray(g2l, dtype=jnp.int32), jnp.asarray(selected_padded, dtype=jnp.int32))
def make_sharded_gnn_force(model_fn: Callable, species: Array, n_layers: int, config: DomainConfig, box: Array, max_atoms: Optional[int]=None, max_edges: Optional[int]=None, use_reduce_scatter: bool=True) -> Callable:
    mesh = config.mesh
    axis_name = config.axis_name
    n_domains = config.n_domains
    n_per_domain = config.n_atoms_per_domain
    n_atoms = n_domains * n_per_domain
    box_matrix = jnp.asarray(box)
    sharding = NamedSharding(mesh, P(axis_name))
    _max_atoms: int = n_atoms if max_atoms is None else max_atoms
    _max_edges: int = n_atoms * 200 if max_edges is None else max_edges
    def _squeeze(x):
        if x.ndim >= 2 and x.shape[0] == 1:
            return jnp.squeeze(x, axis=0)
        return x
    @partial(shard_map, mesh=mesh, in_specs=(P(axis_name), P(axis_name), P(axis_name), P(axis_name), P(axis_name), P(axis_name), P(axis_name)), out_specs=(P(), P(axis_name)), check_vma=False)
    def _sharded_ef(R_local, sp_sub_pad, s_sub, r_sub, shifts_sub, g2l_local, selected_global):
        device_idx = lax.axis_index(axis_name)
        R_all = lax.all_gather(R_local, axis_name=axis_name).reshape(-1, 3)
        start = device_idx * n_per_domain
        sp_sub_pad = _squeeze(sp_sub_pad)
        s_sub = _squeeze(s_sub)
        r_sub = _squeeze(r_sub)
        shifts_sub = _squeeze(shifts_sub)
        g2l_local = _squeeze(g2l_local)
        selected_global = _squeeze(selected_global)
        local_start = g2l_local[start]
        def local_energy(r_all_):
            r_sub_ = r_all_[selected_global]
            pos_real = space.transform(box_matrix, r_sub_)
            shifts_real = space.transform(box_matrix, shifts_sub)
            pos_real_pad = jnp.concatenate([pos_real, jnp.zeros((1, 3))])
            displacements = pos_real_pad[r_sub] - pos_real_pad[s_sub] - shifts_real
            edge_valid = (s_sub < _max_atoms) & (r_sub < _max_atoms)
            displacements = jnp.where(edge_valid[:, None], displacements, 1.0)
            ne = model_fn(displacements, sp_sub_pad, s_sub, r_sub)
            owned = lax.dynamic_slice_in_dim(ne[:_max_atoms], local_start, n_per_domain, axis=0)
            return jnp.sum(owned)
        E_local, grad_all = jax.value_and_grad(local_energy)(R_all)
        E_total = lax.psum(E_local, axis_name=axis_name)
        if use_reduce_scatter:
            F_local = -lax.psum_scatter(grad_all, axis_name=axis_name, scatter_dimension=0, tiled=True)
        else:
            grad_total = lax.psum(grad_all, axis_name=axis_name)
            F_local = -lax.dynamic_slice_in_dim(grad_total, start, n_per_domain, axis=0)
        return (E_total, F_local)
    def wrapped(R_flat, neighbor):
        R_all_np = onp.asarray(jax.device_get(R_flat))
        species_np = onp.asarray(jax.device_get(species))
        receivers_np, senders_np = (onp.asarray(jax.device_get(neighbor.idx[0])), onp.asarray(jax.device_get(neighbor.idx[1])))
        shifts_np = onp.asarray(jax.device_get(neighbor.shifts))
        per_domain_data = []
        for d in range(n_domains):
            start = d * n_per_domain
            atom_mask = _expand_neighborhood_host(senders_np, receivers_np, start, n_per_domain, n_layers, n_atoms)
            _, sp_sub, s_sub, r_sub, shifts_sub, n_sub, g2l, sel_padded = _extract_subgraph_host(R_all_np, species_np, senders_np, receivers_np, shifts_np, atom_mask, _max_atoms, _max_edges)
            sp_sub_pad = jnp.concatenate([sp_sub, jnp.zeros(1, dtype=sp_sub.dtype)])
            per_domain_data.append((sp_sub_pad, s_sub, r_sub, shifts_sub, g2l, sel_padded))
        sp_all = jnp.stack([d[0] for d in per_domain_data])
        s_all = jnp.stack([d[1] for d in per_domain_data])
        r_all = jnp.stack([d[2] for d in per_domain_data])
        sh_all = jnp.stack([d[3] for d in per_domain_data])
        g2l_all = jnp.stack([d[4] for d in per_domain_data])
        sel_all = jnp.stack([d[5] for d in per_domain_data])
        R_on_mesh = jax.device_put(R_flat.reshape(n_domains, n_per_domain, 3), sharding)
        sp_mesh = jax.device_put(sp_all, sharding)
        s_mesh = jax.device_put(s_all, sharding)
        r_mesh = jax.device_put(r_all, sharding)
        sh_mesh = jax.device_put(sh_all, sharding)
        g2l_mesh = jax.device_put(g2l_all, sharding)
        sel_mesh = jax.device_put(sel_all, sharding)
        E, F = _sharded_ef(R_on_mesh, sp_mesh, s_mesh, r_mesh, sh_mesh, g2l_mesh, sel_mesh)
        return (E, F.reshape(-1, 3))
    return wrapped
