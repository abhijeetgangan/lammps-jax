"""Replicated-data atom-decomposition parallelism for jax-md energy functions.

Each of D devices owns a block of M = N / D atoms, all-gathers every block
into the full (N, 3) array, and evaluates its block energy against it. Block
energies are psummed; forces are the block-energy gradient over all N
coordinates, folded across devices. ``lax.all_gather`` is the expand, tiled
``lax.psum_scatter`` the fold, autodiff the reverse communication.

References:
    S. Plimpton, "Fast Parallel Algorithms for Short-Range Molecular
    Dynamics", J. Comput. Phys. 117, 1-19 (1995),
    doi:10.1006/jcph.1995.1039.
    A. P. Thompson et al., "LAMMPS - a flexible simulation tool for
    particle-based materials modeling at the atomic, meso, and continuum
    scales", Comput. Phys. Commun. 271, 108171 (2022),
    doi:10.1016/j.cpc.2021.108171.
    W. Smith, "Molecular dynamics on hypercube parallel computers",
    Comput. Phys. Commun. 62, 229-248 (1991),
    doi:10.1016/0010-4655(91)90097-5: the replicated-data name.
    G. C. Fox et al., "Solving Problems on Concurrent Processors",
    Prentice-Hall (1988): origin of the expand/fold collectives.
"""

import inspect
from functools import partial
from typing import Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as onp
from jax import grad, lax, shard_map
from jax.ops import segment_sum
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax_md import space

Array = jnp.ndarray


class DomainConfig(NamedTuple):
    """Mesh and block-size description shared by the sharded wrappers.

    Attributes:
        mesh: 1-D device mesh of D devices.
        n_domains: Number of devices D.
        n_atoms_per_domain: Atoms owned per device M; N = D * M.
    """

    mesh: Mesh
    n_domains: int
    n_atoms_per_domain: int
    axis_name: str


def default_mesh(n_domains: int, axis_name: str) -> Mesh:
    """1-D mesh over the first D local devices; raises ValueError when fewer exist."""
    devices = jax.devices()
    if n_domains > len(devices):
        raise ValueError(f'n_domains {n_domains} exceeds the {len(devices)} local devices')
    return Mesh(devices[:n_domains], axis_names=(axis_name,))


def check_mesh(mesh: Mesh, axis_name: str, n_domains: int) -> None:
    """Raises ValueError unless mesh has n_domains devices along axis_name."""
    if axis_name not in mesh.axis_names or mesh.shape[axis_name] != n_domains:
        raise ValueError(f'mesh axis {axis_name!r} must hold n_domains {n_domains} devices, '
                         f'got {dict(mesh.shape)}')


def create_config(n_domains: int, n_atoms_per_domain: int,
                  mesh: Optional[Mesh] = None,
                  axis_name: str = 'i') -> DomainConfig:
    """Builds a DomainConfig; mesh defaults to the first D local devices.

    Raises:
        ValueError: mesh does not hold n_domains devices along axis_name.
    """
    if mesh is None:
        mesh = default_mesh(n_domains, axis_name)
    check_mesh(mesh, axis_name, n_domains)
    return DomainConfig(mesh=mesh, n_domains=n_domains,
                        n_atoms_per_domain=n_atoms_per_domain,
                        axis_name=axis_name)


def place_on_mesh(R_flat: Array, n_domains: int, n_per_domain: int,
                  sharding: NamedSharding) -> Array:
    """Reshapes (N, 3) positions to (D, M, 3) and places one block per device."""
    expected = (n_domains, n_per_domain, 3)
    if R_flat.shape == expected:
        if getattr(R_flat, 'sharding', None) == sharding:
            return R_flat
        return jax.device_put(R_flat, sharding)
    return jax.device_put(R_flat.reshape(*expected), sharding)


def owned_start(axis_name: str, n_per_domain: int) -> Array:
    """This device's first owned atom index, in the dynamic-slice index dtype."""
    index_dtype = jnp.int64 if bool(getattr(jax.config, 'jax_enable_x64', False)) else jnp.int32
    return lax.convert_element_type(lax.axis_index(axis_name) * n_per_domain, index_dtype)


def fold_forces(grad_all: Array, start: Array, n_per_domain: int,
                axis_name: str, use_reduce_scatter: bool) -> Array:
    """Folds per-device (N, 3) partial gradients into owned (M, 3) forces."""
    if use_reduce_scatter:
        return -lax.psum_scatter(grad_all, axis_name=axis_name,
                                 scatter_dimension=0, tiled=True)
    grad_total = lax.psum(grad_all, axis_name=axis_name)
    return -lax.dynamic_slice_in_dim(grad_total, start, n_per_domain, axis=0)


def make_sharded_energy(energy_fn: Callable, config: DomainConfig) -> Callable:
    """Replicated-data total energy of a domain energy.

    Args:
        energy_fn: ``(R_all (N, 3), local_start, n_local) -> scalar`` energy of the owned block.
        config: Mesh and block sizes; atom i is owned by device i // M.

    Returns:
        ``(N, 3) or (D, M, 3) positions -> scalar total energy``.
    """
    mesh = config.mesh
    axis_name = config.axis_name
    n_domains = config.n_domains
    n_per_domain = config.n_atoms_per_domain
    sharding = NamedSharding(mesh, P(axis_name))

    @partial(shard_map, mesh=mesh, in_specs=(P(axis_name),), out_specs=P(),
             check_vma=False)
    def sharded_energy(R_local):
        R_all = lax.all_gather(R_local, axis_name=axis_name).reshape(-1, 3)
        start = owned_start(axis_name, n_per_domain)
        E_local = energy_fn(R_all, start, n_per_domain)
        return lax.psum(E_local, axis_name=axis_name)

    @jax.jit
    def wrapped(R_flat):
        R_on_mesh = place_on_mesh(R_flat, n_domains, n_per_domain, sharding)
        return sharded_energy(R_on_mesh)

    return wrapped


def make_sharded_force(energy_fn: Callable, config: DomainConfig,
                       use_reduce_scatter: bool = False) -> Callable:
    """Force counterpart of ``make_sharded_energy``: ``positions -> (N, 3) forces``.

    Args:
        use_reduce_scatter: Tiled ``psum_scatter`` fold, else psum then slice.
    """
    mesh = config.mesh
    axis_name = config.axis_name
    n_domains = config.n_domains
    n_per_domain = config.n_atoms_per_domain
    sharding = NamedSharding(mesh, P(axis_name))

    @partial(shard_map, mesh=mesh, in_specs=(P(axis_name),),
             out_specs=P(axis_name), check_vma=False)
    def sharded_force(R_local):
        R_all = lax.all_gather(R_local, axis_name=axis_name).reshape(-1, 3)
        start = owned_start(axis_name, n_per_domain)
        grad_all = grad(lambda r: energy_fn(r, start, n_per_domain))(R_all)
        return fold_forces(grad_all, start, n_per_domain, axis_name, use_reduce_scatter)

    @jax.jit
    def wrapped(R_flat):
        R_on_mesh = place_on_mesh(R_flat, n_domains, n_per_domain, sharding)
        return sharded_force(R_on_mesh).reshape(-1, 3)

    return wrapped


def make_sharded_energy_force(energy_fn: Callable, config: DomainConfig,
                              use_reduce_scatter: bool = False) -> Callable:
    """make_sharded_energy and make_sharded_force in one call: ``positions -> (energy, (N, 3) forces)``."""
    mesh = config.mesh
    axis_name = config.axis_name
    n_domains = config.n_domains
    n_per_domain = config.n_atoms_per_domain
    sharding = NamedSharding(mesh, P(axis_name))

    @partial(shard_map, mesh=mesh, in_specs=(P(axis_name),),
             out_specs=(P(), P(axis_name)), check_vma=False)
    def sharded_energy_force(R_local):
        R_all = lax.all_gather(R_local, axis_name=axis_name).reshape(-1, 3)
        start = owned_start(axis_name, n_per_domain)
        E_local, grad_all = jax.value_and_grad(
            lambda r: energy_fn(r, start, n_per_domain))(R_all)
        E_total = lax.psum(E_local, axis_name=axis_name)
        F_local = fold_forces(grad_all, start, n_per_domain, axis_name, use_reduce_scatter)
        return (E_total, F_local)

    @jax.jit
    def wrapped(R_flat):
        R_on_mesh = place_on_mesh(R_flat, n_domains, n_per_domain, sharding)
        E, F = sharded_energy_force(R_on_mesh)
        return (E, F.reshape(-1, 3))

    return wrapped


def domain_energy_from_pair(pair_energy_fn: Callable,
                            displacement_fn: Callable) -> Callable:
    """Builds a dense-distance domain energy from an isotropic pair potential.

    Args:
        pair_energy_fn: Vectorized ``dr -> energy`` over a distance array.

    Returns:
        ``(R_all (N, 3), local_start, n_local) -> scalar`` over the (M, N) distance
        matrix with the diagonal masked and each pair weighted 0.5.
    """
    compute_distances = lambda R_local, R_all: space.distance(jax.vmap(
        lambda r_i: jax.vmap(lambda r_j: displacement_fn(r_i, r_j))(R_all))(R_local))

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


def domain_energy_from_eam(charge_fn: Callable, embed_fn: Callable,
                           pair_fn: Callable, displacement_fn: Callable) -> Callable:
    """Builds a dense-distance domain energy for an EAM-form potential.

    Args:
        charge_fn: Vectorized ``dr -> density``, zero beyond the cutoff.
        embed_fn: Vectorized ``rho -> embedding energy``.
        pair_fn: Vectorized ``dr -> pair energy``, zero beyond the cutoff.

    Returns:
        As ``domain_energy_from_pair``; rho (M,) includes the ``charge_fn(0)`` self term.
    """
    compute_distances = lambda R_local, R_all: space.distance(jax.vmap(
        lambda r_i: jax.vmap(lambda r_j: displacement_fn(r_i, r_j))(R_all))(R_local))

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


def energy_fn_accepts_n_real_arg(energy_fn: Callable) -> bool:
    """Detects the optional n_real parameter by name or a **kwargs catch-all; it is passed by keyword.

    A partial with n_real already bound keeps its own value and is reported as not accepting it.
    """
    if 'n_real' in (getattr(energy_fn, 'keywords', None) or {}):
        return False
    try:
        params = inspect.signature(energy_fn).parameters
    except (TypeError, ValueError):
        return False
    return 'n_real' in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def has_tag(energy_fn: Callable, name: str) -> bool:
    """True when energy_fn, or a callable it wraps through func or __wrapped__, carries the tag."""
    seen = set()
    while energy_fn is not None and id(energy_fn) not in seen:
        if getattr(energy_fn, name, False):
            return True
        seen.add(id(energy_fn))
        energy_fn = getattr(energy_fn, 'func', None) or getattr(energy_fn, '__wrapped__', None)
    return False


def validate_symmetric_neighbor_list(neighbor_idx: Array, n_atoms: Optional[int] = None) -> None:
    """Raises ValueError unless a (2, E) list holds (j, i) for every (i, j).

    Negative endpoints are padding, and so are endpoints at or above n_atoms when it is given.
    """
    idx = jax.device_get(neighbor_idx)
    if idx.ndim != 2 or idx.shape[0] != 2:
        raise ValueError('neighbor_idx must have shape (2, n_pairs) for sparse pair lists.')
    senders = onp.asarray(idx[0], dtype=onp.int64)
    receivers = onp.asarray(idx[1], dtype=onp.int64)
    valid = (senders >= 0) & (receivers >= 0)
    if n_atoms is not None:
        valid &= (senders < n_atoms) & (receivers < n_atoms)
    senders = senders[valid]
    receivers = receivers[valid]
    edge_codes = senders << 32 | receivers & 4294967295
    rev_codes = receivers << 32 | senders & 4294967295
    if not onp.array_equal(onp.sort(edge_codes), onp.sort(rev_codes)):
        raise ValueError('Neighbor energies require a symmetric sparse list with both (i, j) and (j, i) edges.')


def shard_neighbor_idx_by_sender(neighbor_idx: Array, n_domains: int,
                                 n_per_domain: int, max_edges_per_domain: Optional[int] = None) -> Array:
    """Partitions a (2, E) edge list on host by sender domain s // M into (D, 2, E_max), padded with -1.

    Args:
        max_edges_per_domain: Capacity E_max; default rounds the largest count up to a multiple of 256.

    Raises:
        ValueError: A domain holds more edges than max_edges_per_domain.
    """
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
    if max_edges_per_domain is None:
        max_edges = -(-max(max_edges, 1) // 256) * 256
    elif max_edges > max_edges_per_domain:
        raise ValueError(f'a domain holds {max_edges} edges, more than max_edges_per_domain {max_edges_per_domain}')
    else:
        max_edges = max_edges_per_domain
    sharded = -onp.ones((n_domains, 2, max_edges), dtype=idx.dtype)
    for domain, local_edges in enumerate(per_domain_edges):
        n_edges = local_edges.shape[1]
        if n_edges > 0:
            sharded[domain, :, :n_edges] = local_edges
    return jnp.asarray(sharded)


def domain_energy_from_pair_with_neighbors(pair_energy_fn: Callable,
                                           displacement_fn: Callable) -> Callable:
    """Sparse-neighbor variant of ``domain_energy_from_pair`` with the same arguments.

    Distances are ``space.distance`` of the batched displacement; the vmapped jax_md metric
    miscompiles on the XLA CPU backend in float64 for about 16385 to 40000 edges (jax 0.10.2).

    Returns:
        ``(R_all (N, 3), neighbor_idx (2, E), local_start, n_local, n_real=None) -> scalar``,
        0.5 per edge with an owned real sender; negative or >= n_real endpoints are masked.
    """
    displacement = jax.vmap(displacement_fn)

    def domain_energy(R_all, neighbor_idx, local_start, n_local, n_real=None):
        n_all = R_all.shape[0]
        n_real_i = jnp.asarray(n_all if n_real is None else n_real, dtype=jnp.int32)
        senders, receivers = (neighbor_idx[0], neighbor_idx[1])
        dr = space.distance(displacement(R_all[senders], R_all[receivers]))
        valid = (senders >= 0) & (receivers >= 0) & (senders < n_all) & (receivers < n_all) \
            & (senders < n_real_i) & (receivers < n_real_i) & (senders != receivers)
        safe_dr = jnp.where(valid, dr, jnp.ones_like(dr))
        pair_E = pair_energy_fn(safe_dr) * valid.astype(dr.dtype)
        local_mask = ((senders >= local_start) & (senders < local_start + n_local)
                      & (senders < n_real_i)).astype(dr.dtype)
        return 0.5 * jnp.sum(pair_E * local_mask)

    domain_energy.sender_weighted = True
    return domain_energy


def domain_energy_from_eam_with_neighbors(charge_fn: Callable, embed_fn: Callable,
                                          pair_fn: Callable,
                                          displacement_fn: Callable) -> Callable:
    """Sparse-neighbor variant of ``domain_energy_from_eam``, tagged ``requires_replicated_neighbors``.

    Distances as in ``domain_energy_from_pair_with_neighbors``.

    Returns:
        Same signature as ``domain_energy_from_pair_with_neighbors`` plus embedding on owned
        real atoms, with rho including the ``charge_fn(0)`` self term.
    """
    displacement = jax.vmap(displacement_fn)

    def domain_energy(R_all, neighbor_idx, local_start, n_local, n_real=None):
        n_all = R_all.shape[0]
        n_real_i = jnp.asarray(n_all if n_real is None else n_real, dtype=jnp.int32)
        senders, receivers = (neighbor_idx[0], neighbor_idx[1])
        dr = space.distance(displacement(R_all[senders], R_all[receivers]))
        valid = (senders >= 0) & (receivers >= 0) & (senders < n_all) & (receivers < n_all) \
            & (senders < n_real_i) & (receivers < n_real_i) & (senders != receivers)
        safe_dr = jnp.where(valid, dr, jnp.ones_like(dr))
        rho_contrib = charge_fn(safe_dr) * valid.astype(dr.dtype)
        rho_neighbors = segment_sum(rho_contrib, receivers, n_all)
        real_atom_mask = (jnp.arange(n_all) < n_real_i).astype(rho_neighbors.dtype)
        rho = rho_neighbors + charge_fn(0.0) * real_atom_mask
        idx = jnp.arange(n_all)
        local_mask = ((idx >= local_start) & (idx < local_start + n_local)
                      & (idx < n_real_i)).astype(rho_neighbors.dtype)
        E_embed = jnp.sum(embed_fn(rho) * local_mask)
        pair_E = pair_fn(safe_dr) * valid.astype(dr.dtype)
        local_pair_mask = ((senders >= local_start) & (senders < local_start + n_local)
                           & (senders < n_real_i)).astype(dr.dtype)
        E_pair = 0.5 * jnp.sum(pair_E * local_pair_mask)
        return E_embed + E_pair

    domain_energy.requires_replicated_neighbors = True
    return domain_energy


def check_neighbor_sharding(energy_fn: Callable, neighbor_sharding: Optional[str]) -> str:
    """Resolves None from the energy's tag: sender_weighted -> 'sender', requires_replicated_neighbors -> 'replicated'.

    Raises:
        ValueError: Unknown mode, an untagged energy without an explicit mode, or 'sender' for a
        receiver-scattering energy.
    """
    if neighbor_sharding is None:
        if has_tag(energy_fn, 'requires_replicated_neighbors'):
            neighbor_sharding = 'replicated'
        elif has_tag(energy_fn, 'sender_weighted'):
            neighbor_sharding = 'sender'
        else:
            raise ValueError("neighbor_sharding is required for an untagged energy: 'sender' if it weights "
                             "edges by owned sender, 'replicated' otherwise.")
    if neighbor_sharding not in ('replicated', 'sender'):
        raise ValueError("neighbor_sharding must be either 'replicated' or 'sender'.")
    if neighbor_sharding == 'sender' and has_tag(energy_fn, 'requires_replicated_neighbors'):
        raise ValueError("energy_fn scatters over edge receivers and requires neighbor_sharding='replicated'.")
    return neighbor_sharding



def make_nbrs_placement(n_domains: int, n_per_domain: int, mesh: Mesh,
                        axis_name: str, n_real: Optional[int],
                        validate_neighbor_symmetry: bool,
                        validate_neighbor_symmetry_once: bool,
                        neighbor_sharding: str,
                        max_edges_per_domain: Optional[int] = None,
                        accepts_n_real: bool = True) -> Callable:
    """Host-side ``place(R_flat, neighbor_idx, n_real_override)`` shared by the neighbor-list builders.

    Args:
        max_edges_per_domain: Fixed sender-sharded capacity so rebuilds keep one compiled shape.
        accepts_n_real: Whether the energy takes n_real; an override for one that does not raises.

    Returns:
        ``place`` yields R (D, M, 3), nbrs (2, E) or (D, 2, E_max) by sender, n_real int32 scalar.
        A jax Array neighbor list is placed once and reused while the same object is passed; numpy
        lists are placed on every call because they may be rewritten in place.
    """
    sharding = NamedSharding(mesh, P(axis_name))
    replicated_sharding = NamedSharding(mesh, P())
    state = {'validated': False, 'source': None, 'placed': None}

    def place(R_flat, neighbor_idx, n_real_override):
        """Places the inputs; a jax Array neighbor list is re-partitioned only when a new one arrives."""
        if n_real_override is not None and not accepts_n_real:
            raise ValueError('n_real_override given but energy_fn takes no n_real keyword')
        if validate_neighbor_symmetry and (not validate_neighbor_symmetry_once or not state['validated']):
            if neighbor_sharding == 'sender' and neighbor_idx.ndim == 3:
                if neighbor_idx.shape[0] != n_domains or neighbor_idx.shape[1] != 2:
                    raise ValueError('Sender-sharded neighbor_idx must have shape (n_domains, 2, max_edges_per_domain).')
                slab_senders = onp.asarray(jax.device_get(neighbor_idx[:, 0, :]), dtype=onp.int64)
                real = (slab_senders >= 0) & (slab_senders < n_domains * n_per_domain)
                if not onp.all((slab_senders // n_per_domain == onp.arange(n_domains)[:, None]) | ~real):
                    raise ValueError('sender-sharded slab d must hold only edges whose sender lies in '
                                     'domain d.')
                validate_symmetric_neighbor_list(jnp.transpose(neighbor_idx, (1, 0, 2)).reshape(2, -1),
                                                 n_atoms=n_domains * n_per_domain)
            else:
                validate_symmetric_neighbor_list(neighbor_idx, n_atoms=n_domains * n_per_domain)
            state['validated'] = True
        n_real_value = n_real if n_real is not None else n_real_override
        if n_real_value is None:
            n_real_value = n_domains * n_per_domain if R_flat.ndim == 3 else R_flat.shape[0]
        R_on_mesh = place_on_mesh(R_flat, n_domains, n_per_domain, sharding)
        n_real_on_mesh = jax.device_put(jnp.asarray(n_real_value, dtype=jnp.int32), replicated_sharding)
        if state['source'] is neighbor_idx:
            return (R_on_mesh, state['placed'], n_real_on_mesh)
        if neighbor_sharding == 'replicated':
            if neighbor_idx.ndim != 2:
                raise ValueError('Replicated neighbor_idx must have shape (2, n_pairs).')
            nbrs_on_mesh = jax.device_put(neighbor_idx, replicated_sharding)
        else:
            sharded = neighbor_idx
            if sharded.ndim == 2:
                sharded = shard_neighbor_idx_by_sender(sharded, n_domains=n_domains, n_per_domain=n_per_domain,
                                                      max_edges_per_domain=max_edges_per_domain)
            elif sharded.ndim != 3:
                raise ValueError('Sender-sharded neighbor_idx must have shape (n_domains, 2, max_edges_per_domain).')
            if sharded.shape[0] != n_domains or sharded.shape[1] != 2:
                raise ValueError('Sender-sharded neighbor_idx must have shape (n_domains, 2, max_edges_per_domain).')
            nbrs_on_mesh = jax.device_put(sharded, sharding)
        if isinstance(neighbor_idx, jax.Array):
            state['source'], state['placed'] = (neighbor_idx, nbrs_on_mesh)
        return (R_on_mesh, nbrs_on_mesh, n_real_on_mesh)

    return place


def make_sharded_energy_with_nbrs(energy_fn: Callable, n_domains: int,
                                  n_per_domain: int, mesh: Optional[Mesh] = None,
                                  axis_name: str = 'i', n_real: Optional[int] = None,
                                  validate_neighbor_symmetry: bool = True,
                                  validate_neighbor_symmetry_once: bool = True,
                                  neighbor_sharding: Optional[str] = None,
                                  max_edges_per_domain: Optional[int] = None) -> Callable:
    """Replicated-data total energy for neighbor-list domain energies.

    Args:
        energy_fn: ``(R_all (N, 3), neighbor_idx, local_start, n_local[, n_real]) -> scalar``; tags
            are read through functools.partial and __wrapped__ chains.
        mesh: Optional 1-D mesh with n_domains devices, else the first D local devices.
        n_real: Fixed real-atom count overriding the per-call n_real_override; edges that touch
            padding atoms in [n_real, D * M) must be symmetric too.
        validate_neighbor_symmetry: Host-check the (i, j)/(j, i) pairing.
        validate_neighbor_symmetry_once: Validate only the first call.
        neighbor_sharding: ``'sender'`` (D, 2, E_max) split by sender, valid only for energies that
            weight edges by owned sender, or ``'replicated'`` (2, E) on every device; None reads the
            energy's ``sender_weighted`` or ``requires_replicated_neighbors`` tag.
        max_edges_per_domain: Fixed E_max for sender sharding so rebuilds do not recompile.

    Returns:
        ``wrapped(R_flat, neighbor_idx, n_real_override=None) -> scalar``.
    """
    if mesh is None:
        mesh = default_mesh(n_domains, axis_name)
    check_mesh(mesh, axis_name, n_domains)
    neighbor_sharding = check_neighbor_sharding(energy_fn, neighbor_sharding)
    energy_accepts_n_real = energy_fn_accepts_n_real_arg(energy_fn)
    if n_real is not None and not energy_accepts_n_real:
        raise ValueError('n_real given but energy_fn takes no n_real keyword')

    def energy_core(R_local, neighbor_idx, n_real_device):
        R_all = lax.all_gather(R_local, axis_name=axis_name).reshape(-1, 3)
        if neighbor_idx.ndim == 3:
            neighbor_idx = jnp.squeeze(neighbor_idx, axis=0)
        start = owned_start(axis_name, n_per_domain)
        if energy_accepts_n_real:
            E_local = energy_fn(R_all, neighbor_idx, start, n_per_domain, n_real=n_real_device)
        else:
            E_local = energy_fn(R_all, neighbor_idx, start, n_per_domain)
        return lax.psum(E_local, axis_name=axis_name)

    nbrs_spec = P() if neighbor_sharding == 'replicated' else P(axis_name)
    sharded_energy = jax.jit(shard_map(energy_core, mesh=mesh, in_specs=(P(axis_name), nbrs_spec, P()),
                                       out_specs=P(), check_vma=False))
    place = make_nbrs_placement(n_domains, n_per_domain, mesh, axis_name, n_real,
                                validate_neighbor_symmetry,
                                validate_neighbor_symmetry_once, neighbor_sharding, max_edges_per_domain,
                                accepts_n_real=energy_accepts_n_real)

    def wrapped_energy(R_flat, neighbor_idx, n_real_override=None):
        return sharded_energy(*place(R_flat, neighbor_idx, n_real_override))

    return wrapped_energy


def make_sharded_force_with_nbrs(energy_fn: Callable, n_domains: int,
                                 n_per_domain: int, mesh: Optional[Mesh] = None,
                                 axis_name: str = 'i', n_real: Optional[int] = None,
                                 validate_neighbor_symmetry: bool = True,
                                 validate_neighbor_symmetry_once: bool = True,
                                 neighbor_sharding: Optional[str] = None,
                                 use_reduce_scatter: bool = False,
                                 max_edges_per_domain: Optional[int] = None) -> Callable:
    """Force counterpart of ``make_sharded_energy_with_nbrs``; ``wrapped`` returns (N, 3) forces.

    Args:
        use_reduce_scatter: Fold mode as in ``make_sharded_force``.
    """
    if mesh is None:
        mesh = default_mesh(n_domains, axis_name)
    check_mesh(mesh, axis_name, n_domains)
    neighbor_sharding = check_neighbor_sharding(energy_fn, neighbor_sharding)
    energy_accepts_n_real = energy_fn_accepts_n_real_arg(energy_fn)
    if n_real is not None and not energy_accepts_n_real:
        raise ValueError('n_real given but energy_fn takes no n_real keyword')

    def force_core(R_local, neighbor_idx, n_real_device):
        R_all = lax.all_gather(R_local, axis_name=axis_name).reshape(-1, 3)
        if neighbor_idx.ndim == 3:
            neighbor_idx = jnp.squeeze(neighbor_idx, axis=0)
        start = owned_start(axis_name, n_per_domain)
        if energy_accepts_n_real:
            grad_all = grad(lambda r: energy_fn(r, neighbor_idx, start, n_per_domain,
                                                n_real=n_real_device))(R_all)
        else:
            grad_all = grad(lambda r: energy_fn(r, neighbor_idx, start, n_per_domain))(R_all)
        return fold_forces(grad_all, start, n_per_domain, axis_name, use_reduce_scatter)

    nbrs_spec = P() if neighbor_sharding == 'replicated' else P(axis_name)
    sharded_force = jax.jit(shard_map(force_core, mesh=mesh, in_specs=(P(axis_name), nbrs_spec, P()),
                                      out_specs=P(axis_name), check_vma=False))
    place = make_nbrs_placement(n_domains, n_per_domain, mesh, axis_name, n_real,
                                validate_neighbor_symmetry,
                                validate_neighbor_symmetry_once, neighbor_sharding, max_edges_per_domain,
                                accepts_n_real=energy_accepts_n_real)

    def wrapped_force(R_flat, neighbor_idx, n_real_override=None):
        return sharded_force(*place(R_flat, neighbor_idx, n_real_override)).reshape(-1, 3)

    return wrapped_force
