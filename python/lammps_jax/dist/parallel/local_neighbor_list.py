from typing import Tuple
import numpy as onp
import jax
import jax.numpy as jnp
from jax import lax
def estimate_cell_params(box, cutoff: float, n_atoms: int, capacity_mult: float=1.25) -> Tuple[onp.ndarray, int, int]:
    box_np = onp.asarray(box)
    if box_np.ndim == 2:
        side_lengths = onp.array([box_np[0, 0], box_np[1, 1], box_np[2, 2]])
    elif box_np.ndim == 1:
        side_lengths = box_np
    else:
        side_lengths = onp.full(3, float(box_np))
    cells_per_side = onp.floor(side_lengths / cutoff).astype(onp.int32)
    cells_per_side = onp.maximum(cells_per_side, 3)
    total_cells = int(onp.prod(cells_per_side))
    atoms_per_cell = n_atoms / total_cells
    cell_capacity = max(int(onp.ceil(atoms_per_cell * capacity_mult)), 4)
    volume = float(onp.prod(side_lengths))
    density = n_atoms / volume
    sphere_vol = 4.0 / 3.0 * onp.pi * cutoff ** 3
    max_neighbors = max(int(onp.ceil(density * sphere_vol * capacity_mult)), 8)
    return (cells_per_side, cell_capacity, max_neighbors)
def _cell_bin(R_frac, N, cells_per_side, cell_capacity):
    nx, ny, nz = (int(cells_per_side[0]), int(cells_per_side[1]), int(cells_per_side[2]))
    cell_count = nx * ny * nz
    idx = jnp.floor(R_frac * jnp.array([nx, ny, nz], dtype=R_frac.dtype)).astype(jnp.int32)
    idx = jnp.clip(idx, 0, jnp.array([nx - 1, ny - 1, nz - 1]))
    hashes = idx[:, 0] * (ny * nz) + idx[:, 1] * nz + idx[:, 2]
    sort_map = jnp.argsort(hashes)
    sorted_hash = hashes[sort_map]
    sorted_id = sort_map
    occupancy = jax.ops.segment_sum(jnp.ones(N, dtype=jnp.int32), hashes, cell_count)
    first = jnp.searchsorted(sorted_hash, sorted_hash, side='left')
    slot = jnp.arange(N, dtype=jnp.int32) - first
    overflow_slot = cell_count * cell_capacity
    flat_idx = jnp.where(slot < cell_capacity, sorted_hash * cell_capacity + slot, overflow_slot)
    id_buffer = N * jnp.ones(cell_count * cell_capacity + 1, dtype=jnp.int32)
    id_buffer = id_buffer.at[flat_idx].set(sorted_id)
    id_buffer = id_buffer[:cell_count * cell_capacity].reshape(nx, ny, nz, cell_capacity)
    did_overflow = jnp.max(occupancy) > cell_capacity
    return (id_buffer, did_overflow)
def _transform_box(box, dr):
    if box.ndim == 1:
        return dr * box
    return jnp.einsum('ij,...j->...i', box, dr)
def local_neighbor_list(R_frac, box, cutoff: float, owned_start, n_owned: int, cells_per_side, cell_capacity: int, max_neighbors: int):
    N = R_frac.shape[0]
    cells_per_side = jnp.asarray(cells_per_side, dtype=jnp.int32)
    cutoff_sq = cutoff ** 2
    id_buffer, cell_overflow = _cell_bin(R_frac, N, cells_per_side, cell_capacity)
    offsets = jnp.array([[dx, dy, dz] for dx in range(-1, 2) for dy in range(-1, 2) for dz in range(-1, 2)], dtype=jnp.int32)
    def _search_one_atom(atom_pos, atom_global_idx):
        cell_idx = jnp.floor(atom_pos * cells_per_side).astype(jnp.int32)
        cell_idx = jnp.clip(cell_idx, 0, cells_per_side - 1)
        neighbor_cells = (cell_idx[None, :] + offsets) % cells_per_side
        candidate_ids = id_buffer[neighbor_cells[:, 0], neighbor_cells[:, 1], neighbor_cells[:, 2]]
        candidate_ids = candidate_ids.reshape(-1)
        safe_ids = jnp.where(candidate_ids < N, candidate_ids, 0)
        dr = R_frac[safe_ids] - atom_pos
        dr = dr - jnp.round(dr)
        dr_real = _transform_box(box, dr)
        dist_sq = jnp.sum(dr_real ** 2, axis=-1)
        valid = (dist_sq < cutoff_sq) & (candidate_ids < N) & (candidate_ids != atom_global_idx)
        cumsum = jnp.cumsum(valid)
        slot = jnp.where(valid, cumsum - 1, n_candidates)
        receivers = jnp.full(max_neighbors, N, dtype=jnp.int32)
        receivers = receivers.at[jnp.clip(slot, 0, max_neighbors - 1)].set(jnp.where(valid, candidate_ids, N))
        n_found = cumsum[-1]
        atom_overflow = n_found > max_neighbors
        return (receivers, atom_overflow)
    n_candidates = 27 * cell_capacity
    owned_global_indices = owned_start + jnp.arange(n_owned, dtype=jnp.int32)
    owned_positions = lax.dynamic_slice_in_dim(R_frac, owned_start, n_owned, axis=0)
    receivers_all, atom_overflows = jax.vmap(_search_one_atom)(owned_positions, owned_global_indices)
    senders_all = jnp.broadcast_to(owned_global_indices[:, None], (n_owned, max_neighbors))
    neighbor_idx = jnp.stack([senders_all.reshape(-1), receivers_all.reshape(-1)])
    did_overflow = cell_overflow | jnp.any(atom_overflows)
    return (neighbor_idx, did_overflow)
_OFFSETS_27 = [(dx, dy, dz) for dx in range(-1, 2) for dy in range(-1, 2) for dz in range(-1, 2)]
def estimate_subgraph_params(box, cutoff: float, n_atoms: int, n_per: int, n_hops: int, capacity_mult: float=1.25) -> Tuple[onp.ndarray, int, int, int]:
    cells_per_side, cell_capacity, max_neighbors = estimate_cell_params(box, cutoff, n_atoms, capacity_mult)
    box_np = onp.asarray(box)
    if box_np.ndim == 2:
        side_lengths = onp.array([box_np[0, 0], box_np[1, 1], box_np[2, 2]])
    elif box_np.ndim == 1:
        side_lengths = box_np
    else:
        side_lengths = onp.full(3, float(box_np))
    volume = float(onp.prod(side_lengths))
    density = n_atoms / volume
    halo_radius = n_hops * cutoff
    sphere_vol = 4.0 / 3.0 * onp.pi * halo_radius ** 3
    n_halo_est = int(density * sphere_vol * capacity_mult)
    max_nodes = max(n_per + n_halo_est + 1, n_per + 1)
    return (cells_per_side, cell_capacity, max_nodes, max_neighbors)
def _bfs_halo_jit(R_frac, id_buffer, cells_per_side, box, cutoff_sq, owned_start, n_per, n_hops, N, cell_capacity):
    atom_indices = jnp.arange(N, dtype=jnp.int32)
    active = (atom_indices >= owned_start) & (atom_indices < owned_start + n_per)
    offsets = jnp.array(_OFFSETS_27, dtype=jnp.int32)
    def _neighbors_of(pos_i):
        ci = jnp.floor(pos_i * cells_per_side).astype(jnp.int32)
        ci = jnp.clip(ci, 0, cells_per_side - 1)
        ncells = (ci[None, :] + offsets) % cells_per_side
        cands = id_buffer[ncells[:, 0], ncells[:, 1], ncells[:, 2]].reshape(-1)
        safe = jnp.where(cands < N, cands, 0)
        dr = R_frac[safe] - pos_i
        dr = dr - jnp.round(dr)
        dr_real = _transform_box(box, dr)
        dist_sq = jnp.sum(dr_real ** 2, axis=-1)
        return (cands, (dist_sq < cutoff_sq) & (cands < N))
    def one_hop(_, active_mask):
        all_cands, all_valid = jax.vmap(_neighbors_of)(R_frac)
        masked = all_valid & active_mask[:, None]
        scatter_ids = jnp.where(masked, all_cands, N).reshape(-1)
        new_flags = jnp.zeros(N + 1, dtype=jnp.bool_)
        new_flags = new_flags.at[scatter_ids].max(masked.reshape(-1))
        return active_mask | new_flags[:N]
    return lax.fori_loop(0, n_hops, one_hop, active)
def _build_g2l(active_mask, owned_start, n_per, N, species, max_nodes):
    pad_node = max_nodes - 1
    idx = jnp.arange(N, dtype=jnp.int32)
    owned_mask = (idx >= owned_start) & (idx < owned_start + n_per)
    halo_mask = active_mask & ~owned_mask
    halo_cumsum = jnp.cumsum(halo_mask)
    g2l = jnp.where(owned_mask, idx - owned_start, jnp.where(halo_mask, n_per + halo_cumsum - 1, pad_node))
    n_halo = jnp.sum(halo_mask.astype(jnp.int32))
    n_active = n_per + n_halo
    node_idx = jnp.zeros(max_nodes, dtype=jnp.int32)
    node_idx = node_idx.at[:n_per].set(owned_start + jnp.arange(n_per, dtype=jnp.int32))
    max_halo = max_nodes - n_per - 1
    halo_global = jnp.where(halo_mask, idx, N)
    halo_sorted = jnp.sort(halo_global)[:max_halo]
    halo_safe = jnp.where(halo_sorted < N, halo_sorted, 0)
    node_idx = node_idx.at[n_per:n_per + max_halo].set(halo_safe)
    species_local = species[node_idx]
    species_local = jnp.where(jnp.arange(max_nodes) < n_active, species_local, 0)
    return (node_idx, g2l, species_local, n_active, n_active > pad_node)
def local_subgraph(R_frac, species, box, cutoff: float, owned_start, n_per: int, n_hops: int, cells_per_side, cell_capacity: int, max_nodes: int, max_edges_per_atom: int):
    N = R_frac.shape[0]
    cells_per_side = jnp.asarray(cells_per_side, dtype=jnp.int32)
    cutoff_sq = cutoff ** 2
    pad_node = max_nodes - 1
    max_edges = n_per * max_edges_per_atom
    id_buffer, cell_overflow = _cell_bin(R_frac, N, cells_per_side, cell_capacity)
    active_mask = _bfs_halo_jit(R_frac, id_buffer, cells_per_side, box, cutoff_sq, owned_start, n_per, n_hops, N, cell_capacity)
    node_idx, g2l, species_local, n_active, node_overflow = _build_g2l(active_mask, owned_start, n_per, N, species, max_nodes)
    offsets = jnp.array(_OFFSETS_27, dtype=jnp.int32)
    def _edges_for_owned_atom(atom_pos, atom_global_idx, atom_local_idx):
        ci = jnp.floor(atom_pos * cells_per_side).astype(jnp.int32)
        ci = jnp.clip(ci, 0, cells_per_side - 1)
        ncells = (ci[None, :] + offsets) % cells_per_side
        cands = id_buffer[ncells[:, 0], ncells[:, 1], ncells[:, 2]].reshape(-1)
        safe = jnp.where(cands < N, cands, 0)
        dr_frac = R_frac[safe] - atom_pos
        shift = jnp.round(dr_frac)
        dr_wrapped = dr_frac - shift
        dr_real = _transform_box(box, dr_wrapped)
        dist_sq = jnp.sum(dr_real ** 2, axis=-1)
        sender_local = g2l[safe]
        valid = (dist_sq < cutoff_sq) & (cands < N) & (cands != atom_global_idx) & (sender_local != pad_node)
        n_cands = 27 * cell_capacity
        cumsum = jnp.cumsum(valid)
        slot = jnp.where(valid, cumsum - 1, n_cands)
        buf_s = jnp.full(max_edges_per_atom + 1, pad_node, dtype=jnp.int32)
        buf_r = jnp.full(max_edges_per_atom + 1, pad_node, dtype=jnp.int32)
        buf_sh = jnp.zeros((max_edges_per_atom + 1, 3), dtype=R_frac.dtype)
        clip_slot = jnp.clip(slot, 0, max_edges_per_atom)
        buf_s = buf_s.at[clip_slot].set(jnp.where(valid, sender_local, pad_node))
        buf_r = buf_r.at[clip_slot].set(jnp.where(valid, atom_local_idx, pad_node))
        buf_sh = buf_sh.at[clip_slot].set(jnp.where(valid[:, None], shift, 0.0))
        n_found = cumsum[-1]
        em = jnp.arange(max_edges_per_atom) < n_found
        return (buf_s[:max_edges_per_atom], buf_r[:max_edges_per_atom], buf_sh[:max_edges_per_atom], em, n_found > max_edges_per_atom)
    owned_pos = lax.dynamic_slice_in_dim(R_frac, owned_start, n_per, axis=0)
    owned_global = owned_start + jnp.arange(n_per, dtype=jnp.int32)
    owned_local = jnp.arange(n_per, dtype=jnp.int32)
    all_s, all_r, all_sh, all_em, all_ov = jax.vmap(_edges_for_owned_atom)(owned_pos, owned_global, owned_local)
    senders = all_s.reshape(max_edges)
    receivers = all_r.reshape(max_edges)
    shifts = all_sh.reshape(max_edges, 3)
    edge_mask = all_em.reshape(max_edges)
    did_overflow = cell_overflow | node_overflow | jnp.any(all_ov)
    return (node_idx, senders, receivers, shifts, species_local, edge_mask, did_overflow)
