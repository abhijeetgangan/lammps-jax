from typing import NamedTuple, Tuple
import numpy as onp
import jax
import jax.numpy as jnp
from jax import lax
class GhostExchangeConfig(NamedTuple):
    n_ranks: int
    max_owned: int
    max_ghost: int
    max_send: int
    cutoff: float
    cutoff_frac: onp.ndarray
    hops_x: int
    tile_width_x: float
def create_config(n_ranks: int, n_atoms: int, box, cutoff: float, capacity_mult: float=1.5) -> GhostExchangeConfig:
    box_np = onp.asarray(box)
    if box_np.ndim == 2:
        side = onp.array([box_np[0, 0], box_np[1, 1], box_np[2, 2]])
    elif box_np.ndim == 1:
        side = box_np.copy()
    else:
        side = onp.full(3, float(box_np))
    cutoff_frac = cutoff / side
    tile_width_x = 1.0 / n_ranks
    hops_x = max(int(onp.ceil(cutoff_frac[0] / tile_width_x)), 1)
    max_owned = int(onp.ceil(n_atoms / n_ranks * capacity_mult))
    halo_atoms_x = int(onp.ceil(n_atoms * cutoff_frac[0] * capacity_mult)) * 2 * hops_x
    max_ghost = halo_atoms_x
    max_ghost = max(max_ghost, 8)
    max_send = int(onp.ceil(n_atoms * cutoff_frac[0] * capacity_mult))
    max_send = max(max_send, 4)
    return GhostExchangeConfig(n_ranks=n_ranks, max_owned=max_owned, max_ghost=max_ghost, max_send=max_send, cutoff=cutoff, cutoff_frac=cutoff_frac, hops_x=hops_x, tile_width_x=tile_width_x)
def _pack_selected(arr, n_valid, mask, max_out):
    full_mask = mask & (jnp.arange(arr.shape[0]) < n_valid)
    cumsum = jnp.cumsum(full_mask)
    n_sel = cumsum[arr.shape[0] - 1]
    slot = jnp.where(full_mask, cumsum - 1, arr.shape[0])
    buf = jnp.zeros((max_out + 1,) + arr.shape[1:], dtype=arr.dtype)
    clip_slot = jnp.clip(slot, 0, max_out)
    buf = buf.at[clip_slot].set(jnp.where(full_mask.reshape(-1, *[1] * (arr.ndim - 1)), arr, 0))
    return (buf[:max_out], n_sel)
def _append(buf, n_buf, new_data, n_new, max_buf):
    n_to_copy = jnp.minimum(n_new, max_buf - n_buf)
    idx = jnp.arange(new_data.shape[0])
    target = n_buf + idx
    mask = (idx < n_to_copy) & (target < max_buf)
    safe_target = jnp.where(mask, target, 0)
    values = jnp.where(mask.reshape(-1, *[1] * (new_data.ndim - 1)), new_data, buf[safe_target])
    buf = buf.at[safe_target].set(values)
    return (buf, n_buf + n_to_copy)
def ghost_exchange(owned_data, n_owned, config: GhostExchangeConfig, axis_name: str):
    n_ranks = config.n_ranks
    max_ghost = config.max_ghost
    max_send = config.max_send
    cutoff_frac = jnp.asarray(config.cutoff_frac)
    tw = config.tile_width_x
    hops_x = config.hops_x
    D = owned_data.shape[1]
    device_idx = lax.axis_index(axis_name)
    lo_x = device_idx * tw
    hi_x = lo_x + tw
    max_local = config.max_owned + max_ghost
    all_data = jnp.zeros((max_local, D), dtype=owned_data.dtype)
    all_data = lax.dynamic_update_slice_in_dim(all_data, owned_data, 0, axis=0)
    n_all = n_owned
    ghost_data = jnp.zeros((max_ghost, D), dtype=owned_data.dtype)
    n_ghost = jnp.int32(0)
    owned_only = jnp.zeros_like(all_data)
    owned_only = lax.dynamic_update_slice_in_dim(owned_only, owned_data, 0, axis=0)
    for direction in [+1, -1]:
        perm = [(i, (i + direction) % n_ranks) for i in range(n_ranks)]
        cand = owned_only
        n_cand = n_owned
        for _ in range(hops_x):
            if direction == +1:
                halo_mask = cand[:, 0] >= hi_x - cutoff_frac[0]
            else:
                halo_mask = cand[:, 0] <= lo_x + cutoff_frac[0]
            send_buf, n_send = _pack_selected(cand, n_cand, halo_mask, max_send)
            recv_buf = lax.ppermute(send_buf, axis_name, perm)
            n_recv = lax.ppermute(jnp.array(n_send), axis_name, perm)
            at_lo = device_idx == 0
            at_hi = device_idx == n_ranks - 1
            need_unwrap = (direction == +1) & at_lo | (direction == -1) & at_hi
            shift = jnp.where(need_unwrap, jnp.float32(-direction), jnp.float32(0))
            recv_buf = recv_buf.at[:, 0].add(shift)
            ghost_data, n_ghost = _append(ghost_data, n_ghost, recv_buf, n_recv, max_ghost)
            all_data = lax.dynamic_update_slice_in_dim(all_data, recv_buf, n_all, axis=0)
            n_all = n_all + n_recv
            cand = recv_buf
            n_cand = n_recv
    return (ghost_data, n_ghost)
def ghost_exchange_neighbor_list(owned_pos, ghost_pos, n_owned, n_ghost, box, cutoff: float, cells_per_side, cell_capacity: int, max_neighbors: int, max_local: int):
    nx, ny, nz = (int(cells_per_side[0]), int(cells_per_side[1]), int(cells_per_side[2]))
    cells_j = jnp.array([nx, ny, nz], dtype=jnp.int32)
    cutoff_sq = cutoff ** 2
    N = max_local
    all_pos = jnp.zeros((N, 3), dtype=owned_pos.dtype)
    all_pos = lax.dynamic_update_slice_in_dim(all_pos, owned_pos, 0, axis=0)
    all_pos = lax.dynamic_update_slice_in_dim(all_pos, ghost_pos, n_owned, axis=0)
    n_all = n_owned + n_ghost
    cell_size = jnp.ones(3, dtype=owned_pos.dtype) / cells_j
    cell_count = nx * ny * nz
    idx_3d = jnp.floor(jnp.clip(all_pos, 0.0, 0.999) / cell_size).astype(jnp.int32)
    idx_3d = jnp.clip(idx_3d, 0, cells_j - 1)
    real_hashes = idx_3d[:, 0] * (ny * nz) + idx_3d[:, 1] * nz + idx_3d[:, 2]
    is_real = jnp.arange(N) < n_all
    hashes = jnp.where(is_real, real_hashes, cell_count)
    sort_map = jnp.argsort(hashes)
    sorted_hash = hashes[sort_map]
    id_buf_size = (cell_count + 1) * cell_capacity
    id_buffer_flat = N * jnp.ones(id_buf_size, dtype=jnp.int32)
    first = jnp.searchsorted(sorted_hash, sorted_hash, side='left')
    slot = jnp.arange(N, dtype=jnp.int32) - first
    flat_idx = jnp.where(slot < cell_capacity, sorted_hash * cell_capacity + slot, id_buf_size - 1)
    id_buffer_flat = id_buffer_flat.at[flat_idx].set(sort_map)
    id_buffer = id_buffer_flat[:cell_count * cell_capacity].reshape(nx, ny, nz, cell_capacity)
    real_counts = jnp.where(is_real, jnp.ones(N, dtype=jnp.int32), jnp.zeros(N, dtype=jnp.int32))
    occupancy = jax.ops.segment_sum(real_counts, real_hashes, cell_count)
    cell_overflow = jnp.max(occupancy) > cell_capacity
    offsets = jnp.array([[dx, dy, dz] for dx in range(-1, 2) for dy in range(-1, 2) for dz in range(-1, 2)], dtype=jnp.int32)
    def _search(atom_pos, atom_idx):
        ci = jnp.floor(jnp.clip(atom_pos, 0.0, 0.999) / cell_size).astype(jnp.int32)
        ci = jnp.clip(ci, 0, cells_j - 1)
        ncells = (ci[None, :] + offsets) % cells_j
        cands = id_buffer[ncells[:, 0], ncells[:, 1], ncells[:, 2]].reshape(-1)
        safe = jnp.where(cands < N, cands, 0)
        dr_frac = all_pos[safe] - atom_pos
        dr_frac = dr_frac.at[:, 1].set(dr_frac[:, 1] - jnp.round(dr_frac[:, 1]))
        dr_frac = dr_frac.at[:, 2].set(dr_frac[:, 2] - jnp.round(dr_frac[:, 2]))
        dr_real = jnp.einsum('ij,...j->...i', box, dr_frac) if box.ndim == 2 else dr_frac * box
        dist_sq = jnp.sum(dr_real ** 2, axis=-1)
        valid = (dist_sq < cutoff_sq) & (cands < N) & (safe < n_all) & (cands != atom_idx)
        n_cands = 27 * cell_capacity
        cumsum = jnp.cumsum(valid)
        slot = jnp.where(valid, cumsum - 1, n_cands)
        receivers = jnp.full(max_neighbors, N, dtype=jnp.int32)
        receivers = receivers.at[jnp.clip(slot, 0, max_neighbors - 1)].set(jnp.where(valid, cands, N))
        return (receivers, cumsum[27 * cell_capacity - 1] > max_neighbors)
    n_owned_static = owned_pos.shape[0]
    owned_positions = owned_pos
    owned_indices = jnp.arange(n_owned_static, dtype=jnp.int32)
    receivers_all, overflows = jax.vmap(_search)(owned_positions, owned_indices)
    senders_all = jnp.broadcast_to(owned_indices[:, None], (n_owned_static, max_neighbors))
    neighbor_idx = jnp.stack([senders_all.reshape(-1), receivers_all.reshape(-1)])
    did_overflow = cell_overflow | jnp.any(overflows)
    return (neighbor_idx, did_overflow)
def backward_pass(owned_forces, ghost_forces, n_owned, n_ghost, config: GhostExchangeConfig, axis_name: str):
    max_owned = config.max_owned
    F_full = jnp.zeros((config.max_owned + config.max_ghost, 3), dtype=owned_forces.dtype)
    F_full = F_full.at[:max_owned].set(owned_forces)
    F_full = lax.dynamic_update_slice_in_dim(F_full, ghost_forces, n_owned, axis=0)
    F_total = lax.psum(F_full, axis_name=axis_name)
    return lax.dynamic_slice_in_dim(F_total, 0, max_owned, axis=0)
def redistribute(owned_pos, n_owned, config: GhostExchangeConfig, axis_name: str):
    n_ranks = config.n_ranks
    max_owned = config.max_owned
    max_send = config.max_send
    tw = config.tile_width_x
    device_idx = lax.axis_index(axis_name)
    lo_x = device_idx * tw
    hi_x = lo_x + tw
    owned_pos = owned_pos % 1.0
    migrate_right = owned_pos[:, 0] >= hi_x
    migrate_left = owned_pos[:, 0] < lo_x
    stay = ~migrate_right & ~migrate_left & (jnp.arange(max_owned) < n_owned)
    stayed, n_stay = _pack_selected(owned_pos, n_owned, stay, max_owned)
    perm_right = [(i, (i + 1) % n_ranks) for i in range(n_ranks)]
    perm_left = [(i, (i - 1) % n_ranks) for i in range(n_ranks)]
    send_r, n_sr = _pack_selected(owned_pos, n_owned, migrate_right, max_send)
    recv_r = lax.ppermute(send_r, axis_name, perm_right)
    n_rr = lax.ppermute(jnp.array(n_sr), axis_name, perm_right)
    send_l, n_sl = _pack_selected(owned_pos, n_owned, migrate_left, max_send)
    recv_l = lax.ppermute(send_l, axis_name, perm_left)
    n_rl = lax.ppermute(jnp.array(n_sl), axis_name, perm_left)
    new_pos = stayed
    new_n = n_stay
    new_pos, new_n = _append(new_pos, new_n, recv_r, n_rr, max_owned)
    new_pos, new_n = _append(new_pos, new_n, recv_l, n_rl, max_owned)
    return (new_pos, new_n)
def estimate_nl_params(box, cutoff: float, max_local: int, capacity_mult: float=1.25):
    box_np = onp.asarray(box)
    if box_np.ndim == 2:
        side = onp.array([box_np[0, 0], box_np[1, 1], box_np[2, 2]])
    elif box_np.ndim == 1:
        side = box_np
    else:
        side = onp.full(3, float(box_np))
    cells_per_side = onp.floor(side / cutoff).astype(onp.int32)
    cells_per_side = onp.maximum(cells_per_side, 3)
    total_cells = int(onp.prod(cells_per_side))
    cell_capacity = max(int(onp.ceil(max_local / total_cells * capacity_mult)), 4)
    volume = float(onp.prod(side))
    density = max_local / volume
    sphere_vol = 4.0 / 3.0 * onp.pi * cutoff ** 3
    max_neighbors = max(int(onp.ceil(density * sphere_vol * capacity_mult)), 8)
    return (cells_per_side, cell_capacity, max_neighbors)
def ghost_exchange_subgraph(owned_pos, ghost_pos, owned_species, ghost_species, owned_global_idx, ghost_global_idx, n_owned, n_ghost, box, cutoff: float, cells_per_side, cell_capacity: int, max_neighbors: int, max_local: int, max_nodes: int, n_hops: int):
    nx, ny, nz = (int(cells_per_side[0]), int(cells_per_side[1]), int(cells_per_side[2]))
    cells_j = jnp.array([nx, ny, nz], dtype=jnp.int32)
    cutoff_sq = cutoff ** 2
    pad_node = max_nodes - 1
    N = max_local
    all_pos = jnp.zeros((N, 3), dtype=owned_pos.dtype)
    all_pos = lax.dynamic_update_slice_in_dim(all_pos, owned_pos, 0, axis=0)
    all_pos = lax.dynamic_update_slice_in_dim(all_pos, ghost_pos, n_owned, axis=0)
    all_species = jnp.zeros(N, dtype=owned_species.dtype)
    all_species = lax.dynamic_update_slice_in_dim(all_species, owned_species, 0, axis=0)
    all_species = lax.dynamic_update_slice_in_dim(all_species, ghost_species, n_owned, axis=0)
    all_gidx = jnp.zeros(N, dtype=jnp.int32)
    all_gidx = lax.dynamic_update_slice_in_dim(all_gidx, owned_global_idx, 0, axis=0)
    all_gidx = lax.dynamic_update_slice_in_dim(all_gidx, ghost_global_idx, n_owned, axis=0)
    n_all = n_owned + n_ghost
    cell_size = jnp.ones(3, dtype=owned_pos.dtype) / cells_j
    cell_count = nx * ny * nz
    idx_3d = jnp.floor(jnp.clip(all_pos, 0.0, 0.999) / cell_size).astype(jnp.int32)
    idx_3d = jnp.clip(idx_3d, 0, cells_j - 1)
    real_hashes = idx_3d[:, 0] * (ny * nz) + idx_3d[:, 1] * nz + idx_3d[:, 2]
    is_real = jnp.arange(N) < n_all
    hashes = jnp.where(is_real, real_hashes, cell_count)
    sort_map = jnp.argsort(hashes)
    sorted_hash = hashes[sort_map]
    id_buf_size = (cell_count + 1) * cell_capacity
    id_buffer_flat = N * jnp.ones(id_buf_size, dtype=jnp.int32)
    first = jnp.searchsorted(sorted_hash, sorted_hash, side='left')
    slot = jnp.arange(N, dtype=jnp.int32) - first
    flat_idx = jnp.where(slot < cell_capacity, sorted_hash * cell_capacity + slot, id_buf_size - 1)
    id_buffer_flat = id_buffer_flat.at[flat_idx].set(sort_map)
    id_buffer = id_buffer_flat[:cell_count * cell_capacity].reshape(nx, ny, nz, cell_capacity)
    real_counts = jnp.where(is_real, jnp.ones(N, dtype=jnp.int32), jnp.zeros(N, dtype=jnp.int32))
    occupancy = jax.ops.segment_sum(real_counts, jnp.where(is_real, real_hashes, 0), cell_count)
    cell_overflow = jnp.max(occupancy) > cell_capacity
    offsets = jnp.array([(dx, dy, dz) for dx in range(-1, 2) for dy in range(-1, 2) for dz in range(-1, 2)], dtype=jnp.int32)
    n_owned_static = owned_pos.shape[0]
    active = jnp.arange(N) < n_owned
    def _neighbors_of(pos_i):
        ci = jnp.floor(jnp.clip(pos_i, 0.0, 0.999) / cell_size).astype(jnp.int32)
        ci = jnp.clip(ci, 0, cells_j - 1)
        ncells = (ci[None, :] + offsets) % cells_j
        cands = id_buffer[ncells[:, 0], ncells[:, 1], ncells[:, 2]].reshape(-1)
        safe = jnp.where(cands < N, cands, 0)
        dr_frac = all_pos[safe] - pos_i
        dr_frac = dr_frac.at[:, 1].set(dr_frac[:, 1] - jnp.round(dr_frac[:, 1]))
        dr_frac = dr_frac.at[:, 2].set(dr_frac[:, 2] - jnp.round(dr_frac[:, 2]))
        dr_real = jnp.einsum('ij,...j->...i', box, dr_frac)
        dist_sq = jnp.sum(dr_real ** 2, axis=-1)
        return (cands, (dist_sq < cutoff_sq) & (cands < N) & (safe < n_all))
    def one_hop(_, active_mask):
        all_cands, all_valid = jax.vmap(_neighbors_of)(all_pos)
        masked = all_valid & active_mask[:, None]
        scatter_ids = jnp.where(masked, all_cands, N).reshape(-1)
        new_flags = jnp.zeros(N + 1, dtype=jnp.bool_)
        new_flags = new_flags.at[scatter_ids].max(masked.reshape(-1))
        return active_mask | new_flags[:N]
    active = lax.fori_loop(0, n_hops, one_hop, active)
    idx_arr = jnp.arange(N, dtype=jnp.int32)
    owned_mask = idx_arr < n_owned
    halo_mask = active & ~owned_mask
    halo_cumsum = jnp.cumsum(halo_mask)
    g2l = jnp.where(owned_mask, idx_arr, jnp.where(halo_mask, n_owned_static + halo_cumsum - 1, pad_node))
    n_halo = jnp.sum(halo_mask.astype(jnp.int32))
    n_active = n_owned + n_halo
    node_idx = jnp.zeros(max_nodes, dtype=jnp.int32)
    node_idx = node_idx.at[:n_owned_static].set(owned_global_idx)
    max_halo = max_nodes - n_owned_static - 1
    halo_local = jnp.where(halo_mask, idx_arr, N)
    halo_local_sorted = jnp.sort(halo_local)
    halo_padded = jnp.zeros(max_halo, dtype=jnp.int32)
    n_copy = min(N, max_halo)
    halo_padded = halo_padded.at[:n_copy].set(halo_local_sorted[:n_copy])
    halo_valid = halo_padded < N
    halo_gidx = jnp.where(halo_valid, all_gidx[jnp.where(halo_valid, halo_padded, 0)], 0)
    node_idx = node_idx.at[n_owned_static:n_owned_static + max_halo].set(halo_gidx)
    local_to_node = jnp.where(halo_valid, halo_padded, 0)
    species_local = jnp.zeros(max_nodes, dtype=owned_species.dtype)
    species_local = species_local.at[:n_owned_static].set(owned_species)
    halo_sp = jnp.where(halo_valid, all_species[local_to_node], 0)
    species_local = species_local.at[n_owned_static:n_owned_static + max_halo].set(halo_sp)
    species_local = jnp.where(jnp.arange(max_nodes) < n_active, species_local, 0)
    node_overflow = n_active > pad_node
    def _edges_for_atom(atom_pos, atom_idx, atom_local_idx):
        ci = jnp.floor(jnp.clip(atom_pos, 0.0, 0.999) / cell_size).astype(jnp.int32)
        ci = jnp.clip(ci, 0, cells_j - 1)
        ncells = (ci[None, :] + offsets) % cells_j
        cands = id_buffer[ncells[:, 0], ncells[:, 1], ncells[:, 2]].reshape(-1)
        safe = jnp.where(cands < N, cands, 0)
        dr_frac = all_pos[safe] - atom_pos
        dr_frac = dr_frac.at[:, 1].set(dr_frac[:, 1] - jnp.round(dr_frac[:, 1]))
        dr_frac = dr_frac.at[:, 2].set(dr_frac[:, 2] - jnp.round(dr_frac[:, 2]))
        dr_real = jnp.einsum('ij,...j->...i', box, dr_frac)
        dist_sq = jnp.sum(dr_real ** 2, axis=-1)
        sender_local = g2l[safe]
        valid = (dist_sq < cutoff_sq) & (cands < N) & (safe < n_all) & (cands != atom_idx) & (sender_local != pad_node)
        n_cands = 27 * cell_capacity
        cumsum = jnp.cumsum(valid)
        sl = jnp.where(valid, cumsum - 1, n_cands)
        buf_s = jnp.full(max_neighbors + 1, pad_node, dtype=jnp.int32)
        buf_r = jnp.full(max_neighbors + 1, pad_node, dtype=jnp.int32)
        buf_sh = jnp.zeros((max_neighbors + 1, 3), dtype=all_pos.dtype)
        clip_sl = jnp.clip(sl, 0, max_neighbors)
        buf_s = buf_s.at[clip_sl].set(jnp.where(valid, sender_local, pad_node))
        buf_r = buf_r.at[clip_sl].set(jnp.where(valid, atom_local_idx, pad_node))
        buf_sh = buf_sh.at[clip_sl].set(jnp.zeros_like(dr_frac))
        em = jnp.arange(max_neighbors) < cumsum[n_cands - 1]
        return (buf_s[:max_neighbors], buf_r[:max_neighbors], buf_sh[:max_neighbors], em, cumsum[n_cands - 1] > max_neighbors)
    owned_positions = owned_pos
    owned_indices = jnp.arange(n_owned_static, dtype=jnp.int32)
    owned_local = jnp.arange(n_owned_static, dtype=jnp.int32)
    all_s, all_r, all_sh, all_em, all_ov = jax.vmap(_edges_for_atom)(owned_positions, owned_indices, owned_local)
    max_edges = n_owned_static * max_neighbors
    senders = all_s.reshape(max_edges)
    receivers = all_r.reshape(max_edges)
    shifts = all_sh.reshape(max_edges, 3)
    edge_mask = all_em.reshape(max_edges)
    did_overflow = cell_overflow | node_overflow | jnp.any(all_ov)
    return (node_idx, senders, receivers, shifts, species_local, edge_mask, did_overflow)
