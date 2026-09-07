"""Cell-list neighbor list and halo subgraph over wrapped fractional coordinates.

Each device owns the index block [owned_start, owned_start + n_per) and rebuilds
its neighbor list or n_hops halo subgraph every step from replicated (N, 3)
fractional positions in [0, 1) with minimum image on all three axes. Edge
convention: displacement = pos[receiver] - (pos[sender] + transform_box(box, shift)).
"""

from typing import Tuple

import numpy as onp
import jax
import jax.numpy as jnp
from jax import lax

OFFSETS_27 = [(dx, dy, dz) for dx in range(-1, 2) for dy in range(-1, 2) for dz in range(-1, 2)]


def box_widths(box) -> Tuple[onp.ndarray, float]:
    """Perpendicular widths (3,) and volume of a scalar, (3,) sides, or (3, 3) box."""
    box_np = onp.asarray(box)
    if box_np.ndim == 2:
        normals = [onp.cross(box_np[:, (k + 1) % 3], box_np[:, (k + 2) % 3]) for k in range(3)]
        volume = float(abs(onp.dot(box_np[:, 0], normals[0])))
        widths = onp.array([volume / onp.linalg.norm(normal) for normal in normals])
        return (widths, volume)
    if box_np.ndim == 1:
        side = box_np.astype(onp.float64)
    else:
        side = onp.full(3, float(box_np))
    return (side, float(onp.prod(side)))


def estimate_cell_params(box, cutoff: float, n_atoms: int,
                         capacity_mult: float = 1.25) -> Tuple[onp.ndarray, int, int]:
    """Cell grid and capacities from mean density.

    Args:
        box: Scalar, (3,) sides, or (3, 3) lattice vectors as columns.
        capacity_mult: Safety factor on both capacities.

    Returns:
        cells_per_side (3,) int32, cell_capacity, max_neighbors.
    """
    widths, volume = box_widths(box)
    cells_per_side = onp.floor(widths / cutoff).astype(onp.int32)
    cells_per_side = onp.maximum(cells_per_side, 3)
    total_cells = int(onp.prod(cells_per_side))
    atoms_per_cell = n_atoms / total_cells
    cell_capacity = max(int(onp.ceil(atoms_per_cell * capacity_mult)), 4)
    density = n_atoms / volume
    sphere_vol = 4.0 / 3.0 * onp.pi * cutoff ** 3
    max_neighbors = max(int(onp.ceil(density * sphere_vol * capacity_mult)), 8)
    return (cells_per_side, cell_capacity, max_neighbors)


def cell_index(pos, cells_per_side):
    """Cell coordinates (..., 3) of fractional positions, periodic in a (3,) grid."""
    cells = jnp.asarray(cells_per_side, dtype=jnp.int32)
    return jnp.floor(pos * cells).astype(jnp.int32) % cells


def cell_bin(pos, n_live, cells_per_side, cell_capacity):
    """Bins the first n_live rows of pos (rows, 3) into a periodic cell list.

    Args:
        cells_per_side: (3,) host-side cell grid; static under jit.

    Returns:
        id_buffer (nx, ny, nz, cell_capacity) int32 row indices padded with rows, did_overflow.
    """
    return cell_bin_from_index(cell_index(pos, cells_per_side), n_live, cells_per_side, cell_capacity)


def cell_bin_from_index(idx, n_live, cells_per_side, cell_capacity):
    """cell_bin on precomputed (rows, 3) int32 cell coordinates."""
    nx, ny, nz = (int(cells_per_side[0]), int(cells_per_side[1]), int(cells_per_side[2]))
    rows = idx.shape[0]
    cell_count = nx * ny * nz
    live = jnp.arange(rows) < n_live
    cell_hashes = idx[:, 0] * (ny * nz) + idx[:, 1] * nz + idx[:, 2]
    hashes = jnp.where(live, cell_hashes, cell_count)
    sort_map = jnp.argsort(hashes).astype(jnp.int32)
    sorted_hash = hashes[sort_map]
    first = jnp.searchsorted(sorted_hash, sorted_hash, side='left')
    slot = jnp.arange(rows, dtype=jnp.int32) - first
    buffer_size = (cell_count + 1) * cell_capacity
    flat_idx = jnp.where(slot < cell_capacity, sorted_hash * cell_capacity + slot, buffer_size - 1)
    id_buffer = jnp.full(buffer_size, rows, dtype=jnp.int32).at[flat_idx].set(sort_map)
    id_buffer = id_buffer[:cell_count * cell_capacity].reshape(nx, ny, nz, cell_capacity)
    occupancy = jax.ops.segment_sum(live.astype(jnp.int32), jnp.where(live, cell_hashes, 0), cell_count)
    return (id_buffer, jnp.max(occupancy) > cell_capacity)


def cells_too_fine(box, cells_per_side, cutoff: float):
    """True when a cell is narrower than the cutoff on an axis the 27-cell stencil does not span."""
    box = jnp.asarray(box)
    if box.ndim == 2:
        normals = jnp.stack([jnp.cross(box[:, (k + 1) % 3], box[:, (k + 2) % 3]) for k in range(3)])
        widths = jnp.abs(jnp.dot(box[:, 0], normals[0])) / jnp.linalg.norm(normals, axis=1)
    else:
        widths = jnp.broadcast_to(box, (3,))
    cells = jnp.asarray(cells_per_side)
    return jnp.any((cells < 3) | ((cells > 3) & (widths / cells < cutoff)))


def transform_box(box, dr):
    """Fractional (..., 3) dr to real space as box @ dr for a (3,) or (3, 3) box."""
    if box.ndim == 1:
        return dr * box
    return jnp.einsum('ij,...j->...i', box, dr)


def local_neighbor_list(R_frac, box, cutoff: float, owned_start, n_per: int,
                        cells_per_side, cell_capacity: int, max_neighbors: int):
    """Neighbor list of the owned index block in global indices.

    Args:
        R_frac: (N, 3) wrapped fractional positions.
        box: (3,) sides or (3, 3) lattice vectors.
        owned_start: First owned global index, static or traced.
        n_per: Owned block length, static.
        cells_per_side: (3,) cell grid.
        max_neighbors: Neighbor capacity per owned atom.

    Returns:
        neighbor_idx (2, n_per * max_neighbors) int32 [senders; receivers] padded
        with N, did_overflow; the flag also fires when cells_too_fine.
    """
    N = R_frac.shape[0]
    cells = jnp.asarray(cells_per_side, dtype=jnp.int32)
    cutoff_sq = cutoff ** 2
    id_buffer, cell_overflow = cell_bin(R_frac, N, cells_per_side, cell_capacity)
    offsets = jnp.array(OFFSETS_27, dtype=jnp.int32)
    n_candidates = 27 * cell_capacity

    def search_one_atom(atom_pos, atom_global_idx):
        """Receivers within cutoff of one owned (3,) position, padded with N."""
        neighbor_cells = (cell_index(atom_pos, cells)[None, :] + offsets) % cells
        candidates = id_buffer[neighbor_cells[:, 0], neighbor_cells[:, 1], neighbor_cells[:, 2]].reshape(-1)
        safe_ids = jnp.where(candidates < N, candidates, 0)
        dr = R_frac[safe_ids] - atom_pos
        dr = dr - jnp.round(dr)
        dist_sq = jnp.sum(transform_box(box, dr) ** 2, axis=-1)
        valid = (dist_sq < cutoff_sq) & (candidates < N) & (candidates != atom_global_idx)
        cumsum = jnp.cumsum(valid)
        slot = jnp.where(valid, cumsum - 1, n_candidates)
        receivers = jnp.full(max_neighbors + 1, N, dtype=jnp.int32)
        receivers = receivers.at[jnp.clip(slot, 0, max_neighbors)].set(jnp.where(valid, candidates, N))
        n_found = cumsum[-1]
        return (receivers[:max_neighbors], n_found > max_neighbors)

    owned_global_indices = owned_start + jnp.arange(n_per, dtype=jnp.int32)
    owned_positions = lax.dynamic_slice_in_dim(R_frac, owned_start, n_per, axis=0)
    receivers_all, atom_overflows = jax.vmap(search_one_atom)(owned_positions, owned_global_indices)
    senders_all = jnp.broadcast_to(owned_global_indices[:, None], (n_per, max_neighbors))
    neighbor_idx = jnp.stack([senders_all.reshape(-1), receivers_all.reshape(-1)])
    did_overflow = cell_overflow | jnp.any(atom_overflows) | cells_too_fine(box, cells_per_side, cutoff)
    return (neighbor_idx, did_overflow)


def estimate_subgraph_params(box, cutoff: float, n_atoms: int, n_per: int, n_hops: int,
                             capacity_mult: float = 1.25) -> Tuple[onp.ndarray, int, int, int]:
    """Cell params plus node capacity with the halo modeled as a cube shell.

    Args:
        box: Scalar, (3,) sides, or (3, 3) lattice vectors.
        n_hops: Halo depth in cutoffs.

    Returns:
        cells_per_side (3,) int32, cell_capacity, max_nodes, max_edges_per_atom.
    """
    cells_per_side, cell_capacity, max_neighbors = estimate_cell_params(box, cutoff, n_atoms, capacity_mult)
    volume = box_widths(box)[1]
    density = n_atoms / volume
    own_volume = n_per / density
    own_side = own_volume ** (1.0 / 3.0)
    halo_radius = n_hops * cutoff
    shell_volume = (own_side + 2.0 * halo_radius) ** 3 - own_volume
    n_halo_est = min(n_atoms - n_per, int(onp.ceil(density * shell_volume * capacity_mult)))
    max_nodes = n_per + n_halo_est + 1
    return (cells_per_side, cell_capacity, max_nodes, max_neighbors)


def local_subgraph(R_frac, species, box, cutoff: float, owned_start, n_per: int,
                   n_hops: int, cells_per_side, cell_capacity: int, max_nodes: int,
                   max_edges_per_atom: int):
    """Halo subgraph in local indices; shared arguments as in local_neighbor_list.

    Args:
        species: (N,) global species indices.
        n_hops: Halo flood depth; edges end at owned receivers only, so 1 suffices.
        max_nodes: Node capacity with pad_node = max_nodes - 1.
        max_edges_per_atom: Edge capacity per owned receiver, E = n_per * max_edges_per_atom.

    Returns:
        node_idx (max_nodes,) int32, senders and receivers (E,) int32 padded with
        pad_node, shifts (E, 3), species (max_nodes,), edge_mask (E,) bool, did_overflow;
        the flag also fires when cells_too_fine.
    """
    N = R_frac.shape[0]
    cells = jnp.asarray(cells_per_side, dtype=jnp.int32)
    cutoff_sq = cutoff ** 2
    pad_node = max_nodes - 1
    max_edges = n_per * max_edges_per_atom
    id_buffer, cell_overflow = cell_bin(R_frac, N, cells_per_side, cell_capacity)
    offsets = jnp.array(OFFSETS_27, dtype=jnp.int32)
    n_candidates = 27 * cell_capacity

    def stencil(atom_pos):
        """Stencil candidates, safe ids, and fractional dr of one (3,) position."""
        neighbor_cells = (cell_index(atom_pos, cells)[None, :] + offsets) % cells
        candidates = id_buffer[neighbor_cells[:, 0], neighbor_cells[:, 1], neighbor_cells[:, 2]].reshape(-1)
        safe_ids = jnp.where(candidates < N, candidates, 0)
        return (candidates, safe_ids, R_frac[safe_ids] - atom_pos)

    def one_hop(hop, active_mask):
        """Adds every in-range neighbor of an active atom to the (N,) mask."""
        all_candidates, safe_ids, dr = jax.vmap(stencil)(R_frac)
        dist_sq = jnp.sum(transform_box(box, dr - jnp.round(dr)) ** 2, axis=-1)
        masked = (dist_sq < cutoff_sq) & (all_candidates < N) & active_mask[:, None]
        scatter_ids = jnp.where(masked, all_candidates, N).reshape(-1)
        new_flags = jnp.zeros(N + 1, dtype=jnp.bool_)
        new_flags = new_flags.at[scatter_ids].max(masked.reshape(-1))
        return active_mask | new_flags[:N]

    atom_indices = jnp.arange(N, dtype=jnp.int32)
    owned_mask = (atom_indices >= owned_start) & (atom_indices < owned_start + n_per)
    active_mask = lax.fori_loop(0, n_hops, one_hop, owned_mask)
    halo_mask = active_mask & ~owned_mask
    halo_cumsum = jnp.cumsum(halo_mask).astype(jnp.int32)
    global_to_local = jnp.where(owned_mask, atom_indices - owned_start, jnp.where(halo_mask, n_per + halo_cumsum - 1, pad_node))
    n_active = n_per + jnp.sum(halo_mask.astype(jnp.int32))
    max_halo = max_nodes - n_per - 1
    halo_sorted = jnp.sort(jnp.where(halo_mask, atom_indices, N))[:max_halo]
    node_idx = jnp.zeros(max_nodes, dtype=jnp.int32)
    node_idx = node_idx.at[:n_per].set(owned_start + jnp.arange(n_per, dtype=jnp.int32))
    node_idx = node_idx.at[n_per:n_per + max_halo].set(jnp.where(halo_sorted < N, halo_sorted, 0))
    species_local = jnp.where(jnp.arange(max_nodes) < n_active, species[node_idx], 0)
    node_overflow = n_active > pad_node

    def edges_for_owned_atom(atom_pos, atom_global_idx, atom_local_idx):
        """Edges from active local atoms into one owned (3,) receiver."""
        candidates, safe_ids, dr_frac = stencil(atom_pos)
        shift = jnp.round(dr_frac)
        dist_sq = jnp.sum(transform_box(box, dr_frac - shift) ** 2, axis=-1)
        sender_local = global_to_local[safe_ids]
        valid = (dist_sq < cutoff_sq) & (candidates < N) & (candidates != atom_global_idx) & (sender_local != pad_node)
        cumsum = jnp.cumsum(valid)
        slot = jnp.clip(jnp.where(valid, cumsum - 1, n_candidates), 0, max_edges_per_atom)
        sender_buf = jnp.full(max_edges_per_atom + 1, pad_node, dtype=jnp.int32)
        receiver_buf = jnp.full(max_edges_per_atom + 1, pad_node, dtype=jnp.int32)
        shift_buf = jnp.zeros((max_edges_per_atom + 1, 3), dtype=R_frac.dtype)
        sender_buf = sender_buf.at[slot].set(jnp.where(valid, sender_local, pad_node))
        receiver_buf = receiver_buf.at[slot].set(jnp.where(valid, atom_local_idx, pad_node))
        shift_buf = shift_buf.at[slot].set(jnp.where(valid[:, None], -shift, 0.0))
        n_found = cumsum[-1]
        edge_mask = jnp.arange(max_edges_per_atom) < n_found
        return (sender_buf[:max_edges_per_atom], receiver_buf[:max_edges_per_atom],
                shift_buf[:max_edges_per_atom], edge_mask, n_found > max_edges_per_atom)

    owned_pos = lax.dynamic_slice_in_dim(R_frac, owned_start, n_per, axis=0)
    owned_global = owned_start + jnp.arange(n_per, dtype=jnp.int32)
    owned_local = jnp.arange(n_per, dtype=jnp.int32)
    senders_all, receivers_all, shifts_all, edge_mask_all, overflow_all = jax.vmap(
        edges_for_owned_atom)(owned_pos, owned_global, owned_local)
    senders = senders_all.reshape(max_edges)
    receivers = receivers_all.reshape(max_edges)
    shifts = shifts_all.reshape(max_edges, 3)
    edge_mask = edge_mask_all.reshape(max_edges)
    did_overflow = (cell_overflow | node_overflow | jnp.any(overflow_all)
                    | cells_too_fine(box, cells_per_side, cutoff))
    return (node_idx, senders, receivers, shifts, species_local, edge_mask, did_overflow)
