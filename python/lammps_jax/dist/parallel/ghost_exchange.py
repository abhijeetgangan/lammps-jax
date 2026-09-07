"""Grid spatial decomposition with a ppermute ghost-atom exchange.

Ranks form a (px, py, pz) grid over fractional space and collect ghosts axis
by axis in the LAMMPS order, both ring directions of a hop in flight together.
ghost_exchange records the ppermute plan on a halo of cutoff plus skin, exchange_apply
replays it on positions and features until an atom has moved half the skin, and autodiff
of either is the reverse communication. Owned positions stay unwrapped between rebuilds;
redistribute wraps them, migrates atoms and maintains the geometric ownership that
pack_tiles creates. Overflow flags are rank-local; reduce them across ranks.

References:
    S. Plimpton, "Fast Parallel Algorithms for Short-Range Molecular
    Dynamics", J. Comput. Phys. 117, 1-19 (1995),
    doi:10.1006/jcph.1995.1039.
    A. P. Thompson et al., "LAMMPS - a flexible simulation tool for
    particle-based materials modeling at the atomic, meso, and continuum
    scales", Comput. Phys. Commun. 271, 108171 (2022),
    doi:10.1016/j.cpc.2021.108171.
"""

from typing import NamedTuple, Optional, Tuple

import numpy as onp
import jax
import jax.numpy as jnp
from jax import lax

from lammps_jax.dist.parallel.local_neighbor_list import (OFFSETS_27, box_widths, cell_bin_from_index,
                                                          transform_box)


class GhostExchangeConfig(NamedTuple):
    """Capacities and grid geometry for the exchange routines.

    Attributes:
        grid: (px, py, pz) ranks per fractional axis.
        max_owned: Owned rows per rank.
        max_ghost: Ghost rows per rank.
        max_send: Rows per hop.
        cutoff: Interaction cutoff; edges are built out to cutoff + skin.
        cutoff_frac: (3,) cutoff + skin over the perpendicular box widths.
        hops: Hops per direction per axis, 0 on undecomposed axes.
        tile_width: (3,) 1 / grid.
        skin: Verlet skin; a plan stays valid while no atom moves more than skin / 2.
    """

    n_ranks: int
    grid: Tuple[int, int, int]
    max_owned: int
    max_ghost: int
    max_send: int
    cutoff: float
    cutoff_frac: onp.ndarray
    hops: Tuple[int, int, int]
    tile_width: onp.ndarray
    skin: float = 0.0


def halo_fractions(grid, cutoff_frac):
    """(hops, ghost fraction, largest send fraction) of a grid for a fractional halo width."""
    grid = onp.asarray(grid, dtype=onp.int64)
    tile_width = 1.0 / grid.astype(onp.float64)
    hop_counts = onp.ceil(cutoff_frac / tile_width).astype(onp.int64)
    hop_counts += hop_counts * tile_width < cutoff_frac
    hops = tuple(int(hop_counts[axis]) if grid[axis] > 1 else 0 for axis in range(3))
    decomposed = grid > 1
    width = tile_width + 2.0 * cutoff_frac * decomposed
    ghost_fraction = float(onp.prod(width) - onp.prod(tile_width))
    send_fractions = [cutoff_frac[axis] * float(onp.prod(onp.delete(width, axis)))
                      for axis in range(3) if grid[axis] > 1]
    return (hops, ghost_fraction, max(send_fractions) if send_fractions else 0.0)


def default_grid(n_ranks: int, cutoff_frac) -> Tuple[int, int, int]:
    """The factorisation of n_ranks with the fewest ghosts, then the fewest sends and hops."""
    best = None
    for px in range(1, n_ranks + 1):
        if n_ranks % px:
            continue
        for py in range(1, n_ranks // px + 1):
            if (n_ranks // px) % py:
                continue
            grid = (px, py, n_ranks // (px * py))
            hops, ghost_fraction, send_fraction = halo_fractions(grid, cutoff_frac)
            key = (ghost_fraction, send_fraction, sum(hops))
            if best is None or key < best[0]:
                best = (key, grid)
    return best[1]


def create_config(n_ranks: int, n_atoms: int, box, cutoff: float,
                  capacity_mult: float = 1.5,
                  grid: Optional[Tuple[int, int, int]] = None,
                  skin: float = 0.0) -> GhostExchangeConfig:
    """Sizes the exchange buffers from grid geometry and mean density.

    Args:
        box: Scalar, (3,) sides, or (3, 3) lattice vectors as columns.
        capacity_mult: Safety factor on every capacity.
        grid: (px, py, pz) ranks per axis; defaults to the factorisation with the fewest ghosts.
        skin: Verlet skin added to the cutoff for the halo, the cell list and the edge list.

    Raises:
        ValueError: Grid product is not n_ranks, or cutoff + skin reaches half a box width.
    """
    widths = box_widths(box)[0]
    cutoff_frac = (cutoff + skin) / widths
    if onp.any(cutoff_frac >= 0.5):
        raise ValueError(f'cutoff {cutoff} + skin {skin} reaches half the box width on some axis; '
                         'minimum image breaks')
    if grid is None:
        grid = default_grid(n_ranks, cutoff_frac)
    grid = tuple(int(ranks) for ranks in grid)
    if int(onp.prod(grid)) != n_ranks:
        raise ValueError(f'grid {grid} does not multiply to n_ranks {n_ranks}')
    tile_width = 1.0 / onp.asarray(grid, dtype=onp.float64)
    hops, ghost_fraction, send_fraction = halo_fractions(grid, cutoff_frac)
    max_owned = int(onp.ceil(n_atoms / n_ranks * capacity_mult))
    max_ghost = max(int(onp.ceil(n_atoms * ghost_fraction * capacity_mult)), 8)
    max_send = max(int(onp.ceil(n_atoms * send_fraction * capacity_mult)), 4)
    return GhostExchangeConfig(n_ranks=n_ranks, grid=grid, max_owned=max_owned,
                               max_ghost=max_ghost, max_send=max_send,
                               cutoff=cutoff, cutoff_frac=cutoff_frac,
                               hops=hops, tile_width=tile_width, skin=float(skin))


def local_extent(config: GhostExchangeConfig) -> onp.ndarray:
    """(3,) fractional extent of the tile plus its halo on decomposed axes, 1 on the others."""
    decomposed = onp.asarray(config.grid) > 1
    return onp.where(decomposed, config.tile_width + 2.0 * config.cutoff_frac, 1.0)


def subgraph_params(config: GhostExchangeConfig, box, n_atoms: int, capacity_mult: float = 1.25):
    """Tile-local cell grid and capacities for ghost_exchange_subgraph from the mean density.

    Returns:
        (cells_per_side (3,) int32, cell_capacity, max_neighbors, max_edges); cells span the tile
        plus its halo on decomposed axes, each at least one reach wide, and the whole periodic box
        on the others with at least three cells so the stencil covers the axis without revisiting
        a cell; max_edges holds a tile full to max_owned or to capacity_mult times the mean.
    """
    volume = box_widths(box)[1]
    reach = config.cutoff + config.skin
    extent = local_extent(config)
    cells_per_side = onp.floor(extent / config.cutoff_frac).astype(onp.int32)
    periodic = onp.asarray(config.grid) == 1
    cells_per_side = onp.where(periodic, onp.maximum(cells_per_side, 3), onp.maximum(cells_per_side, 1))
    density = n_atoms / volume
    cell_volume = volume * float(onp.prod(extent / cells_per_side))
    cell_capacity = max(int(onp.ceil(density * cell_volume * capacity_mult)), 4)
    sphere_vol = 4.0 / 3.0 * onp.pi * reach ** 3
    max_neighbors = max(int(onp.ceil(density * sphere_vol * capacity_mult)), 8)
    owned_bound = max(n_atoms / config.n_ranks * capacity_mult, config.max_owned)
    max_edges = max(int(onp.ceil(owned_bound * density * sphere_vol)), max_neighbors)
    return (cells_per_side, cell_capacity, max_neighbors, max_edges)


def pack_tiles(R_frac, config: GhostExchangeConfig, ints=None):
    """Packs global atoms into the per-rank tile buffers that ghost_exchange requires.

    Args:
        R_frac: (N, 3) fractional positions; tile membership is decided in this dtype, so hand the
            buffers to ghost_exchange in the same dtype or call redistribute first.
        ints: Optional (N, K) integer columns, for example species and global id.

    Returns:
        (owned (n_ranks, max_owned, 3), owned_ints (n_ranks, max_owned, K) or None,
        counts (n_ranks,) int32, rank_of (N,)); dead rows are zero.

    Raises:
        ValueError: A tile holds more than max_owned atoms.
    """
    R = onp.asarray(R_frac) % 1.0
    R = onp.where(R >= 1.0, 0.0, R).astype(R.dtype)
    grid = config.grid
    rank_of = onp.zeros(R.shape[0], dtype=onp.int64)
    for axis in range(3):
        coord = onp.clip(onp.floor(R[:, axis] * grid[axis]).astype(onp.int64), 0, grid[axis] - 1)
        rank_of = rank_of * grid[axis] + coord
    counts = onp.bincount(rank_of, minlength=config.n_ranks).astype(onp.int32)
    if counts.max() > config.max_owned:
        raise ValueError(f'a tile holds {counts.max()} atoms, more than max_owned {config.max_owned}')
    order = onp.argsort(rank_of, kind='stable')
    sorted_rank = rank_of[order]
    slot = onp.arange(R.shape[0]) - onp.searchsorted(sorted_rank, sorted_rank, side='left')
    owned = onp.zeros((config.n_ranks, config.max_owned, 3), dtype=R.dtype)
    owned[sorted_rank, slot] = R[order]
    owned_ints = None
    if ints is not None:
        ints = onp.asarray(ints)
        owned_ints = onp.zeros((config.n_ranks, config.max_owned) + ints.shape[1:], dtype=onp.int32)
        owned_ints[sorted_rank, slot] = ints[order]
    return (owned, owned_ints, counts, rank_of)


def ring_perm(grid, axis: int, direction: int):
    """ppermute pairs, one ring step along an axis of the (px, py, pz) grid."""
    px, py, pz = grid
    pairs = []
    for rank in range(px * py * pz):
        coords = [rank // (py * pz), (rank // pz) % py, rank % pz]
        coords[axis] = (coords[axis] + direction) % grid[axis]
        pairs.append((rank, coords[0] * py * pz + coords[1] * pz + coords[2]))
    return pairs


def rank_coords(config: GhostExchangeConfig, axis_name: str):
    """This rank's (px, py, pz) grid coordinates as three scalars."""
    grid = config.grid
    rank = lax.axis_index(axis_name)
    return (rank // (grid[1] * grid[2]), (rank // grid[2]) % grid[1], rank % grid[2])


def pack_selected(arr, n_valid, mask, max_out):
    """Packs the masked live rows of arr into (max_out, ...); returns (buffer, unclipped count)."""
    full_mask = mask & (jnp.arange(arr.shape[0]) < n_valid)
    cumsum = jnp.cumsum(full_mask, dtype=jnp.int32)
    n_selected = cumsum[arr.shape[0] - 1]
    slot = jnp.where(full_mask, cumsum - 1, arr.shape[0])
    buffer = jnp.zeros((max_out + 1,) + arr.shape[1:], dtype=arr.dtype)
    clip_slot = jnp.clip(slot, 0, max_out)
    buffer = buffer.at[clip_slot].set(jnp.where(full_mask.reshape(-1, *[1] * (arr.ndim - 1)), arr, 0))
    return (buffer[:max_out], n_selected)


def append_rows(buffer, n_buffer, new_data, n_new, max_buffer):
    """Appends n_new rows of new_data (rows, ...) at row n_buffer of buffer; rows past max_buffer are dropped."""
    n_to_copy = jnp.minimum(n_new, max_buffer - n_buffer)
    idx = jnp.arange(new_data.shape[0])
    target = n_buffer + idx
    mask = (idx < n_to_copy) & (target < max_buffer)
    buffer = buffer.at[jnp.where(mask, target, buffer.shape[0] + idx)].set(new_data, mode='drop',
                                                                          unique_indices=True)
    return (buffer, n_buffer + n_to_copy)


class ExchangeStage(NamedTuple):
    """One ppermute stage recorded by ghost_exchange and replayed by exchange_apply.

    Attributes:
        send_idx: (max_send,) int32 pool rows sent.
        n_send: Live rows sent.
        n_recv: Live rows received.
        offset: Pool row where the arrivals were appended.
        shift: Seam unwrap added to the stage axis of arriving positions.
    """

    send_idx: jnp.ndarray
    n_send: jnp.ndarray
    n_recv: jnp.ndarray
    offset: jnp.ndarray
    shift: jnp.ndarray


def stage_order(config: GhostExchangeConfig):
    """(axis, direction, ppermute pairs) per stage in ghost_exchange order: axis, hop, direction."""
    return [(axis, direction, ring_perm(config.grid, axis, direction))
            for axis in range(3) if config.grid[axis] > 1
            for hop in range(config.hops[axis])
            for direction in (+1, -1)]


def send_stage(pool, send_idx, n_send, axis, shift, permutation, axis_name):
    """ppermutes n_send pool rows at send_idx; a non-None shift is added to column axis of arrivals."""
    valid = jnp.arange(send_idx.shape[0]) < n_send
    send_buf = jnp.where(valid[:, None], pool[jnp.where(valid, send_idx, 0)], 0)
    recv_buf = lax.ppermute(send_buf, axis_name, permutation)
    if shift is None:
        return recv_buf
    return recv_buf.at[:, axis].add(shift.astype(recv_buf.dtype))


def ghost_exchange(owned_data, n_owned, config: GhostExchangeConfig, axis_name: str):
    """Collects ghost atoms from neighboring tiles via per-axis ppermute rings.

    Args:
        owned_data: (rows, C) rows with fractional positions in the first three columns; live
            rows must lie in this rank's tile as pack_tiles and redistribute place them.
        n_owned: Live owned count.

    Returns:
        (ghost_data (max_ghost, C), n_ghost, flags (2,) bool [misplaced, overflow], plan);
        ghost_data is differentiable in owned_data and plan is a tuple of ExchangeStage.
        Integer columns travel through exchange_apply(plan, ints) instead.
    """
    grid = config.grid
    max_ghost = config.max_ghost
    max_send = config.max_send
    coords = rank_coords(config, axis_name)
    rows, n_cols = owned_data.shape
    pool_rows = rows + max_ghost
    pool = jnp.zeros((pool_rows, n_cols), dtype=owned_data.dtype)
    pool = lax.dynamic_update_slice_in_dim(pool, owned_data, 0, axis=0)
    pool_index = jnp.arange(pool_rows, dtype=jnp.int32)
    n_owned = jnp.asarray(n_owned, jnp.int32)
    n_pool = n_owned
    misplaced = jnp.bool_(False)
    overflow = jnp.bool_(False)
    plan = []
    for axis in range(3):
        if grid[axis] == 1:
            continue
        tile_width = float(config.tile_width[axis])
        cutoff_frac = float(config.cutoff_frac[axis])
        tile_lo = coords[axis] * tile_width
        tile_hi = tile_lo + tile_width
        tile_of = jnp.floor(pool[:, axis] * grid[axis]).astype(jnp.int32)
        misplaced = misplaced | jnp.any((pool_index < n_owned) & (tile_of != coords[axis]))
        windows = {direction: (jnp.int32(0), n_pool) for direction in (+1, -1)}
        for hop in range(config.hops[axis]):
            received = []
            for direction in (+1, -1):
                candidate_start, n_candidates = windows[direction]
                in_range = (pool_index >= candidate_start) & (pool_index < candidate_start + n_candidates)
                if direction == +1:
                    in_halo = pool[:, axis] >= tile_hi - cutoff_frac
                else:
                    in_halo = pool[:, axis] <= tile_lo + cutoff_frac
                send_idx, n_send = pack_selected(pool_index, pool_rows, in_range & in_halo, max_send)
                overflow = overflow | (n_send > max_send)
                n_send = jnp.minimum(n_send, max_send)
                permutation = ring_perm(grid, axis, direction)
                seam_rank = 0 if direction == +1 else grid[axis] - 1
                shift = jnp.where(coords[axis] == seam_rank, -direction, 0).astype(owned_data.dtype)
                recv_buf = send_stage(pool, send_idx, n_send, axis, shift, permutation, axis_name)
                n_recv = lax.ppermute(n_send, axis_name, permutation)
                received.append((direction, send_idx, n_send, recv_buf, n_recv, shift))
            for direction, send_idx, n_send, recv_buf, n_recv, shift in received:
                overflow = overflow | (n_pool - n_owned + n_recv > max_ghost)
                plan.append(ExchangeStage(send_idx, n_send, n_recv, n_pool, shift))
                windows[direction] = (n_pool, n_recv)
                pool, n_pool = append_rows(pool, n_pool, recv_buf, n_recv, pool_rows)
    ghost_data = lax.dynamic_slice_in_dim(pool, n_owned, max_ghost, axis=0)
    flags = jnp.stack([misplaced, overflow])
    return (ghost_data, jnp.minimum(n_pool - n_owned, max_ghost), flags, tuple(plan))


def exchange_apply(plan, owned_data, config: GhostExchangeConfig, axis_name: str,
                   unwrap: bool = False):
    """Replays a recorded exchange on new per-row data.

    Args:
        plan: Stages from ghost_exchange on the same atoms, no more than skin / 2 of motion ago.
        owned_data: (rows, C) per-row data of any dtype; positions must not have been wrapped
            since the plan was recorded.
        unwrap: Add the seam shifts to arriving rows; True when the columns are fractional positions.

    Returns:
        (rows + max_ghost + 1, C) in node order: live owned rows, ghosts in arrival order, unused
        slots, pad row; index it only through senders and receivers of ghost_exchange_subgraph.

    Raises:
        TypeError: unwrap on integer columns.
    """
    if unwrap and not jnp.issubdtype(owned_data.dtype, jnp.floating):
        raise TypeError('unwrap=True adds seam shifts to positions; replay integer columns with '
                        'unwrap=False')
    rows, n_cols = owned_data.shape
    pool_rows = rows + config.max_ghost
    pool = jnp.zeros((pool_rows + 1, n_cols), dtype=owned_data.dtype)
    pool = lax.dynamic_update_slice_in_dim(pool, owned_data, 0, axis=0)
    stages = list(zip(stage_order(config), plan, strict=True))
    for pair in range(0, len(stages), 2):
        received = []
        for (axis, direction, permutation), stage in stages[pair:pair + 2]:
            recv_buf = send_stage(pool, stage.send_idx, stage.n_send, axis,
                                  stage.shift if unwrap else None, permutation, axis_name)
            received.append((stage, recv_buf))
        for stage, recv_buf in received:
            pool = append_rows(pool, stage.offset, recv_buf, stage.n_recv, pool_rows)[0]
    return pool


def needs_rebuild(owned_pos, ref_pos, n_owned, box, config: GhostExchangeConfig, axis_name: str):
    """True on every rank once a live atom has moved more than skin / 2 since ref_pos.

    A wrapped coordinate counts as a move of one box length, so wrapping forces a rebuild.

    Args:
        owned_pos: (rows, 3) current fractional positions.
        ref_pos: (rows, 3) fractional positions the plan was recorded on.
        box: (3,) sides or (3, 3) lattice vectors as columns.
    """
    dr = owned_pos[:, :3] - ref_pos[:, :3]
    live = jnp.arange(owned_pos.shape[0]) < n_owned
    dist = jnp.sqrt(jnp.sum(transform_box(jnp.asarray(box), dr) ** 2, axis=-1))
    moved = jnp.max(jnp.where(live, dist, 0.0))
    return lax.pmax(moved, axis_name) > 0.5 * config.skin


def redistribute(owned_data, n_owned, config: GhostExchangeConfig, axis_name: str, owned_ints=None):
    """Migrates atoms one tile per axis to the rank whose tile contains them.

    Args:
        owned_data: (max_owned, C) rows with fractional positions in the first three columns.
        n_owned: Live owned count.
        owned_ints: Optional (max_owned, K) integer columns that travel with their rows.

    Returns:
        (new_data (max_owned, C), new_ints or None, n_live, flags (3,) bool): a live atom more
        than one tile away stayed behind, a send buffer overflowed, the owned buffer overflowed.
    """
    grid = config.grid
    max_owned = owned_data.shape[0]
    max_send = config.max_send
    coords = rank_coords(config, axis_name)
    wrapped = owned_data[:, :3] % 1.0
    buffer = owned_data.at[:, :3].set(jnp.where(wrapped >= 1.0, 0.0, wrapped))
    ints = owned_ints
    n_live = jnp.asarray(n_owned, jnp.int32)
    misplaced = jnp.bool_(False)
    send_overflow = jnp.bool_(False)
    owned_overflow = jnp.bool_(False)
    for axis in range(3):
        if grid[axis] == 1:
            continue
        n_tiles = grid[axis]
        target = jnp.clip(jnp.floor(buffer[:, axis] * n_tiles).astype(jnp.int32), 0, n_tiles - 1)
        delta = (target - coords[axis]) % n_tiles
        live = jnp.arange(max_owned) < n_live
        misplaced = misplaced | jnp.any(live & (delta != 0) & (delta != 1) & (delta != n_tiles - 1))
        right_mask = delta == 1
        left_mask = (delta == n_tiles - 1) & (delta != 1)
        stay_mask = ~right_mask & ~left_mask
        packed = []
        for mask, capacity in ((stay_mask, max_owned), (right_mask, max_send), (left_mask, max_send)):
            data, count = pack_selected(buffer, n_live, mask, capacity)
            packed_ints = pack_selected(ints, n_live, mask, capacity)[0] if ints is not None else None
            packed.append((data, packed_ints, count))
        (stayed, stayed_ints, n_stay), (right, right_ints, n_right), (left, left_ints, n_left) = packed
        send_overflow = send_overflow | (n_right > max_send) | (n_left > max_send)
        n_right = jnp.minimum(n_right, max_send)
        n_left = jnp.minimum(n_left, max_send)
        received = []
        for direction, data, packed_ints, count in ((+1, right, right_ints, n_right), (-1, left, left_ints, n_left)):
            permutation = ring_perm(grid, axis, direction)
            recv_ints = lax.ppermute(packed_ints, axis_name, permutation) if packed_ints is not None else None
            received.append((lax.ppermute(data, axis_name, permutation), recv_ints,
                             lax.ppermute(count, axis_name, permutation)))
        owned_overflow = owned_overflow | (n_stay + received[0][2] + received[1][2] > max_owned)
        buffer, ints, n_live = (stayed, stayed_ints, n_stay)
        for data, recv_ints, count in received:
            if ints is not None:
                ints = append_rows(ints, n_live, recv_ints, count, max_owned)[0]
            buffer, n_live = append_rows(buffer, n_live, data, count, max_owned)
    return (buffer, ints, n_live, jnp.stack([misplaced, send_overflow, owned_overflow]))


def ghost_exchange_subgraph(owned_pos, ghost_pos, owned_species, ghost_species,
                            owned_global_idx, ghost_global_idx, n_owned, n_ghost,
                            box, cells_per_side, cell_capacity: int, max_neighbors: int,
                            max_edges: int, config: GhostExchangeConfig, axis_name: str):
    """Edges into owned atoms over the node order of exchange_apply, out to cutoff + skin.

    Args:
        owned_pos: (rows, 3) fractional positions in this rank's tile; dead rows produce no edges.
        ghost_pos: (max_ghost, 3) unwrapped fractional positions.
        owned_species: (rows,) species indices.
        ghost_species: (max_ghost,) species indices.
        owned_global_idx: (rows,) global atom ids.
        ghost_global_idx: (max_ghost,) global atom ids.
        box: (3, 3) lattice vectors as columns.
        cells_per_side: (3,) tile-local cell grid from subgraph_params.
        max_neighbors: Candidate capacity per owned atom while the stencil runs.
        max_edges: Capacity E of the compacted edge list, from subgraph_params.
        config: Supplies the cutoff, skin and grid.

    Returns:
        (node_idx (rows + max_ghost + 1,) int32, senders (E,), receivers (E,), species
        (rows + max_ghost + 1,), edge_mask (E,) bool, did_overflow); edges are compacted in
        owned-row order and padded with the pad node, which is last.

    Raises:
        ValueError: cells_per_side is finer than the reach on an axis the stencil does not span.
    """
    cells_np = onp.asarray(cells_per_side, dtype=onp.int64)
    periodic_np = onp.asarray(config.grid) == 1
    spans_axis = onp.where(periodic_np, cells_np == 3, cells_np <= 2)
    too_fine = ~spans_axis & (local_extent(config) / cells_np < config.cutoff_frac * (1 - 1e-9))
    if onp.any(periodic_np & (cells_np < 3)) or onp.any(too_fine):
        raise ValueError(f'cells_per_side {cells_np.tolist()} is finer than cutoff + skin or has '
                         'fewer than three cells on a periodic axis; use subgraph_params')
    rows = owned_pos.shape[0]
    N = rows + ghost_pos.shape[0]
    dtype = owned_pos.dtype
    cells = jnp.asarray(cells_per_side, dtype=jnp.int32)
    reach_sq = (config.cutoff + config.skin) ** 2
    periodic = jnp.asarray([ranks == 1 for ranks in config.grid])
    wrap = periodic.astype(dtype)
    coords = jnp.stack(rank_coords(config, axis_name)).astype(dtype)
    origin = jnp.where(periodic, 0.0, coords * jnp.asarray(config.tile_width, dtype)
                       - jnp.asarray(config.cutoff_frac, dtype))
    extent = jnp.asarray(local_extent(config), dtype)

    def cell_of(pos):
        """Cell coordinates (..., 3) of positions in the tile frame; periodic only where grid is 1."""
        u = jnp.where(periodic, pos % 1.0, (pos - origin) / extent)
        idx = jnp.floor(u * cells).astype(jnp.int32)
        return jnp.where(periodic, idx % cells, jnp.clip(idx, 0, cells - 1))

    all_pos = jnp.zeros((N, 3), dtype=dtype)
    all_pos = lax.dynamic_update_slice_in_dim(all_pos, owned_pos, 0, axis=0)
    all_pos = lax.dynamic_update_slice_in_dim(all_pos, ghost_pos, n_owned, axis=0)
    live = jnp.arange(N + 1) < n_owned + n_ghost
    node_idx = jnp.zeros(N + 1, dtype=jnp.int32).at[:rows].set(owned_global_idx.astype(jnp.int32))
    node_idx = lax.dynamic_update_slice_in_dim(node_idx, ghost_global_idx.astype(jnp.int32), n_owned, axis=0)
    species = jnp.zeros(N + 1, dtype=owned_species.dtype).at[:rows].set(owned_species)
    species = lax.dynamic_update_slice_in_dim(species, ghost_species.astype(owned_species.dtype), n_owned, axis=0)
    node_idx = jnp.where(live, node_idx, 0)
    species = jnp.where(live, species, 0)
    id_buffer, cell_overflow = cell_bin_from_index(cell_of(all_pos), n_owned + n_ghost, cells_per_side, cell_capacity)
    offsets = jnp.array(OFFSETS_27, dtype=jnp.int32)
    n_candidates = 27 * cell_capacity

    def edges_for_atom(atom_pos, atom_idx):
        """Edges from local atoms into one owned (3,) receiver, padded with N."""
        raw_cells = cell_of(atom_pos)[None, :] + offsets
        in_grid = jnp.all(periodic | ((raw_cells >= 0) & (raw_cells < cells)), axis=1)
        neighbor_cells = jnp.where(periodic, raw_cells % cells, jnp.clip(raw_cells, 0, cells - 1))
        candidates = id_buffer[neighbor_cells[:, 0], neighbor_cells[:, 1], neighbor_cells[:, 2]]
        candidates = jnp.where(in_grid[:, None], candidates, N).reshape(-1)
        safe_ids = jnp.where(candidates < N, candidates, 0)
        dr_frac = all_pos[safe_ids] - atom_pos
        dr_frac = dr_frac - jnp.round(dr_frac) * wrap
        dist_sq = jnp.sum(jnp.einsum('ij,...j->...i', box, dr_frac) ** 2, axis=-1)
        valid = (dist_sq < reach_sq) & (candidates < N) & (candidates != atom_idx) & (atom_idx < n_owned)
        cumsum = jnp.cumsum(valid)
        slot = jnp.clip(jnp.where(valid, cumsum - 1, n_candidates), 0, max_neighbors)
        sender_buf = jnp.full(max_neighbors + 1, N, dtype=jnp.int32)
        receiver_buf = jnp.full(max_neighbors + 1, N, dtype=jnp.int32)
        sender_buf = sender_buf.at[slot].set(jnp.where(valid, candidates, N))
        receiver_buf = receiver_buf.at[slot].set(jnp.where(valid, atom_idx, N))
        n_found = cumsum[n_candidates - 1]
        edge_mask = jnp.arange(max_neighbors) < n_found
        return (sender_buf[:max_neighbors], receiver_buf[:max_neighbors], edge_mask, n_found > max_neighbors)

    owned_indices = jnp.arange(rows, dtype=jnp.int32)
    senders_all, receivers_all, edge_mask_all, overflow_all = jax.vmap(edges_for_atom)(owned_pos, owned_indices)
    valid = edge_mask_all.reshape(-1)
    cumsum = jnp.cumsum(valid, dtype=jnp.int32)
    n_edges = cumsum[-1]
    dropped = max_edges + valid.shape[0] + jnp.arange(valid.shape[0], dtype=jnp.int32)
    slot = jnp.where(valid, cumsum - 1, dropped)
    senders = jnp.full(max_edges, N, dtype=jnp.int32).at[slot].set(senders_all.reshape(-1), mode='drop',
                                                                  unique_indices=True)
    receivers = jnp.full(max_edges, N, dtype=jnp.int32).at[slot].set(receivers_all.reshape(-1),
                                                                    mode='drop', unique_indices=True)
    edge_mask = jnp.arange(max_edges) < n_edges
    did_overflow = cell_overflow | jnp.any(overflow_all) | (n_edges > max_edges)
    return (node_idx, senders, receivers, species, edge_mask, did_overflow)
