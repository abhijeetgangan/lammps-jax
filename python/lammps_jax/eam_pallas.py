"""Tabulated EAM kernels in Pallas, float32 only, over edges or over the LAMMPS list itself.

Edge kernels run one edge per lane over the compacted half edges and accumulate both endpoints
through atomic adds. Row kernels run one atom per lane over the slot-major neighbor matrix,
looping over the row's slots with the row's sums in registers and only the neighbor side through
atomic adds, as pair_eam_kokkos does. Padding indexes max_atoms and drops out through the masks.
"""

import contextlib
import inspect
from collections.abc import Callable
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

from lammps_jax.eam import spline_lookup


def pair_row(si, sj):
    """Row of the stacked pair table for species si and sj, lower triangle in file order."""
    hi = jnp.maximum(si, sj)
    lo = jnp.minimum(si, sj)
    return ((hi * (hi + 1)) >> 1) + lo


def spline(coeffs_ref, table, x, inv_delta, n):
    """Value and slope of stacked spline rows at x, saturating past the table like spline_lookup."""
    idx = x * inv_delta
    node = jnp.clip(jnp.floor(idx).astype(jnp.int32), 0, n - 2)
    t = jnp.minimum(idx - node.astype(jnp.float32), 1.0)
    row = table * n + node
    c0, c1, c2, c3 = (coeffs_ref[row, k] for k in range(4))
    value = c0 + t * (c1 + t * (c2 + t * c3))
    slope = (c1 + t * (2.0 * c2 + t * (3.0 * c3))) * inv_delta
    return value, slope


@contextlib.contextmanager
def relaxed_atomics():
    """Lower Pallas atomic adds with relaxed ordering while exporting, as Kokkos::atomic_add does.

    Pallas has no public knob and emits acquire-release atomics, which order like fences and
    ran 2.5x slower here; unordered accumulation needs none of that. Wrap the export call.
    """
    from jax._src.pallas.triton import lowering
    from jaxlib.triton import dialect as tt

    original = lowering._atomic_rmw
    expected = ["op", "ptr", "val", "mask", "semantic", "sync_scope"]
    if list(inspect.signature(original).parameters) != expected:
        raise RuntimeError("jax changed the Pallas Triton atomic lowering; update relaxed_atomics")

    def relaxed(op, ptr, val, mask=None, semantic=None, sync_scope=tt.MemSyncScope.GPU):
        del semantic
        return original(op, ptr, val, mask=mask, semantic=tt.MemSemantic.RELAXED,
                        sync_scope=sync_scope)

    lowering._atomic_rmw = relaxed
    try:
        yield
    finally:
        lowering._atomic_rmw = original


def scatter_add(ref, index, value, valid, interpret):
    """Masked atomic add; the interpreter lacks masks, so it adds zeros at clamped indices.

    The interpreter also keeps only the last of repeated indices within one call, clamped zeros
    included, so interpret runs need distinct indices per program.
    """
    if interpret:
        plgpu.atomic_add(ref, index, jnp.where(valid, value, 0.0))
    else:
        plgpu.atomic_add(ref, index, value, mask=valid)


def edge_geometry(s_ref, r_ref, pos_ref, block, n_edges, n_atoms, cutoff_sq):
    """Endpoints, pair vector, distance, and mask of this program's edge block."""
    e = pl.program_id(0) * block + jnp.arange(block, dtype=jnp.int32)
    esafe = jnp.minimum(e, n_edges - 1)
    i = s_ref[esafe]
    j = r_ref[esafe]
    valid = (e < n_edges) & (i < n_atoms) & (j < n_atoms)
    i = jnp.where(valid, i, 0)
    j = jnp.where(valid, j, 0)
    dx = pos_ref[j, 0] - pos_ref[i, 0]
    dy = pos_ref[j, 1] - pos_ref[i, 1]
    dz = pos_ref[j, 2] - pos_ref[i, 2]
    r_sq = dx * dx + dy * dy + dz * dz
    valid = valid & (r_sq < cutoff_sq)
    r = jnp.sqrt(jnp.where(valid, r_sq, 1.0))
    return i, j, (dx, dy, dz), r, 1.0 / r, valid


def density_kernel(s_ref, r_ref, pos_ref, sp_ref, dens_ref, _acc_ref, rho_ref, *, block, n_edges,
                   n_atoms, cutoff_sq, inv_dr, nr, interpret):
    i, j, _rij, r, _rinv, valid = edge_geometry(s_ref, r_ref, pos_ref, block, n_edges, n_atoms,
                                                cutoff_sq)
    si = sp_ref[i]
    sj = sp_ref[j]
    # Each endpoint receives the other element's density.
    rho_i, _ = spline(dens_ref, sj, r, inv_dr, nr)
    rho_j, _ = spline(dens_ref, si, r, inv_dr, nr)
    scatter_add(rho_ref, (i,), rho_i, valid, interpret)
    scatter_add(rho_ref, (j,), rho_j, valid, interpret)


def force_kernel(s_ref, r_ref, pos_ref, sp_ref, fp_ref, dens_ref, pair_ref, _acc_ref, f_ref, *,
                 block, n_edges, n_atoms, cutoff_sq, inv_dr, nr, interpret):
    i, j, (dx, dy, dz), r, rinv, valid = edge_geometry(s_ref, r_ref, pos_ref, block, n_edges,
                                                       n_atoms, cutoff_sq)
    si = sp_ref[i]
    sj = sp_ref[j]
    _, rhojp = spline(dens_ref, sj, r, inv_dr, nr)
    _, rhoip = spline(dens_ref, si, r, inv_dr, nr)
    z2, z2p = spline(pair_ref, pair_row(si, sj), r, inv_dr, nr)
    phi = z2 * rinv
    phip = (z2p - phi) * rinv
    psip = fp_ref[i] * rhojp + fp_ref[j] * rhoip + phip
    scale = jnp.where(valid, psip * rinv, 0.0)
    for k, d in enumerate((dx, dy, dz)):
        scatter_add(f_ref, (i, k), scale * d, valid, interpret)
        scatter_add(f_ref, (j, k), -scale * d, valid, interpret)


def block_launch(kernel, operands, accumulators, n_items, *, block, num_warps, interpret, name):
    """Run kernel over blocks of items, accumulating into donated arrays aliased to the outputs."""
    operands = (*operands, *accumulators)
    full = [pl.BlockSpec(op.shape, lambda e, nd=op.ndim: (0,) * nd) for op in operands]
    n_acc = len(accumulators)
    return pl.pallas_call(
        kernel,
        out_shape=tuple(jax.ShapeDtypeStruct(acc.shape, acc.dtype) for acc in accumulators),
        grid=(pl.cdiv(n_items, block),),
        in_specs=full,
        out_specs=tuple(full[-n_acc:]),
        input_output_aliases={len(operands) - n_acc + k: k for k in range(n_acc)},
        compiler_params=plgpu.CompilerParams(num_warps=num_warps),
        interpret=interpret,
        name=name,
    )(*operands)


def program_rows(num_ref, pos_ref, sp_ref, block):
    """Rows of this program: clamped index, validity, list count, position, and species."""
    a = pl.program_id(0) * block + jnp.arange(block, dtype=jnp.int32)
    valid = a < num_ref.shape[0]
    a = jnp.where(valid, a, 0)
    count = jnp.where(valid, num_ref[a], 0)
    xyz = tuple(pos_ref[a, k] for k in range(3))
    return a, valid, count, xyz, sp_ref[a]


def list_pair(nb_ref, pos_ref, sp_ref, s, a, count, xyz, n_atoms, cutoff_sq):
    """Slot s of each row: neighbor, its species, pair vector, distance, and validity."""
    valid = s < count
    j = nb_ref[s, a]
    valid = valid & (j < n_atoms)
    j = jnp.where(valid, j, 0)
    dx = pos_ref[j, 0] - xyz[0]
    dy = pos_ref[j, 1] - xyz[1]
    dz = pos_ref[j, 2] - xyz[2]
    r_sq = dx * dx + dy * dy + dz * dz
    valid = valid & (r_sq < cutoff_sq)
    r = jnp.sqrt(jnp.where(valid, r_sq, 1.0))
    return j, sp_ref[j], (dx, dy, dz), r, valid


def program_max(num_neighbors, block, max_neighbors):
    """Widest row of each program, at most the slot count; an in-kernel reduction stalls warps."""
    padded = jnp.pad(num_neighbors, (0, -num_neighbors.shape[0] % block))
    return jnp.minimum(jnp.max(padded.reshape(-1, block), axis=1), max_neighbors)


def row_density_kernel(bmax_ref, num_ref, nb_ref, pos_ref, sp_ref, dens_ref, _acc_ref, rho_ref, *,
                       block, n_atoms, cutoff_sq, inv_dr, nr, half, interpret):
    a, avalid, count, xyz, si = program_rows(num_ref, pos_ref, sp_ref, block)

    def body(s, rho_i):
        j, sj, _rij, r, valid = list_pair(nb_ref, pos_ref, sp_ref, s, a, count, xyz, n_atoms,
                                          cutoff_sq)
        rho_to_i, _ = spline(dens_ref, sj, r, inv_dr, nr)
        if half:
            # A half list holds each pair once, so the neighbor takes its share from this row.
            rho_to_j, _ = spline(dens_ref, si, r, inv_dr, nr)
            scatter_add(rho_ref, (j,), rho_to_j, valid, interpret)
        return rho_i + jnp.where(valid, rho_to_i, 0.0)

    bound = bmax_ref[pl.program_id(0)]
    rho_i = lax.fori_loop(0, bound, body, jnp.zeros((block,), jnp.float32))
    scatter_add(rho_ref, (a,), rho_i, avalid, interpret)


def row_energy_kernel(bmax_ref, num_ref, nb_ref, pos_ref, sp_ref, dens_ref, pair_ref, _rho_acc,
                      _e_acc, rho_ref, e_ref, *, block, n_atoms, cutoff_sq, inv_dr, nr, half,
                      interpret):
    a, avalid, count, xyz, si = program_rows(num_ref, pos_ref, sp_ref, block)

    def body(s, carry):
        rho_i, e_i = carry
        j, sj, _rij, r, valid = list_pair(nb_ref, pos_ref, sp_ref, s, a, count, xyz, n_atoms,
                                          cutoff_sq)
        rho_to_i, _ = spline(dens_ref, sj, r, inv_dr, nr)
        z2, _ = spline(pair_ref, pair_row(si, sj), r, inv_dr, nr)
        if half:
            rho_to_j, _ = spline(dens_ref, si, r, inv_dr, nr)
            scatter_add(rho_ref, (j,), rho_to_j, valid, interpret)
        # A half list holds each pair once, so the row carries the whole phi; a full list twice.
        phi = z2 / r if half else 0.5 * z2 / r
        return rho_i + jnp.where(valid, rho_to_i, 0.0), e_i + jnp.where(valid, phi, 0.0)

    zeros = jnp.zeros((block,), jnp.float32)
    rho_i, e_i = lax.fori_loop(0, bmax_ref[pl.program_id(0)], body, (zeros, zeros))
    scatter_add(rho_ref, (a,), rho_i, avalid, interpret)
    scatter_add(e_ref, (a,), e_i, avalid, interpret)


def row_force_kernel(bmax_ref, num_ref, nb_ref, pos_ref, sp_ref, fp_ref, dens_ref, pair_ref,
                     _acc_ref, f_ref, *, block, n_atoms, cutoff_sq, inv_dr, nr, half,
                     interpret):
    a, avalid, count, xyz, si = program_rows(num_ref, pos_ref, sp_ref, block)
    fp_i = fp_ref[a]

    def body(s, f_i):
        j, sj, rij, r, valid = list_pair(nb_ref, pos_ref, sp_ref, s, a, count, xyz, n_atoms,
                                         cutoff_sq)
        rinv = 1.0 / r
        _, rhojp = spline(dens_ref, sj, r, inv_dr, nr)
        _, rhoip = spline(dens_ref, si, r, inv_dr, nr)
        z2, z2p = spline(pair_ref, pair_row(si, sj), r, inv_dr, nr)
        phi = z2 * rinv
        phip = (z2p - phi) * rinv
        psip = fp_i * rhojp + fp_ref[j] * rhoip + phip
        scale = jnp.where(valid, psip * rinv, 0.0)
        if half:
            for k, d in enumerate(rij):
                scatter_add(f_ref, (j, k), -scale * d, valid, interpret)
        return tuple(f + scale * d for f, d in zip(f_i, rij))

    zeros = jnp.zeros((block,), jnp.float32)
    f_i = lax.fori_loop(0, bmax_ref[pl.program_id(0)], body, (zeros, zeros, zeros))
    for k in range(3):
        scatter_add(f_ref, (a, k), f_i[k], avalid, interpret)


def check_launch_shape(block, num_warps):
    """Triton needs power-of-two program shapes."""
    for name, value in (("block", block), ("num_warps", num_warps)):
        if value <= 0 or value & (value - 1):
            raise ValueError(f"{name} must be a power of two, got {value}")


def kernel_tables(tables):
    """Static kernel parameters and the float32 coefficient rows of a load_setfl table set."""
    common = dict(cutoff_sq=float(tables["cutoff"]) ** 2, inv_dr=1.0 / float(tables["dr"]),
                  nr=int(tables["nr"]))
    rows = {name: np.asarray(tables[name], np.float32).reshape(-1, 4)
            for name in ("density", "pair")}
    return common, rows, jnp.asarray(tables["embedding"], jnp.float32)


def check_float32(positions):
    if positions.dtype != jnp.float32:
        raise ValueError("the Pallas EAM kernels are float32; export float64 bundles with "
                         "make_setfl_edge_force")


def make_setfl_pallas_force(tables: dict, *, block: int = 1024, num_warps: int = 4,
                            interpret: bool = False) -> Callable[..., Any]:
    """Per-atom eam/alloy forces from two Pallas launches over a half edge list, float32 only.

    The communicating export completes densities through the exchange between the launches,
    as make_setfl_edge_force does; both endpoints of every edge receive their force, so the
    bundle exports with force_output atom-force, half_edges and newton on, inside
    relaxed_atomics(). block and num_warps must be powers of two (Triton).
    """
    check_launch_shape(block, num_warps)
    static, rows, embedding = kernel_tables(tables)
    launch = partial(block_launch, block=block, num_warps=num_warps, interpret=interpret)

    def forces(positions, species, graph, comm):
        check_float32(positions)
        n_atoms, n_edges = positions.shape[0], graph.senders.shape[0]
        species = species.astype(jnp.int32)
        dens = jnp.asarray(rows["density"])
        common = dict(block=block, n_edges=n_edges, n_atoms=n_atoms, interpret=interpret, **static)
        (rho,) = launch(partial(density_kernel, **common),
                        (graph.senders, graph.receivers, positions, species, dens),
                        (jnp.zeros((n_atoms,), jnp.float32),), n_edges, name="eam_density")
        # Owners complete on reverse; ghost fp needs the forward.
        rho = comm.forward_comm(comm.reverse_comm(rho))
        fp = spline_lookup(embedding, species, rho, tables["drho"], extrapolate=True,
                           derivative=True)
        (f,) = launch(partial(force_kernel, **common),
                      (graph.senders, graph.receivers, positions, species, fp, dens,
                       jnp.asarray(rows["pair"])),
                      (jnp.zeros((n_atoms, 3), jnp.float32),), n_edges, name="eam_forces")
        return f

    return forces


def make_setfl_pallas_row_force(tables: dict, *, block: int = 512, num_warps: int = 16,
                                half_edges: bool = True,
                                interpret: bool = False) -> Callable[..., Any]:
    """Per-atom eam/alloy forces from two Pallas launches over the LAMMPS list, float32 only.

    Each lane owns a row of the slot-major neighbor matrix and loops over its slots with the row
    accumulating in registers. On a half list neighbors take their share through atomic adds
    and the reverse exchange completes owned densities; on a full list (half_edges False) rows
    are complete and only the forward exchange runs. Export with max_neighbors and comm, with
    half_edges and newton on for the half list or newton off for the full list, inside
    relaxed_atomics(); block and num_warps must be powers of two. A program loops to its
    widest row, which XLA finds beforehand.
    """
    check_launch_shape(block, num_warps)
    static, rows, embedding = kernel_tables(tables)
    launch = partial(block_launch, block=block, num_warps=num_warps, interpret=interpret)

    def forces(positions, species, graph, comm):
        check_float32(positions)
        n_atoms, n_rows = positions.shape[0], graph.num_neighbors.shape[0]
        species = species.astype(jnp.int32)
        dens = jnp.asarray(rows["density"])
        common = dict(block=block, n_atoms=n_atoms, half=half_edges, interpret=interpret, **static)
        bmax = program_max(graph.num_neighbors, block, graph.neighbors.shape[0])
        (rho,) = launch(partial(row_density_kernel, **common),
                        (bmax, graph.num_neighbors, graph.neighbors, positions, species, dens),
                        (jnp.zeros((n_atoms,), jnp.float32),), n_rows, name="eam_row_density")
        # Half lists leave neighbor shares on ghost rows for their owners; ghost fp needs forward.
        rho = comm.forward_comm(comm.reverse_comm(rho) if half_edges else rho)
        fp = spline_lookup(embedding, species, rho, tables["drho"], extrapolate=True,
                           derivative=True)
        (f,) = launch(partial(row_force_kernel, **common),
                      (bmax, graph.num_neighbors, graph.neighbors, positions, species, fp, dens,
                       jnp.asarray(rows["pair"])),
                      (jnp.zeros((n_atoms, 3), jnp.float32),), n_rows, name="eam_row_forces")
        return f

    return forces


def make_setfl_pallas_row_energy(tables: dict, *, block: int = 512, num_warps: int = 16,
                                 half_edges: bool = True,
                                 interpret: bool = False) -> Callable[..., Any]:
    """Per-atom eam/alloy energies from one Pallas launch over the LAMMPS list, float32 only.

    Rows carry their embedding and the pair energy of their list entries, whole on a half list
    and halved on a full one. Half lists leave partial densities on ghost rows that the reverse
    exchange folds into their owners; full rows need no exchange. Only owned rows count.
    """
    check_launch_shape(block, num_warps)
    static, rows, embedding = kernel_tables(tables)
    launch = partial(block_launch, block=block, num_warps=num_warps, interpret=interpret)

    def energies(positions, species, graph, comm):
        check_float32(positions)
        n_atoms, n_rows = positions.shape[0], graph.num_neighbors.shape[0]
        species = species.astype(jnp.int32)
        common = dict(block=block, n_atoms=n_atoms, half=half_edges, interpret=interpret, **static)
        rho, pair_energy = launch(
            partial(row_energy_kernel, **common),
            (program_max(graph.num_neighbors, block, graph.neighbors.shape[0]),
             graph.num_neighbors, graph.neighbors, positions, species,
             jnp.asarray(rows["density"]), jnp.asarray(rows["pair"])),
            (jnp.zeros((n_atoms,), jnp.float32), jnp.zeros((n_rows,), jnp.float32)), n_rows,
            name="eam_row_energy")
        if half_edges:
            rho = comm.reverse_comm(rho)
        embed = spline_lookup(embedding, species, rho, tables["drho"], extrapolate=True)
        return embed.at[:n_rows].add(pair_energy)

    return energies
