"""EAM models in exported-graph form.

Analytic Finnis-Sinclair plus tabulated DYNAMO setfl and funcfl, matching native
pair_eam interpolation. Tabulated exports use metal units; export uses n_hops=2.
"""

import gzip
from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import numpy as np


def read_table_lines(path: str) -> list[str]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as handle:
        return handle.read().splitlines()


def make_eam_energy(
    *,
    cutoff: float,
    pair_a: float = 1.0,
    dens_f0: float = 1.0,
    embed_c: float = 1.5,
    embed_eps: float = 1e-3,
    pair_embedding: float = 0.0,
    communicating: bool = False,
    half_edges: bool = False,
) -> Callable[..., Any]:
    """Build a per-atom EAM energy callable with the exported-model signature.

    Nonzero `pair_embedding` reads neighbor densities; plain single-hop exports
    of that form are wrong under decomposition with no error raised.
    """

    def node_energies(positions, species, graph, comm=None):
        del species
        dtype = positions.dtype
        cutoff_sq = jnp.asarray(cutoff * cutoff, dtype)
        zero = jnp.asarray(0.0, dtype)
        safe_senders = jnp.where(graph.edge_mask, graph.senders, 0)
        safe_receivers = jnp.where(graph.edge_mask, graph.receivers, 0)
        rij = positions[safe_receivers] - positions[safe_senders]
        r_sq = jnp.sum(rij * rij, axis=-1)
        valid = graph.edge_mask & (r_sq < cutoff_sq)
        envelope = jnp.where(valid, (1.0 - r_sq / cutoff_sq) ** 2, zero)

        n_atoms = positions.shape[0]
        zeros = jnp.zeros((n_atoms,), dtype=dtype)
        # Sender-side scatter: exact wherever neighbor rows are complete.
        pair_term = 0.5 * jnp.asarray(pair_a, dtype) * envelope
        density_term = jnp.asarray(dens_f0, dtype) * envelope
        pair_energy = zeros.at[safe_senders].add(pair_term)
        density = zeros.at[safe_senders].add(density_term)
        if half_edges:
            pair_energy = pair_energy.at[safe_receivers].add(pair_term)
            density = density.at[safe_receivers].add(density_term)

        eps = jnp.asarray(embed_eps, dtype)

        def embed(rho):
            # Shifted so sqrt stays differentiable at rho = 0.
            return -jnp.asarray(embed_c, dtype) * (jnp.sqrt(rho + eps) - jnp.sqrt(eps))

        if comm is not None:
            density = comm.forward_comm(density)

        energy = pair_energy + embed(density)
        if pair_embedding != 0.0:
            kappa = jnp.asarray(pair_embedding, dtype)
            cross = kappa * envelope * embed(density)[safe_receivers]
            energy = energy + zeros.at[safe_senders].add(cross)
            if half_edges:
                reverse = kappa * envelope * embed(density)[safe_senders]
                energy = energy + zeros.at[safe_receivers].add(reverse)
        return energy

    if communicating:
        # Positional comm: exporting without comm=True must fail, not skip.
        def comm_energy(positions, species, graph, comm):
            return node_energies(positions, species, graph, comm)

        return comm_energy

    def plain_energy(positions, species, graph):
        return node_energies(positions, species, graph)

    return plain_energy


def spline_coefficients(table: np.ndarray) -> np.ndarray:
    """Piecewise-cubic coefficients for a uniformly tabulated function.

    Reproduces `PairEAM::interpolate`; the (n, 4) result gives
    value(t) = c0 + t*(c1 + t*(c2 + t*c3)) for fraction t in each interval.
    """
    f = np.asarray(table, dtype=np.float64)
    n = f.size
    d = np.empty(n)
    d[0] = f[1] - f[0]
    d[1] = 0.5 * (f[2] - f[0])
    d[n - 2] = 0.5 * (f[n - 1] - f[n - 3])
    d[n - 1] = f[n - 1] - f[n - 2]
    m = np.arange(2, n - 2)
    d[m] = ((f[m - 2] - f[m + 2]) + 8.0 * (f[m + 1] - f[m - 1])) / 12.0
    step = np.diff(f)
    c2 = np.zeros(n)
    c3 = np.zeros(n)
    c2[:-1] = 3.0 * step - 2.0 * d[:-1] - d[1:]
    c3[:-1] = d[:-1] + d[1:] - 2.0 * step
    return np.stack([f, d, c2, c3], axis=-1)


def spline_lookup(tables, table_ids, x, delta, extrapolate=False):
    """Evaluate stacked spline tables at x, selecting a table per element.

    Out-of-range points saturate at the end value unless `extrapolate`
    continues linearly with the end slope, matching pair_eam past rhomax.
    """
    dtype = x.dtype
    n = tables.shape[-2]
    idx = x / jnp.asarray(delta, dtype)
    node = jnp.clip(jnp.floor(idx).astype(jnp.int32), 0, n - 2)
    t_raw = idx - node.astype(dtype)
    t = jnp.minimum(t_raw, jnp.asarray(1.0, dtype))
    c = tables[table_ids, node]
    value = c[..., 0] + t * (c[..., 1] + t * (c[..., 2] + t * c[..., 3]))
    if not extrapolate:
        return value
    # t_raw - t is nonzero only past the table, where c holds the end row.
    end_slope = c[..., 1] + 2.0 * c[..., 2] + 3.0 * c[..., 3]
    return value + (t_raw - t) * end_slope


def load_setfl(path: str) -> dict:
    """Parse a DYNAMO setfl (.eam.alloy) file into spline-ready tables.

    Pair tables hold r*phi, indexed by pair_index; a `.gz` suffix reads
    through gzip.
    """
    lines = read_table_lines(path)
    tokens = " ".join(lines[3:]).split()
    n_elements = int(tokens[0])
    elements = tokens[1:1 + n_elements]
    pos = 1 + n_elements
    nrho, drho, nr, dr, cutoff = tokens[pos:pos + 5]
    nrho, nr = int(nrho), int(nr)
    drho, dr, cutoff = float(drho), float(dr), float(cutoff)
    pos += 5

    masses = []
    embedding = []
    density = []
    for _ in range(n_elements):
        masses.append(float(tokens[pos + 1]))
        pos += 4
        embedding.append(spline_coefficients(
            np.array(tokens[pos:pos + nrho], dtype=np.float64)))
        pos += nrho
        density.append(spline_coefficients(
            np.array(tokens[pos:pos + nr], dtype=np.float64)))
        pos += nr

    # r*phi tables for element pairs (i, j) with j <= i, in file order.
    pair = []
    pair_index = np.zeros((n_elements, n_elements), dtype=np.int32)
    for i in range(n_elements):
        for j in range(i + 1):
            pair_index[i, j] = pair_index[j, i] = len(pair)
            pair.append(spline_coefficients(
                np.array(tokens[pos:pos + nr], dtype=np.float64)))
            pos += nr
    if pos != len(tokens):
        raise ValueError(f"{path}: {len(tokens) - pos} unparsed trailing values")

    return {
        "elements": elements,
        "masses": masses,
        "nrho": nrho,
        "drho": drho,
        "nr": nr,
        "dr": dr,
        "cutoff": cutoff,
        "embedding": np.stack(embedding),
        "density": np.stack(density),
        "pair": np.stack(pair),
        "pair_index": pair_index,
    }


FUNCFL_Z2R_SCALE = 27.2 * 0.529  # Hartree*Bohr -> eV*Angstrom, as in pair_eam.cpp


def load_funcfl(path: str) -> dict:
    """Parse a single-element DYNAMO funcfl (.eam) file, `pair_style eam`.

    The stored Z(r) becomes an r*phi table of 27.2*0.529 * Z(r)^2, so
    make_setfl_energy consumes either format. Native regrids a funcfl onto
    nr-1 nodes; splining the file grid instead differs under 2e-6 in-cutoff.
    """
    lines = read_table_lines(path)
    mass = float(lines[1].split()[1])
    tokens = " ".join(lines[2:]).split()
    nrho, nr = int(tokens[0]), int(tokens[2])
    drho, dr, cutoff = float(tokens[1]), float(tokens[3]), float(tokens[4])
    values = np.array(tokens[5:], dtype=np.float64)
    if values.size != nrho + 2 * nr:
        raise ValueError(
            f"{path}: expected {nrho + 2 * nr} table values, got {values.size}")
    z = values[nrho:nrho + nr]
    return {
        "elements": [path.rsplit("/", 1)[-1].split(".")[0]],
        "masses": [mass],
        "nrho": nrho,
        "drho": drho,
        "nr": nr,
        "dr": dr,
        "cutoff": cutoff,
        "embedding": spline_coefficients(values[:nrho])[None],
        "density": spline_coefficients(values[nrho + nr:])[None],
        "pair": spline_coefficients(FUNCFL_Z2R_SCALE * z * z)[None],
        "pair_index": np.zeros((1, 1), dtype=np.int32),
    }


def make_setfl_energy(tables: dict, *, communicating: bool = False,
                      half_edges: bool = False) -> Callable[..., Any]:
    """Per-atom eam/alloy energy for a load_setfl table set.

    LAMMPS type t must carry element t-1 of the file; distribution follows
    make_eam_energy.
    """
    cutoff = tables["cutoff"]

    def node_energies(positions, species, graph, comm=None):
        dtype = positions.dtype
        embedding = jnp.asarray(tables["embedding"], dtype)
        density_tables = jnp.asarray(tables["density"], dtype)
        pair_tables = jnp.asarray(tables["pair"], dtype)
        pair_index = jnp.asarray(tables["pair_index"])
        cutoff_sq = jnp.asarray(cutoff * cutoff, dtype)
        zero = jnp.asarray(0.0, dtype)
        one = jnp.asarray(1.0, dtype)

        safe_senders = jnp.where(graph.edge_mask, graph.senders, 0)
        safe_receivers = jnp.where(graph.edge_mask, graph.receivers, 0)
        rij = positions[safe_receivers] - positions[safe_senders]
        r_sq = jnp.sum(rij * rij, axis=-1)
        valid = graph.edge_mask & (r_sq < cutoff_sq)
        # Nonzero fallback keeps sqrt and the phi division autodiff-safe.
        r = jnp.sqrt(jnp.where(valid, r_sq, one))
        sender_species = species[safe_senders]
        receiver_species = species[safe_receivers]

        n_atoms = positions.shape[0]
        zeros = jnp.zeros((n_atoms,), dtype=dtype)
        rho_edge = spline_lookup(density_tables, receiver_species, r, tables["dr"])
        density = zeros.at[safe_senders].add(jnp.where(valid, rho_edge, zero))
        z2 = spline_lookup(pair_tables, pair_index[sender_species, receiver_species],
                           r, tables["dr"])
        pair_term = jnp.where(valid, 0.5 * z2 / r, zero)
        pair_energy = zeros.at[safe_senders].add(pair_term)
        if half_edges:
            # The reverse direction reads the other endpoint's element.
            rho_reverse = spline_lookup(density_tables, sender_species, r, tables["dr"])
            density = density.at[safe_receivers].add(jnp.where(valid, rho_reverse, zero))
            pair_energy = pair_energy.at[safe_receivers].add(pair_term)

        if comm is not None:
            density = comm.forward_comm(density)

        embed = spline_lookup(embedding, species, density, tables["drho"],
                              extrapolate=True)
        return pair_energy + embed

    if communicating:
        # Positional comm: exporting without comm=True must fail, not skip.
        def comm_energy(positions, species, graph, comm):
            return node_energies(positions, species, graph, comm)

        return comm_energy

    def plain_energy(positions, species, graph):
        return node_energies(positions, species, graph)

    return plain_energy
