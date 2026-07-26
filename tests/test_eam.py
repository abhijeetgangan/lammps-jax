"""EAM through the ghost bundle path, checked against a dense reference.

EAM needs the n_hops=2 contract: F(rho) couples an atom's force to its neighbors'
neighborhoods, and both single-hop conventions get forces wrong.
"""

import contextlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from helpers import (
    call_model,
    edges_within_cutoff,
    export_and_load,
    native_interpolate,
    random_system,
    two_domains,
)
from lammps_jax.eam import (
    FUNCFL_Z2R_SCALE,
    load_funcfl,
    load_setfl,
    make_eam_energy,
    make_setfl_energy,
    spline_coefficients,
    spline_lookup,
)
from lammps_jax.export import (
    DISTRIBUTED_BUNDLE_FORMAT,
    LammpsNeighborList,
    program_text,
    wrap_energy_fn,
)

CUTOFF = 1.0
PARAMS = dict(pair_a=1.0, dens_f0=1.0, embed_c=1.5, embed_eps=1e-3)
MAX_ATOMS, MAX_EDGES = 128, 4096


def dense_energy(positions):
    """Same physics as lammps_jax.eam, written densely with no graph/scatter."""
    delta = positions[None, :, :] - positions[:, None, :]
    r_sq = jnp.sum(delta * delta, axis=-1)
    off_diagonal = ~jnp.eye(positions.shape[0], dtype=bool)
    within = (r_sq < CUTOFF**2) & off_diagonal
    envelope = jnp.where(within, (1.0 - r_sq / CUTOFF**2) ** 2, 0.0)
    pair_energy = 0.5 * PARAMS["pair_a"] * jnp.sum(envelope, axis=1)
    density = PARAMS["dens_f0"] * jnp.sum(envelope, axis=1)
    embedding = -PARAMS["embed_c"] * (
        jnp.sqrt(density + PARAMS["embed_eps"]) - jnp.sqrt(PARAMS["embed_eps"])
    )
    return jnp.sum(pair_energy + embedding)


@pytest.fixture(scope="module")
def dense_reference():
    """Shared 96-atom system with dense reference energy and forces."""
    positions_np = random_system(seed=11)
    ref_energy, ref_grad = jax.value_and_grad(dense_energy)(jnp.asarray(positions_np))
    return positions_np, float(ref_energy), -np.asarray(ref_grad)


@pytest.fixture(scope="module")
def single_hop_fused():
    """Default n_hops=1 fused wrapper, shared by the wrongness tests."""
    _force, _energy, fused_fn = wrap_energy_fn(
        make_eam_energy(cutoff=CUTOFF, **PARAMS), max_atoms=MAX_ATOMS, call_model=call_model
    )
    return fused_fn


def abi_args(positions_np, n_owned, senders, receivers):
    """Pad one rank's graph to the fixed-capacity ABI the pair style packs."""
    n_present = len(positions_np)
    positions = np.zeros((MAX_ATOMS, 3), dtype=np.float32)
    positions[:n_present] = positions_np
    padded_senders = np.full(MAX_EDGES, MAX_ATOMS, dtype=np.int32)
    padded_receivers = np.full(MAX_EDGES, MAX_ATOMS, dtype=np.int32)
    edge_mask = np.zeros(MAX_EDGES, dtype=bool)
    padded_senders[: len(senders)] = senders
    padded_receivers[: len(receivers)] = receivers
    edge_mask[: len(senders)] = True
    return (
        jnp.asarray(positions),
        jnp.zeros((MAX_ATOMS,), dtype=jnp.int32),
        jnp.int32(n_owned),
        jnp.int32(n_present - n_owned),
        jnp.asarray(padded_senders),
        jnp.asarray(padded_receivers),
        jnp.asarray(edge_mask),
    )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_eam_graph_matches_dense(dtype):
    """The dtype-polymorphic model computes at the positions dtype."""
    float64 = dtype == "float64"
    positions_np = random_system(seed=11, n_atoms=48, box=(3.0, 2.0, 2.0)).astype(dtype)
    senders, receivers = edges_within_cutoff(positions_np, CUTOFF)

    with jax.enable_x64(True) if float64 else contextlib.nullcontext():
        positions = jnp.asarray(positions_np)
        graph = LammpsNeighborList(
            senders=jnp.asarray(senders),
            receivers=jnp.asarray(receivers),
            edge_mask=jnp.ones(len(senders), dtype=bool),
        )
        eam_energy = make_eam_energy(cutoff=CUTOFF, **PARAMS)
        assert eam_energy(positions, None, graph).dtype == positions.dtype

        def graph_total(pos):
            return jnp.sum(eam_energy(pos, None, graph))

        energy, grad = jax.value_and_grad(graph_total)(positions)
        ref_energy, ref_grad = jax.value_and_grad(dense_energy)(positions)

    rel, atol = (1e-12, 1e-12) if float64 else (1e-5, 1e-4)
    assert float(energy) == pytest.approx(float(ref_energy), rel=rel)
    assert np.allclose(np.asarray(grad), np.asarray(ref_grad), atol=atol)


def test_eam_ghost_decomposition_matches_dense(dense_reference):
    """Emulates the n_hops=2 pair-style path through the export wrappers."""
    positions_np, ref_energy, ref_forces = dense_reference
    _force, energy_fn, fused_fn = wrap_energy_fn(
        make_eam_energy(cutoff=CUTOFF, **PARAMS),
        max_atoms=MAX_ATOMS,
        call_model=call_model,
        owned_rows_only=True,
    )

    total_energy = 0.0
    total_forces = np.zeros_like(positions_np, dtype=np.float64)
    for present, n_owned in two_domains(positions_np, x_split=3.0, halo=2 * CUTOFF):
        local_positions = positions_np[present]
        senders, receivers = edges_within_cutoff(local_positions, CUTOFF)
        args = abi_args(local_positions, n_owned, senders, receivers)
        energy, forces = fused_fn(*args)
        assert float(energy_fn(*args)) == pytest.approx(float(energy), abs=1e-5)
        total_energy += float(energy)
        np.add.at(total_forces, present, np.asarray(forces[: len(present)], dtype=np.float64))

    assert total_energy == pytest.approx(ref_energy, abs=1e-3)
    assert np.allclose(total_forces, ref_forces, atol=1e-3)


def test_eam_newton_off_singlehop_gets_forces_wrong(dense_reference, single_hop_fused):
    """Single-hop newton-off fails for EAM.

    Full lists keep densities and the masked energy exact, but dropped ghost
    force rows lose embedding forces on the other domain's atoms.
    """
    positions_np, ref_energy, ref_forces = dense_reference

    total_energy = 0.0
    total_forces = np.zeros_like(positions_np, dtype=np.float64)
    for present, n_owned in two_domains(positions_np, x_split=3.0, halo=CUTOFF):
        local_positions = positions_np[present]
        senders, receivers = edges_within_cutoff(local_positions, CUTOFF)
        owned_rows = senders < n_owned
        args = abi_args(local_positions, n_owned, senders[owned_rows], receivers[owned_rows])
        energy, forces = single_hop_fused(*args)
        total_energy += float(energy)
        owned_ids = present[:n_owned]
        np.add.at(total_forces, owned_ids, np.asarray(forces[:n_owned], dtype=np.float64))

    assert total_energy == pytest.approx(ref_energy, abs=1e-3)
    assert not np.allclose(total_forces, ref_forces, atol=1e-3)


def test_eam_default_single_hop_export_is_wrong_multi_rank(dense_reference, single_hop_fused):
    """EAM exported with the default n_hops=1 is silently wrong on two ranks.

    Each rank embeds a partial density for boundary atoms and F(rho) is
    nonlinear, so EAM bundles carry n_hops=2.
    """
    positions_np, ref_energy, ref_forces = dense_reference
    n_atoms = len(positions_np)

    x_split = 3.0
    owner = (positions_np[:, 0] >= x_split).astype(np.int32)
    senders, receivers = edges_within_cutoff(positions_np, CUTOFF)
    once = senders < receivers
    pair_a, pair_b = senders[once], receivers[once]
    assigned = owner[pair_a]

    total_energy = 0.0
    total_forces = np.zeros_like(positions_np, dtype=np.float64)
    domains = enumerate(two_domains(positions_np, x_split=x_split, halo=CUTOFF))
    for rank, (present, n_owned) in domains:
        global_to_local = np.full(n_atoms, -1, dtype=np.int32)
        global_to_local[present] = np.arange(len(present), dtype=np.int32)
        mine = assigned == rank
        local_a = global_to_local[pair_a[mine]]
        local_b = global_to_local[pair_b[mine]]
        assert (local_a >= 0).all() and (local_b >= 0).all()
        rank_senders = np.concatenate([local_a, local_b])
        rank_receivers = np.concatenate([local_b, local_a])
        args = abi_args(positions_np[present], n_owned, rank_senders, rank_receivers)
        energy, forces = single_hop_fused(*args)
        total_energy += float(energy)
        np.add.at(total_forces, present, np.asarray(forces[: len(present)], dtype=np.float64))

    assert total_energy != pytest.approx(ref_energy, abs=1e-3)
    assert not np.allclose(total_forces, ref_forces, atol=1e-3)


def synthetic_setfl(path, nrho=12, nr=10):
    """Write a tiny two-element setfl file; returns the raw tables by section name."""
    drho, dr, cutoff = 0.5, 0.25, 2.0
    rho_grid = np.arange(nrho) * drho
    r_grid = np.arange(nr) * dr
    tables = {
        "F_A": -np.sqrt(rho_grid + 0.1), "F_B": 0.3 * rho_grid**2,
        "rho_A": np.exp(-r_grid), "rho_B": 2.0 * np.exp(-2.0 * r_grid),
        "z2r_AA": r_grid * np.exp(-r_grid),
        "z2r_BA": 0.5 * r_grid * np.exp(-r_grid),
        "z2r_BB": 0.25 * r_grid * np.exp(-2.0 * r_grid),
    }
    lines = ["c1", "c2", "c3", "2 A B", f"{nrho} {drho} {nr} {dr} {cutoff}"]
    for element, mass in (("A", 10.0), ("B", 20.0)):
        lines.append(f"1 {mass} 1.0 fcc")
        lines += [str(v) for v in tables[f"F_{element}"]]
        lines += [str(v) for v in tables[f"rho_{element}"]]
    for pair in ("z2r_AA", "z2r_BA", "z2r_BB"):
        lines += [str(v) for v in tables[pair]]
    path.write_text("\n".join(lines) + "\n")
    return tables


def test_load_setfl_parses_tables_and_pairs(tmp_path):
    path = tmp_path / "tiny.eam.alloy"
    raw = synthetic_setfl(path)
    tables = load_setfl(str(path))
    assert tables["elements"] == ["A", "B"]
    assert tables["masses"] == [10.0, 20.0]
    assert tables["cutoff"] == 2.0
    np.testing.assert_array_equal(tables["embedding"][0, :, 0], raw["F_A"])
    np.testing.assert_array_equal(tables["density"][1, :, 0], raw["rho_B"])
    np.testing.assert_array_equal(tables["pair"][1, :, 0], raw["z2r_BA"])
    np.testing.assert_array_equal(tables["pair_index"], [[0, 1], [1, 2]])


def test_load_funcfl_converts_charge_to_pair(tmp_path):
    path = tmp_path / "tiny.eam"
    nrho, nr = 8, 6
    embed = np.linspace(0.0, -2.0, nrho)
    z = np.linspace(2.0, 0.0, nr)
    rho = np.linspace(1.0, 0.0, nr)
    lines = ["comment", "29 63.55 3.615 fcc", f"{nrho} 0.5 {nr} 0.25 1.5"]
    lines += [str(v) for v in np.concatenate([embed, z, rho])]
    path.write_text("\n".join(lines) + "\n")
    tables = load_funcfl(str(path))
    assert tables["elements"] == ["tiny"]
    assert tables["masses"] == [63.55]
    np.testing.assert_array_equal(tables["embedding"][0, :, 0], embed)
    np.testing.assert_array_equal(tables["density"][0, :, 0], rho)
    np.testing.assert_allclose(tables["pair"][0, :, 0], FUNCFL_Z2R_SCALE * z * z)


def test_setfl_energy_matches_dense_tables(tmp_path):
    """Graph scatter/gather and per-edge table selection against a dense float64 loop.

    The spline scheme itself is validated against native pair_style eam/alloy
    via examples/in.eam_cuzr's backend switch.
    """
    path = tmp_path / "tiny.eam.alloy"
    synthetic_setfl(path)
    tables = load_setfl(str(path))
    positions_np = random_system(seed=23, n_atoms=32, box=(3.0, 2.0, 2.0))
    species_np = np.random.default_rng(29).integers(0, 2, size=32)
    senders, receivers = edges_within_cutoff(positions_np, tables["cutoff"])
    graph = LammpsNeighborList(
        senders=jnp.asarray(senders),
        receivers=jnp.asarray(receivers),
        edge_mask=jnp.ones(len(senders), dtype=bool),
    )
    energy_fn = make_setfl_energy(tables)

    with jax.enable_x64(True):
        positions = jnp.asarray(positions_np.astype(np.float64))
        species = jnp.asarray(species_np, dtype=jnp.int32)
        energy, grad = jax.value_and_grad(
            lambda pos: jnp.sum(energy_fn(pos, species, graph))
        )(positions)

        density_t = jnp.asarray(tables["density"], jnp.float64)
        pair_t = jnp.asarray(tables["pair"], jnp.float64)
        embed_t = jnp.asarray(tables["embedding"], jnp.float64)
        pair_index = jnp.asarray(tables["pair_index"])
        spec_i = jnp.broadcast_to(species[:, None], (32, 32))
        spec_j = jnp.broadcast_to(species[None, :], (32, 32))

        def dense_total(pos):
            delta = pos[None, :, :] - pos[:, None, :]
            r_sq = jnp.sum(delta * delta, axis=-1)
            valid = (~jnp.eye(32, dtype=bool)) & (r_sq < tables["cutoff"] ** 2)
            r = jnp.sqrt(jnp.where(valid, r_sq, 1.0))
            rho_edge = spline_lookup(density_t, spec_j, r, tables["dr"])
            density = jnp.sum(jnp.where(valid, rho_edge, 0.0), axis=1)
            z2 = spline_lookup(pair_t, pair_index[spec_i, spec_j], r, tables["dr"])
            pair_energy = 0.5 * jnp.sum(jnp.where(valid, z2 / r, 0.0), axis=1)
            embed = spline_lookup(embed_t, species, density, tables["drho"],
                                  extrapolate=True)
            return jnp.sum(pair_energy + embed)

        ref_energy, ref_grad = jax.value_and_grad(dense_total)(positions)

    assert float(energy) == pytest.approx(float(ref_energy), rel=1e-12)
    assert np.allclose(np.asarray(grad), np.asarray(ref_grad), atol=1e-12)


def native_numpy_reference(tables, positions, species):
    """Energy, forces, and densities by native pair_eam semantics in numpy.

    Independent port of PairEAM::compute's evaluation, on coefficients rebuilt
    by helpers.native_interpolate; shares only raw table values with the model.
    """
    embedding = np.stack([native_interpolate(c[:, 0]) for c in tables["embedding"]])
    density = np.stack([native_interpolate(c[:, 0]) for c in tables["density"]])
    pair = np.stack([native_interpolate(c[:, 0]) for c in tables["pair"]])

    def evaluate(coeffs, x, delta, extrapolate=False):
        n = coeffs.shape[0]
        idx = x / delta
        node = int(np.clip(np.floor(idx), 0, n - 2))
        t_raw = idx - node
        t = min(t_raw, 1.0)
        c = coeffs[node]
        value = c[0] + t * (c[1] + t * (c[2] + t * c[3]))
        deriv = (c[1] + t * (2.0 * c[2] + 3.0 * t * c[3])) / delta
        if extrapolate:
            value += (t_raw - t) * (c[1] + 2.0 * c[2] + 3.0 * c[3])
        elif t_raw > 1.0:
            deriv = 0.0
        return value, deriv

    n_atoms = len(positions)
    r = np.sqrt(((positions[None] - positions[:, None]) ** 2).sum(-1))
    np.fill_diagonal(r, np.inf)
    within = r < tables["cutoff"]

    densities = np.zeros(n_atoms)
    for i, j in np.argwhere(within):
        densities[i] += evaluate(density[species[j]], r[i, j], tables["dr"])[0]
    energy = 0.0
    fp = np.zeros(n_atoms)
    for i in range(n_atoms):
        value, fp[i] = evaluate(embedding[species[i]], densities[i],
                                tables["drho"], extrapolate=True)
        energy += value

    forces = np.zeros_like(positions)
    for i, j in np.argwhere(within):
        if j <= i:
            continue
        rij = r[i, j]
        z2, z2p = evaluate(pair[tables["pair_index"][species[i], species[j]]],
                           rij, tables["dr"])
        phi = z2 / rij
        phip = (z2p - phi) / rij
        rhoip = evaluate(density[species[i]], rij, tables["dr"])[1]
        rhojp = evaluate(density[species[j]], rij, tables["dr"])[1]
        psip = fp[i] * rhojp + fp[j] * rhoip + phip
        fvec = (positions[i] - positions[j]) * (-psip / rij)
        energy += phi
        forces[i] += fvec
        forces[j] -= fvec
    return energy, forces, densities


def test_setfl_energy_matches_native_numpy_reference(tmp_path):
    """Model vs an independent numpy port of pair_eam.cpp, no shared spline_lookup.

    The fixture drives most atoms past rhomax, so the embedding continuation
    is checked, not just the in-table splines.
    """
    path = tmp_path / "tiny.eam.alloy"
    synthetic_setfl(path)
    tables = load_setfl(str(path))
    positions_np = random_system(seed=23, n_atoms=32,
                                 box=(3.0, 2.0, 2.0)).astype(np.float64)
    species_np = np.random.default_rng(29).integers(0, 2, size=32)

    ref_energy, ref_forces, densities = native_numpy_reference(
        tables, positions_np, species_np)
    rhomax = (tables["nrho"] - 1) * tables["drho"]
    assert (densities > rhomax).sum() >= 10

    senders, receivers = edges_within_cutoff(positions_np, tables["cutoff"])
    graph = LammpsNeighborList(
        senders=jnp.asarray(senders),
        receivers=jnp.asarray(receivers),
        edge_mask=jnp.ones(len(senders), dtype=bool),
    )
    energy_fn = make_setfl_energy(tables)
    with jax.enable_x64(True):
        energy, grad = jax.value_and_grad(
            lambda pos: jnp.sum(energy_fn(pos, jnp.asarray(species_np,
                                                           jnp.int32), graph))
        )(jnp.asarray(positions_np))

    assert float(energy) == pytest.approx(ref_energy, rel=1e-12)
    assert np.allclose(-np.asarray(grad), ref_forces, atol=1e-10)


def test_embedding_extrapolates_linearly_past_rhomax():
    """Densities beyond rhomax continue F with the end slope.

    Matches the fp * (rho - rhomax) term native pair_eam adds past the table.
    """
    rho_nodes = np.linspace(0.0, 3.0, 31)
    drho = float(rho_nodes[1] - rho_nodes[0])
    rhomax = float(rho_nodes[-1])

    with jax.enable_x64(True):
        coeffs = jnp.asarray(spline_coefficients(np.sinh(rho_nodes))[None],
                             jnp.float64)
        table = jnp.zeros((), jnp.int32)

        def value(rho):
            return spline_lookup(coeffs, table, jnp.float64(rho), drho,
                                 extrapolate=True)

        step = 1e-7
        slope = (float(value(rhomax)) - float(value(rhomax - step))) / step
        for excess in (0.5, 2.0, 10.0):
            expected = float(value(rhomax)) + slope * excess
            assert float(value(rhomax + excess)) == pytest.approx(
                expected, rel=1e-5)
            gradient = float(jax.grad(
                lambda rho: spline_lookup(coeffs, table, rho, drho,
                                          extrapolate=True)
            )(jnp.float64(rhomax + excess)))
            assert gradient == pytest.approx(slope, rel=1e-5)
        saturated = spline_lookup(coeffs, table, jnp.float64(rhomax + 2.0),
                                  drho)
        assert float(saturated) == pytest.approx(float(value(rhomax)),
                                                 rel=1e-12)


def test_half_edges_match_full_pairing(tmp_path):
    """Half-edge models on deduplicated graphs match full models exactly.

    Covers the analytic pair_embedding cross term and the setfl per-edge
    element-dependent density selection, energies and forces in float64.
    """
    path = tmp_path / "tiny.eam.alloy"
    synthetic_setfl(path)
    tables = load_setfl(str(path))
    positions_np = random_system(seed=31, n_atoms=32, box=(3.0, 2.0, 2.0))
    species_np = np.random.default_rng(37).integers(0, 2, size=32)
    senders, receivers = edges_within_cutoff(positions_np, tables["cutoff"])
    keep = senders < receivers

    cases = [
        (make_eam_energy(cutoff=CUTOFF, pair_embedding=0.3, **PARAMS),
         make_eam_energy(cutoff=CUTOFF, pair_embedding=0.3, half_edges=True, **PARAMS)),
        (make_setfl_energy(tables),
         make_setfl_energy(tables, half_edges=True)),
    ]
    with jax.enable_x64(True):
        positions = jnp.asarray(positions_np.astype(np.float64))
        species = jnp.asarray(species_np, dtype=jnp.int32)
        full_graph = LammpsNeighborList(
            senders=jnp.asarray(senders),
            receivers=jnp.asarray(receivers),
            edge_mask=jnp.ones(len(senders), dtype=bool),
        )
        half_graph = LammpsNeighborList(
            senders=jnp.asarray(senders[keep]),
            receivers=jnp.asarray(receivers[keep]),
            edge_mask=jnp.ones(int(keep.sum()), dtype=bool),
        )
        for full_fn, half_fn in cases:
            full_energy, full_grad = jax.value_and_grad(
                lambda pos: jnp.sum(full_fn(pos, species, full_graph))
            )(positions)
            half_energy, half_grad = jax.value_and_grad(
                lambda pos: jnp.sum(half_fn(pos, species, half_graph))
            )(positions)
            assert float(half_energy) == pytest.approx(float(full_energy), rel=1e-12)
            assert np.allclose(np.asarray(half_grad), np.asarray(full_grad), atol=1e-12)


def test_export_eam_bundle(tmp_path):
    data = export_and_load(
        tmp_path / "eam.json",
        energy_fn=make_eam_energy(cutoff=1.6, **PARAMS),
        max_atoms=8,
        max_edges=16,
        cutoff=1.6,
        unit_style="lj",
        n_hops=2,
    )
    assert data["format"] == DISTRIBUTED_BUNDLE_FORMAT
    assert data["contract"]["n_hops"] == 2
    assert data["contract"]["newton"] == "on"
    assert data["contract"]["force_output"] == "atom-force"
    assert "func.func public @main" in program_text(data, "energy_and_forces_mlir")
