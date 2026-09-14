"""Pallas EAM kernels against the jnp analytic builder, in Pallas interpret mode on the CPU.

The interpreter's atomic add is last-write-wins on repeated indices within one call and has no
mask, so every program must hold distinct endpoints: one edge or row per program on a general
graph, or a matching graph with several per program.
"""

import jax.numpy as jnp
import numpy as np

from helpers import NoComm, edges_within_cutoff, random_system
from lammps_jax.eam import load_setfl, make_setfl_edge_force, make_setfl_energy
from lammps_jax.eam_pallas import (make_setfl_pallas_force, make_setfl_pallas_row_energy,
                                   make_setfl_pallas_row_force, relaxed_atomics)
from lammps_jax.export import LammpsNeighborMatrix, export_model, program_text
from test_eam import padded_graph, synthetic_setfl


def reference_forces(tables, positions, species, senders, receivers, graph):
    per_edge = np.asarray(make_setfl_edge_force(tables, communicating=True, half_edges=True)(
        positions, species, graph, NoComm()))[: len(senders)]
    expected = np.zeros((positions.shape[0], 3))
    np.add.at(expected, senders, per_edge)
    np.add.at(expected, receivers, -per_edge)
    return expected


def check(tables, positions, species, senders, receivers, graph, block, n_real):
    expected = reference_forces(tables, positions, species, senders, receivers, graph)
    forces = np.asarray(make_setfl_pallas_force(tables, block=block, interpret=True)(
        positions, species, graph, NoComm()))
    assert forces.shape == expected.shape
    assert np.abs(expected[:n_real]).max() > 0.1
    np.testing.assert_allclose(forces, expected, atol=2e-5 * (1.0 + np.abs(expected).max()))
    assert np.all(forces[n_real:] == 0.0)


def test_pallas_forces_match_analytic_builder(tmp_path):
    """Both endpoints of every half edge receive the reference force; padded rows stay zero."""
    path = tmp_path / "tiny.eam.alloy"
    synthetic_setfl(path)
    tables = load_setfl(str(path))
    n_real, n_rows = 32, 40
    positions_np = random_system(seed=41, n_atoms=n_real, box=(3.0, 2.0, 2.0))
    species_np = np.random.default_rng(43).integers(0, 2, size=n_real, dtype=np.int32)
    senders, receivers = edges_within_cutoff(positions_np, tables["cutoff"])
    keep = senders < receivers
    senders, receivers = senders[keep], receivers[keep]
    graph = padded_graph(senders, receivers, n_rows, 3 * len(senders))
    positions = jnp.zeros((n_rows, 3), jnp.float32).at[:n_real].set(positions_np)
    species = jnp.zeros((n_rows,), jnp.int32).at[:n_real].set(species_np)
    check(tables, positions, species, senders, receivers, graph, 1, n_real)


def test_pallas_blocks_with_ragged_tail(tmp_path):
    """Several edges per program, a partial last program, and pad lanes beside valid ones.

    Each atom sits in at most one edge and atom 0 in none, so the clamped pad lanes never
    collide with a real update inside the interpreter.
    """
    path = tmp_path / "tiny.eam.alloy"
    synthetic_setfl(path)
    tables = load_setfl(str(path))
    n_real, n_rows = 32, 40
    positions_np = random_system(seed=47, n_atoms=n_real, box=(3.0, 2.0, 2.0))
    species_np = np.random.default_rng(53).integers(0, 2, size=n_real, dtype=np.int32)
    all_s, all_r = edges_within_cutoff(positions_np, tables["cutoff"])
    used = {0}
    senders, receivers = [], []
    for i, j in zip(all_s, all_r):
        if i < j and i not in used and j not in used:
            used.update((i, j))
            senders.append(i)
            receivers.append(j)
    senders, receivers = np.asarray(senders, np.int32), np.asarray(receivers, np.int32)
    assert len(senders) >= 6
    block = 4
    graph = padded_graph(senders, receivers, n_rows, block * len(senders) + 3)
    positions = jnp.zeros((n_rows, 3), jnp.float32).at[:n_real].set(positions_np)
    species = jnp.zeros((n_rows,), jnp.int32).at[:n_real].set(species_np)
    check(tables, positions, species, senders, receivers, graph, block, n_real)


def matrix_graph(senders, receivers, rows, max_neighbors, max_atoms):
    """The plugin's layout: slot-major rows filled from slot 0; padding indexes max_atoms."""
    neighbors = np.full((max_neighbors, rows), max_atoms, np.int32)
    counts = np.zeros(rows, np.int32)
    for i, j in zip(senders, receivers):
        neighbors[counts[i], i] = j
        counts[i] += 1
    return LammpsNeighborMatrix(jnp.asarray(neighbors), jnp.asarray(counts))


def test_pallas_row_kernels_match_edge_reference(tmp_path):
    """Row kernels over the list matrix reproduce the edge forces and energies row by row."""
    path = tmp_path / "tiny.eam.alloy"
    synthetic_setfl(path)
    tables = load_setfl(str(path))
    n_real, n_rows, rows = 32, 40, 36
    positions_np = random_system(seed=41, n_atoms=n_real, box=(3.0, 2.0, 2.0))
    species_np = np.random.default_rng(43).integers(0, 2, size=n_real, dtype=np.int32)
    senders, receivers = edges_within_cutoff(positions_np, tables["cutoff"])
    keep = senders < receivers
    senders, receivers = senders[keep], receivers[keep]
    edge_graph = padded_graph(senders, receivers, n_rows, 3 * len(senders))
    graph = matrix_graph(senders, receivers, rows, int(np.bincount(senders).max()) + 3, n_rows)
    positions = jnp.zeros((n_rows, 3), jnp.float32).at[:n_real].set(positions_np)
    species = jnp.zeros((n_rows,), jnp.int32).at[:n_real].set(species_np)
    expected = reference_forces(tables, positions, species, senders, receivers, edge_graph)
    forces = np.asarray(make_setfl_pallas_row_force(tables, block=1, interpret=True)(
        positions, species, graph, NoComm()))
    np.testing.assert_allclose(forces, expected, atol=2e-5 * (1.0 + np.abs(expected).max()))
    assert np.all(forces[n_real:] == 0.0)
    expected_energy = np.asarray(make_setfl_energy(tables, communicating=True, half_edges=True)(
        positions, species, edge_graph, NoComm()))
    energies = np.asarray(make_setfl_pallas_row_energy(tables, block=1, interpret=True)(
        positions, species, graph, NoComm()))
    assert energies.shape == (n_rows,)
    assert np.abs(expected_energy[:n_real]).max() > 0.1
    np.testing.assert_allclose(energies, expected_energy,
                               atol=2e-5 * (1.0 + np.abs(expected_energy).max()))


def test_pallas_row_kernels_ragged_programs(tmp_path):
    """Several rows per program with a partial last program on a matching graph."""
    path = tmp_path / "tiny.eam.alloy"
    synthetic_setfl(path)
    tables = load_setfl(str(path))
    n_real, n_rows, rows = 32, 40, 35
    positions_np = random_system(seed=47, n_atoms=n_real, box=(3.0, 2.0, 2.0))
    species_np = np.random.default_rng(53).integers(0, 2, size=n_real, dtype=np.int32)
    all_s, all_r = edges_within_cutoff(positions_np, tables["cutoff"])
    used = {0}
    senders, receivers = [], []
    for i, j in zip(all_s, all_r):
        if i < j and i not in used and j not in used:
            used.update((i, j))
            senders.append(i)
            receivers.append(j)
    senders, receivers = np.asarray(senders, np.int32), np.asarray(receivers, np.int32)
    edge_graph = padded_graph(senders, receivers, n_rows, 3 * len(senders))
    graph = matrix_graph(senders, receivers, rows, 4, n_rows)
    positions = jnp.zeros((n_rows, 3), jnp.float32).at[:n_real].set(positions_np)
    species = jnp.zeros((n_rows,), jnp.int32).at[:n_real].set(species_np)
    expected = reference_forces(tables, positions, species, senders, receivers, edge_graph)
    forces = np.asarray(make_setfl_pallas_row_force(tables, block=4, interpret=True)(
        positions, species, graph, NoComm()))
    np.testing.assert_allclose(forces, expected, atol=2e-5 * (1.0 + np.abs(expected).max()))
    assert np.all(forces[n_real:] == 0.0)


def test_pallas_row_kernels_full_list(tmp_path):
    """Full-list rows complete their own density and force without touching their neighbors."""
    path = tmp_path / "tiny.eam.alloy"
    synthetic_setfl(path)
    tables = load_setfl(str(path))
    n_real, n_rows, rows = 32, 40, 36
    positions_np = random_system(seed=59, n_atoms=n_real, box=(3.0, 2.0, 2.0))
    species_np = np.random.default_rng(61).integers(0, 2, size=n_real, dtype=np.int32)
    senders, receivers = edges_within_cutoff(positions_np, tables["cutoff"])
    keep = senders < receivers
    half_graph = padded_graph(senders[keep], receivers[keep], n_rows, 3 * int(keep.sum()))
    full_graph = padded_graph(senders, receivers, n_rows, 3 * len(senders))
    graph = matrix_graph(senders, receivers, rows, int(np.bincount(senders).max()) + 3, n_rows)
    positions = jnp.zeros((n_rows, 3), jnp.float32).at[:n_real].set(positions_np)
    species = jnp.zeros((n_rows,), jnp.int32).at[:n_real].set(species_np)
    expected = reference_forces(tables, positions, species, senders[keep], receivers[keep],
                                half_graph)
    # Rows write only their own index, so several rows per program are safe in the interpreter.
    full_force = make_setfl_pallas_row_force(tables, block=4, half_edges=False, interpret=True)
    forces = np.asarray(full_force(positions, species, graph, NoComm()))
    np.testing.assert_allclose(forces, expected, atol=2e-5 * (1.0 + np.abs(expected).max()))
    assert np.all(forces[n_real:] == 0.0)
    expected_energy = np.asarray(make_setfl_energy(tables)(positions, species, full_graph))
    full_energy = make_setfl_pallas_row_energy(tables, block=4, half_edges=False, interpret=True)
    energies = np.asarray(full_energy(positions, species, graph, NoComm()))
    assert np.abs(expected_energy[:n_real]).max() > 0.1
    np.testing.assert_allclose(energies, expected_energy,
                               atol=2e-5 * (1.0 + np.abs(expected_energy).max()))


def test_pallas_row_full_export_contract(tmp_path):
    """Full-list row bundles export with newton off and exchange only in the force programs."""
    path = tmp_path / "tiny.eam.alloy"
    synthetic_setfl(path)
    tables = load_setfl(str(path))
    with relaxed_atomics():
        data = export_model(
            energy_fn=make_setfl_pallas_row_energy(tables, block=2, half_edges=False),
            force_fn=make_setfl_pallas_row_force(tables, block=2, half_edges=False),
            path=tmp_path / "rows_full.json", max_atoms=8, max_neighbors=6, max_owned=6,
            cutoff=tables["cutoff"], unit_style="metal", force_output="atom-force", newton="off",
            comm=True, n_species=2, custom_call_targets=("__gpu$xla.gpu.triton",),
        )
    assert data["format"] == "lammps-jax-json-distributed"
    assert data["contract"]["newton"] == "off"
    assert data["contract"]["comm_sites"] == [1, 0, 1]


def test_pallas_row_export_contract(tmp_path):
    """Matrix bundles declare their layout and run Triton in all three programs."""
    path = tmp_path / "tiny.eam.alloy"
    synthetic_setfl(path)
    tables = load_setfl(str(path))
    with relaxed_atomics():
        data = export_model(
            energy_fn=make_setfl_pallas_row_energy(tables, block=2),
            force_fn=make_setfl_pallas_row_force(tables, block=2),
            path=tmp_path / "rows.json", max_atoms=8, max_neighbors=6, max_owned=6,
            cutoff=tables["cutoff"], unit_style="metal", force_output="atom-force", newton="on",
            comm=True, half_edges=True, n_species=2, custom_call_targets=("__gpu$xla.gpu.triton",),
        )
    contract = data["contract"]
    assert contract["input_layout"] == "neighbor-matrix"
    assert contract["max_neighbors"] == 6 and "max_edges" not in contract
    assert contract["max_owned"] == 6
    assert contract["comm_sites"] == [2, 1, 3]
    for program in ("force_mlir", "energy_mlir", "energy_and_forces_mlir"):
        assert "__gpu$xla.gpu.triton" in program_text(data, program)


def test_pallas_export_lowers_to_triton(tmp_path):
    """The bundle carries the Triton custom call, the plain energy program, and no leaked patch."""
    from jax._src.pallas.triton import lowering

    path = tmp_path / "tiny.eam.alloy"
    synthetic_setfl(path)
    tables = load_setfl(str(path))
    original = lowering._atomic_rmw
    with relaxed_atomics():
        assert lowering._atomic_rmw is not original
        data = export_model(
            energy_fn=make_setfl_energy(tables, communicating=True, half_edges=True),
            force_fn=make_setfl_pallas_force(tables, block=16),
            path=tmp_path / "pallas.json", max_atoms=8, max_edges=16, cutoff=tables["cutoff"],
            unit_style="metal", force_output="atom-force", newton="on", comm=True,
            half_edges=True, n_species=2, custom_call_targets=("__gpu$xla.gpu.triton",),
        )
    assert lowering._atomic_rmw is original
    assert data["contract"]["custom_call_targets"] == []
    assert data["contract"]["comm_sites"] == [2, 1, 3]
    assert "__gpu$xla.gpu.triton" in program_text(data, "force_mlir")
    assert "__gpu$xla.gpu.triton" not in program_text(data, "energy_mlir")
