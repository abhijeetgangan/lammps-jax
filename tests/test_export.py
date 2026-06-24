import base64
import json
import re

import jax.numpy as jnp
import pytest

from lammps_jax.export import (
    ATOM_FORCE,
    BUNDLE_FORMAT,
    EDGE_FORCE,
    INPUT_LAYOUT,
    LammpsNeighborList,
    export_model,
    wrap_energy_fn,
)


def pair_energy(positions, species, graph):
    del species
    safe_senders = jnp.where(graph.edge_mask, graph.senders, 0)
    safe_receivers = jnp.where(graph.edge_mask, graph.receivers, 0)
    rij = positions[safe_receivers] - positions[safe_senders]
    per_edge = jnp.sum(rij * rij, axis=-1)
    per_edge = 0.5 * jnp.where(graph.edge_mask, per_edge, 0.0)
    return jnp.zeros((positions.shape[0],), dtype=jnp.float32).at[safe_senders].add(per_edge)


def pair_force(positions, species, graph):
    del species
    safe_senders = jnp.where(graph.edge_mask, graph.senders, 0)
    safe_receivers = jnp.where(graph.edge_mask, graph.receivers, 0)
    edge_force = positions[safe_receivers] - positions[safe_senders]
    edge_force = jnp.where(graph.edge_mask[:, None], edge_force, 0.0)
    return jnp.zeros_like(positions).at[safe_senders].add(edge_force)


def test_export_model_writes_expected_abi(tmp_path):
    path = tmp_path / "model.json"
    export_model(
        energy_fn=pair_energy,
        path=path,
        max_atoms=4,
        max_edges=6,
        cutoff=2.0,
        unit_style="lj",
    )

    data = json.loads(path.read_text())
    assert data["format"] == BUNDLE_FORMAT
    assert data["contract"]["max_atoms"] == 4
    assert data["contract"]["max_edges"] == 6
    assert data["contract"]["input_layout"] == INPUT_LAYOUT
    assert data["contract"]["force_output"] == ATOM_FORCE
    assert data["contract"]["newton"] == "on"
    assert base64.b64decode(data["compile_options_b64"])
    assert "func.func public @main" in data["programs"]["force_mlir"]
    assert "func.func public @main" in data["programs"]["energy_mlir"]
    assert "func.func public @main" in data["programs"]["energy_and_forces_mlir"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"energy_fn": pair_energy, "newton": "off"}, "newton on"),
        ({"force_fn": pair_force, "newton": "maybe"}, "newton"),
    ],
)
def test_export_model_rejects_invalid_configs(tmp_path, kwargs, message):
    with pytest.raises(ValueError, match=message):
        export_model(
            path=tmp_path / "bad-newton.json",
            max_atoms=4,
            max_edges=6,
            cutoff=2.0,
            unit_style="lj",
            **kwargs,
        )


def test_export_model_accepts_force_fn(tmp_path):
    path = tmp_path / "force-model.json"
    export_model(
        force_fn=pair_force,
        path=path,
        max_atoms=4,
        max_edges=6,
        cutoff=2.0,
        unit_style="lj",
    )

    data = json.loads(path.read_text())
    assert data["contract"]["newton"] == "any"
    assert data["contract"]["force_output"] == ATOM_FORCE
    assert "func.func public @main" in data["programs"]["force_mlir"]
    assert data["programs"]["energy_mlir"] == ""
    assert data["programs"]["energy_and_forces_mlir"] == ""


def test_export_model_accepts_box_input(tmp_path):
    def energy(positions, species, graph, box):
        base = pair_energy(positions, species, graph)
        return base + jnp.float32(0.0) * jnp.sum(box)

    path = tmp_path / "box-model.json"
    export_model(
        energy_fn=energy,
        uses_box=True,
        path=path,
        max_atoms=4,
        max_edges=6,
        cutoff=2.0,
        unit_style="lj",
    )

    data = json.loads(path.read_text())
    assert data["contract"]["uses_box"] is True
    assert "%arg7" in data["programs"]["force_mlir"]
    assert "tensor<3x3xf32>" in data["programs"]["force_mlir"]


def test_export_model_accepts_edge_force_output(tmp_path):
    def edge_force(positions, species, graph):
        del species
        safe_senders = jnp.where(graph.edge_mask, graph.senders, 0)
        safe_receivers = jnp.where(graph.edge_mask, graph.receivers, 0)
        edge_force = positions[safe_receivers] - positions[safe_senders]
        return jnp.where(graph.edge_mask[:, None], edge_force, 0.0)

    path = tmp_path / "edge-force-model.json"
    export_model(
        energy_fn=pair_energy,
        force_fn=edge_force,
        force_output=EDGE_FORCE,
        newton="on",
        path=path,
        max_atoms=4,
        max_edges=6,
        cutoff=2.0,
        unit_style="lj",
    )

    data = json.loads(path.read_text())
    assert data["contract"]["input_layout"] == INPUT_LAYOUT
    assert data["contract"]["force_output"] == EDGE_FORCE
    assert data["contract"]["newton"] == "on"
    assert re.search(r"->\s*\(?tensor<6x3xf32>", data["programs"]["force_mlir"])
    assert "func.func public @main" in data["programs"]["energy_and_forces_mlir"]


def test_export_model_rejects_dual_edge_force_newton_any(tmp_path):
    with pytest.raises(ValueError, match="newton-dependent"):
        export_model(
            energy_fn=pair_energy,
            force_fn=pair_force,
            force_output=EDGE_FORCE,
            path=tmp_path / "bad-dual-edge.json",
            max_atoms=4,
            max_edges=6,
            cutoff=2.0,
            unit_style="lj",
        )


def test_energy_select_local_vs_total():
    """Lock in the ghost-owned-edge convention used by the exported wrappers.

    Without ghost-sender edges, as in newton off graphs, the exported energy is
    the sum over local rows. With ghost-sender edges, as in duplicated newton on
    graphs, it is the sum over local plus ghost rows.
    """
    def call_model(model_fn, model_args):
        positions, species, nlocal, nghost, senders, receivers, edge_mask = model_args[:7]
        graph = LammpsNeighborList(senders=senders, receivers=receivers, edge_mask=edge_mask)
        return model_fn(positions, species, graph), graph, nlocal, nghost

    _force_fn, energy_fn, energy_and_forces_fn = wrap_energy_fn(
        pair_energy, max_atoms=4, call_model=call_model
    )

    positions = jnp.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=jnp.float32,
    )
    species = jnp.zeros((4,), dtype=jnp.int32)
    nlocal = jnp.int32(2)
    nghost = jnp.int32(1)

    def args_for(senders, receivers, mask):
        return (
            positions,
            species,
            nlocal,
            nghost,
            jnp.array(senders, dtype=jnp.int32),
            jnp.array(receivers, dtype=jnp.int32),
            jnp.array(mask, dtype=jnp.bool_),
        )

    local_args = args_for([0, 1, 0, 0], [1, 0, 2, 0], [True, True, True, False])
    expected_local = 0.5 * 1.0 + 0.5 * 1.0 + 0.5 * 4.0
    assert float(energy_fn(*local_args)) == pytest.approx(expected_local, abs=1e-5)

    ghost_args = args_for([0, 1, 0, 2], [1, 0, 2, 0], [True, True, True, True])
    expected_total = expected_local + 0.5 * 4.0
    assert float(energy_fn(*ghost_args)) == pytest.approx(expected_total, abs=1e-5)

    energy, forces = energy_and_forces_fn(*ghost_args)
    assert float(energy) == pytest.approx(expected_total, abs=1e-5)
    expected_f0 = jnp.array([2.0, 4.0, 0.0], dtype=jnp.float32)
    assert jnp.allclose(forces[0], expected_f0, atol=1e-5)
