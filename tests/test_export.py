"""Single-hop export ABI: bundle contents, validation, wrapper semantics."""

import base64
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from helpers import call_model, edges_within_cutoff, export_and_load, pair_energy, random_system
from lammps_jax.export import (
    ATOM_FORCE,
    BUNDLE_FORMAT,
    EDGE_FORCE,
    INPUT_LAYOUT,
    program_text,
    wrap_energy_fn,
)


def pair_force(positions, species, graph):
    del species
    safe_senders = jnp.where(graph.edge_mask, graph.senders, 0)
    safe_receivers = jnp.where(graph.edge_mask, graph.receivers, 0)
    edge_force = positions[safe_receivers] - positions[safe_senders]
    edge_force = jnp.where(graph.edge_mask[:, None], edge_force, 0.0)
    return jnp.zeros_like(positions).at[safe_senders].add(edge_force)


def dtype_pair_energy(positions, species, graph):
    """Variant of helpers.pair_energy that computes in positions.dtype for float64 exports."""
    del species
    dtype = positions.dtype
    safe_senders = jnp.where(graph.edge_mask, graph.senders, 0)
    safe_receivers = jnp.where(graph.edge_mask, graph.receivers, 0)
    rij = positions[safe_receivers] - positions[safe_senders]
    per_edge = jnp.sum(rij * rij, axis=-1)
    per_edge = 0.5 * jnp.where(graph.edge_mask, per_edge, jnp.asarray(0.0, dtype))
    return jnp.zeros((positions.shape[0],), dtype=dtype).at[safe_senders].add(per_edge)


def test_export_model_writes_expected_abi(tmp_path):
    data = export_and_load(
        tmp_path / "model.json",
        energy_fn=pair_energy,
        max_atoms=4,
        max_edges=6,
        cutoff=2.0,
        unit_style="lj",
    )
    assert data["format"] == BUNDLE_FORMAT
    assert data["contract"]["max_atoms"] == 4
    assert data["contract"]["max_edges"] == 6
    assert data["contract"]["input_layout"] == INPUT_LAYOUT
    assert data["contract"]["force_output"] == ATOM_FORCE
    assert data["contract"]["newton"] == "on"
    assert "n_species" not in data["contract"]
    options = base64.b64decode(data["compile_options_b64"])
    assert options
    assert len(options) < 100
    from jax._src.lib import xla_client
    parsed = xla_client.CompileOptions.ParseFromString(options)
    assert parsed.compile_portable_executable
    assert parsed.num_replicas == 1
    assert parsed.num_partitions == 1
    for program in ("force_mlir", "energy_mlir", "energy_and_forces_mlir"):
        assert "func.func public @main" in program_text(data, program)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"energy_fn": pair_energy, "newton": "off"}, "newton on"),
        ({"force_fn": pair_force, "newton": "maybe"}, "newton"),
        (
            {"energy_fn": pair_energy, "force_fn": pair_force, "force_output": EDGE_FORCE},
            "newton-dependent",
        ),
        ({"energy_fn": pair_energy, "precision": "bfloat16"}, "precision"),
    ],
)
def test_export_model_rejects_invalid_configs(tmp_path, kwargs, message):
    with pytest.raises(ValueError, match=message):
        export_and_load(
            tmp_path / "bad.json",
            max_atoms=4,
            max_edges=6,
            cutoff=2.0,
            unit_style="lj",
            **kwargs,
        )


def test_export_model_records_n_species(tmp_path):
    """n_species reaches the contract; the pair style rejects decks with more atom types.

    Closes the clamped-gather path where an unknown type computes as the last species.
    """
    data = export_and_load(
        tmp_path / "typed.json",
        energy_fn=pair_energy,
        max_atoms=4,
        max_edges=6,
        cutoff=2.0,
        unit_style="lj",
        n_species=2,
    )
    assert data["contract"]["n_species"] == 2

    with pytest.raises(ValueError, match="n_species"):
        export_and_load(
            tmp_path / "bad-species.json",
            energy_fn=pair_energy,
            max_atoms=4,
            max_edges=6,
            cutoff=2.0,
            unit_style="lj",
            n_species=0,
        )


def test_export_model_accepts_force_fn(tmp_path):
    data = export_and_load(
        tmp_path / "force-model.json",
        force_fn=pair_force,
        max_atoms=4,
        max_edges=6,
        cutoff=2.0,
        unit_style="lj",
    )
    assert data["contract"]["newton"] == "any"
    assert data["contract"]["force_output"] == ATOM_FORCE
    assert "func.func public @main" in program_text(data, "force_mlir")
    assert data["programs"]["energy_mlir_b64"] == ""
    assert data["programs"]["energy_and_forces_mlir_b64"] == ""


def test_export_model_accepts_box_input(tmp_path):
    def energy(positions, species, graph, box):
        base = pair_energy(positions, species, graph)
        return base + jnp.float32(0.0) * jnp.sum(box)

    data = export_and_load(
        tmp_path / "box-model.json",
        energy_fn=energy,
        uses_box=True,
        max_atoms=4,
        max_edges=6,
        cutoff=2.0,
        unit_style="lj",
    )
    assert data["contract"]["uses_box"] is True
    force_text = program_text(data, "force_mlir")
    assert "%arg7" in force_text
    assert "tensor<3x3xf32>" in force_text


def test_export_model_float64_records_contract_and_abi(tmp_path):
    """precision='float64' widens positions/box and both outputs, nothing else.

    jax.enable_x64(True) is a context manager, so x64 stays off for the rest of the suite.
    """

    def energy(positions, species, graph, box):
        base = dtype_pair_energy(positions, species, graph)
        return base + jnp.zeros((), positions.dtype) * jnp.sum(box)

    with jax.enable_x64(True):
        data = export_and_load(
            tmp_path / "model-f64.json",
            energy_fn=energy,
            uses_box=True,
            max_atoms=4,
            max_edges=6,
            cutoff=2.0,
            unit_style="lj",
            precision="float64",
        )
    assert data["contract"]["precision"] == "float64"
    fused_text = program_text(data, "energy_and_forces_mlir")
    assert "tensor<4x3xf64>" in fused_text
    assert "tensor<3x3xf64>" in fused_text
    assert "tensor<4xi32>" in fused_text
    assert "tensor<6xi32>" in fused_text
    assert "tensor<6xi1>" in fused_text
    assert "f32" not in fused_text
    assert re.search(r"->\s*\(tensor<f64>.*tensor<4x3xf64>", fused_text)


def test_wrap_energy_fn_float64_matches_dense_reference():
    """f64 wrapper energies and forces match a dense f64 reference to 1e-12."""
    max_atoms = 32
    cutoff = 1.5
    positions_np = random_system(seed=7, n_atoms=max_atoms, box=(4.0, 2.0, 2.0)).astype(
        np.float64
    )
    senders_np, receivers_np = edges_within_cutoff(positions_np, cutoff)

    with jax.enable_x64(True):
        _force_fn, _energy_fn, fused_fn = wrap_energy_fn(
            dtype_pair_energy, max_atoms=max_atoms, call_model=call_model, dtype=jnp.float64
        )
        energy, forces = fused_fn(
            jnp.asarray(positions_np),
            jnp.zeros((max_atoms,), dtype=jnp.int32),
            jnp.int32(max_atoms),
            jnp.int32(0),
            jnp.asarray(senders_np),
            jnp.asarray(receivers_np),
            jnp.ones((len(senders_np),), dtype=jnp.bool_),
        )

    assert energy.dtype == jnp.float64
    assert forces.dtype == jnp.float64
    rij = positions_np[receivers_np] - positions_np[senders_np]
    ref_energy = 0.5 * np.sum(rij * rij)
    ref_forces = np.zeros_like(positions_np)
    np.add.at(ref_forces, senders_np, rij)
    np.add.at(ref_forces, receivers_np, -rij)
    assert float(energy) == pytest.approx(ref_energy, abs=1e-12)
    assert np.allclose(np.asarray(forces), ref_forces, atol=1e-12)


def test_export_model_float64_requires_x64(tmp_path):
    """Without x64 the trace would silently truncate to f32; it must raise."""
    assert not jax.config.jax_enable_x64  # ty: ignore[unresolved-attribute]
    with pytest.raises(ValueError, match="x64"):
        export_and_load(
            tmp_path / "bad-f64.json",
            energy_fn=dtype_pair_energy,
            max_atoms=4,
            max_edges=6,
            cutoff=2.0,
            unit_style="lj",
            precision="float64",
        )


def test_export_model_accepts_edge_force_output(tmp_path):
    def edge_force(positions, species, graph):
        del species
        safe_senders = jnp.where(graph.edge_mask, graph.senders, 0)
        safe_receivers = jnp.where(graph.edge_mask, graph.receivers, 0)
        edge_force = positions[safe_receivers] - positions[safe_senders]
        return jnp.where(graph.edge_mask[:, None], edge_force, 0.0)

    data = export_and_load(
        tmp_path / "edge-force-model.json",
        energy_fn=pair_energy,
        force_fn=edge_force,
        force_output=EDGE_FORCE,
        newton="on",
        max_atoms=4,
        max_edges=6,
        cutoff=2.0,
        unit_style="lj",
    )
    assert data["contract"]["input_layout"] == INPUT_LAYOUT
    assert data["contract"]["force_output"] == EDGE_FORCE
    assert data["contract"]["newton"] == "on"
    assert re.search(r"->\s*\(?tensor<6x3xf32>", program_text(data, "force_mlir"))
    assert "func.func public @main" in program_text(data, "energy_and_forces_mlir")


def test_export_model_records_custom_call_targets(tmp_path):
    """Models may embed external custom kernels as FFI calls.

    The exporter records the targets; the runtime registers handlers via
    LAMMPS_JAX_FFI_HANDLERS before compiling.
    """
    import jax

    @jax.custom_vjp
    def my_kernel(x):
        return jax.ffi.ffi_call(
            "my_custom_kernel",
            jax.ShapeDtypeStruct(x.shape, x.dtype),
            vmap_method="sequential",
        )(x)

    my_kernel.defvjp(lambda x: (my_kernel(x), None), lambda _, g: (g,))

    def kernel_energy(positions, species, graph):
        return my_kernel(pair_energy(positions, species, graph))

    data = export_and_load(
        tmp_path / "kernel-model.json",
        energy_fn=kernel_energy,
        max_atoms=4,
        max_edges=6,
        cutoff=2.0,
        unit_style="lj",
        custom_call_targets=("my_custom_kernel",),
    )
    assert data["contract"]["custom_call_targets"] == ["my_custom_kernel"]
    assert "my_custom_kernel" in program_text(data, "energy_mlir")
    assert "my_custom_kernel" in program_text(data, "energy_and_forces_mlir")


def test_energy_select_local_vs_total():
    """Ghost-sender energy convention of the exported wrappers.

    Newton off sums local rows only; duplicated newton on graphs add ghost-sender rows.
    """
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
