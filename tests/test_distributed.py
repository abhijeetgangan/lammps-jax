"""The two distributed schemes: ghost with n_hops * cutoff shells, comm with FFI exchange.

The exchange is emulated with JAX gather/scatter; the FFI handlers only run
inside LAMMPS on a GPU, covered by tests/test_lammps.py.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from helpers import (
    FEATURE_WIDTH,
    N_SPECIES,
    Graph,
    assert_comm_scheme_matches_reference,
    call_model,
    edges_within_cutoff,
    export_and_load,
    make_toy_mp_energy,
    pair_energy,
    random_system,
    toy_mp_params,
    two_domains,
    zero_force_fn,
)
from lammps_jax import comm
from lammps_jax.eam import make_eam_energy
from lammps_jax.export import (
    BUNDLE_FORMAT,
    DISTRIBUTED_BUNDLE_FORMAT,
    HALF_EDGE_BUNDLE_FORMAT,
    program_text,
    wrap_energy_fn,
)

CUTOFF = 1.0
N_HOPS = 2


# comm.py units


def test_forward_comm_is_identity_eagerly():
    tree = {"a": jnp.ones((5, 3)), "b": jnp.zeros((5, 2, 2))}
    out = comm.forward_comm(tree)
    assert jax.tree.structure(out) == jax.tree.structure(tree)
    for leaf, expected in zip(jax.tree.leaves(out), jax.tree.leaves(tree)):
        np.testing.assert_array_equal(np.asarray(leaf), np.asarray(expected))


def test_comm_width_counts_trailing_dims():
    assert comm.comm_width({"a": jnp.ones((5, 3)), "b": jnp.zeros((5, 2, 2))}) == 7
    with pytest.raises(TypeError, match="floating"):
        comm.comm_width(jnp.ones((5,), dtype=jnp.int32))
    with pytest.raises(ValueError, match="atom-leading"):
        comm.comm_width(jnp.float32(1.0))


def test_comm_records_and_validates_widths():
    recorder = comm.Comm(enabled=False)
    recorder.forward_comm(jnp.ones((4, 3)))
    recorder.forward_comm({"h": jnp.ones((4, 2))})
    assert recorder.widths == [3, 2]
    recorder.validate()

    checked = comm.Comm(enabled=False, expected_widths=(3, 2))
    checked.forward_comm(jnp.ones((4, 3)))
    with pytest.raises(ValueError, match="width 5, expected 2"):
        checked.forward_comm(jnp.ones((4, 5)))

    # A program may visit a leading subset of the bundle schedule.
    prefix = comm.Comm(enabled=False, expected_widths=(3, 2))
    prefix.forward_comm(jnp.ones((4, 3)))
    prefix.validate()


def test_exchange_differentiates_in_every_mode():
    """The exchange primitives are linear, so all AD modes compose: tangents
    ride a forward exchange, cotangents the adjoint reverse exchange."""

    def energy(x):
        c = comm.Comm(enabled=True)
        y = c.forward_comm(x)
        return jnp.sum(y * y)

    spec = jax.ShapeDtypeStruct((4, 3), jnp.float32)

    def exchange_targets(fn, *specs):
        text = jax.jit(fn).lower(*specs).as_text()
        return comm.FORWARD_TARGET in text, comm.REVERSE_TARGET in text

    assert exchange_targets(jax.grad(energy), spec) == (True, True)
    jvp = lambda x, v: jax.jvp(energy, (x,), (v,))[1]
    assert exchange_targets(jvp, spec, spec) == (True, False)
    hvp = lambda x, v: jax.jvp(jax.grad(energy), (x,), (v,))[1]
    assert exchange_targets(hvp, spec, spec) == (True, True)
    grad_of_grad = lambda x, v: jax.grad(
        lambda y: jnp.vdot(jax.grad(energy)(y), v)
    )(x)
    assert exchange_targets(grad_of_grad, spec, spec) == (True, True)


# Export wiring


def test_export_model_records_n_hops(tmp_path):
    default_data = export_and_load(
        tmp_path / "default.json",
        energy_fn=pair_energy,
        max_atoms=4,
        max_edges=6,
        cutoff=2.0,
        unit_style="lj",
    )
    assert default_data["contract"]["n_hops"] == 1
    assert default_data["contract"]["comm_widths"] == []
    assert default_data["contract"]["edge_pairing"] == "full"
    assert default_data["format"] == BUNDLE_FORMAT

    ghost_data = export_and_load(
        tmp_path / "ghost.json",
        energy_fn=pair_energy,
        max_atoms=4,
        max_edges=6,
        cutoff=2.0,
        unit_style="lj",
        n_hops=2,
    )
    contract = ghost_data["contract"]
    assert contract["n_hops"] == 2
    assert contract["newton"] == "on"
    assert contract["force_output"] == "atom-force"
    assert ghost_data["format"] == DISTRIBUTED_BUNDLE_FORMAT


def test_export_communicating_bundle(tmp_path):
    data = export_and_load(
        tmp_path / "toy-comm.json",
        energy_fn=make_toy_mp_energy(cutoff=CUTOFF, communicating=True),
        max_atoms=16,
        max_edges=64,
        cutoff=CUTOFF,
        unit_style="lj",
        comm=True,
    )
    assert data["format"] == DISTRIBUTED_BUNDLE_FORMAT
    contract = data["contract"]
    assert len(contract["comm_widths"]) == 1
    assert contract["comm_widths"] == [FEATURE_WIDTH]
    assert contract["n_hops"] == 1
    assert contract["newton"] == "on"
    assert contract["force_output"] == "atom-force"

    energy_text = program_text(data, "energy_mlir")
    assert comm.FORWARD_TARGET in energy_text
    assert comm.REVERSE_TARGET not in energy_text
    for program in ("force_mlir", "energy_and_forces_mlir"):
        text = program_text(data, program)
        assert comm.FORWARD_TARGET in text
        assert comm.REVERSE_TARGET in text


def test_export_half_edge_bundle(tmp_path):
    """Half-edge exports carry the half-edge format tag and the pairing key, so
    loaders that pack both directions reject them instead of double
    counting."""
    data = export_and_load(
        tmp_path / "eam-half.json",
        energy_fn=make_eam_energy(cutoff=CUTOFF, half_edges=True),
        max_atoms=8,
        max_edges=16,
        cutoff=CUTOFF,
        unit_style="lj",
        n_hops=2,
        half_edges=True,
    )
    assert data["format"] == HALF_EDGE_BUNDLE_FORMAT
    assert data["contract"]["edge_pairing"] == "half"
    assert data["contract"]["n_hops"] == 2

    comm_data = export_and_load(
        tmp_path / "eam-comm-half.json",
        energy_fn=make_eam_energy(cutoff=CUTOFF, communicating=True,
                                  half_edges=True),
        max_atoms=8,
        max_edges=16,
        cutoff=CUTOFF,
        unit_style="lj",
        comm=True,
        half_edges=True,
    )
    assert comm_data["format"] == HALF_EDGE_BUNDLE_FORMAT
    assert comm_data["contract"]["edge_pairing"] == "half"
    assert comm_data["contract"]["comm_widths"] == [1]
    assert comm.FORWARD_TARGET in program_text(comm_data, "energy_mlir")


def scalar_energy(positions, species, graph):
    del species, graph
    return jnp.sum(positions * positions)


def never_communicates(positions, species, graph, comm_obj):
    del species, comm_obj
    safe = jnp.where(graph.edge_mask, graph.senders, 0)
    return jnp.zeros((positions.shape[0],), jnp.float32).at[safe].add(0.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (dict(n_hops=2, force_fn=zero_force_fn), "autodiff"),
        (dict(n_hops=2, energy_fn=scalar_energy), "per-atom"),
        (dict(comm=True, force_fn=zero_force_fn), "autodiff"),
        (
            dict(
                comm=True,
                energy_fn=make_toy_mp_energy(cutoff=CUTOFF, communicating=True),
                n_hops=2,
            ),
            "n_hops must be 1",
        ),
        (dict(comm=True, energy_fn=never_communicates),
         "never called comm.forward_comm"),
        (dict(half_edges=True, energy_fn=pair_energy), "half_edges requires"),
        (
            dict(
                comm=True,
                energy_fn=make_toy_mp_energy(cutoff=CUTOFF, communicating=True),
                precision="float64",
            ),
            "float32-only",
        ),
    ],
)
def test_export_rejects_invalid_distributed_configs(tmp_path, kwargs, message):
    with pytest.raises(ValueError, match=message):
        export_and_load(
            tmp_path / "bad.json",
            max_atoms=8,
            max_edges=16,
            cutoff=CUTOFF,
            unit_style="lj",
            **kwargs,
        )


# Scheme equivalence


def test_ghost_scheme_energy_masks_ghost_rows():
    """Ghost-sender edges must not flip the energy to the local+ghost sum.

    Ghost-row energies duplicate values owned by neighbor ranks, so
    owned_rows_only must mask them or the sum double counts.
    """
    _force, single_hop_energy, _fused = wrap_energy_fn(
        pair_energy, max_atoms=4, call_model=call_model
    )
    _force, multi_hop_energy, _fused = wrap_energy_fn(
        pair_energy, max_atoms=4, call_model=call_model, owned_rows_only=True
    )

    positions = jnp.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=jnp.float32,
    )
    species = jnp.zeros((4,), dtype=jnp.int32)
    args = (
        positions,
        species,
        jnp.int32(2),
        jnp.int32(1),
        jnp.array([0, 1, 0, 2], dtype=jnp.int32),
        jnp.array([1, 0, 2, 0], dtype=jnp.int32),
        jnp.array([True, True, True, True], dtype=jnp.bool_),
    )
    local_rows = 0.5 * 1.0 + 0.5 * 1.0 + 0.5 * 4.0
    ghost_row = 0.5 * 4.0
    assert float(single_hop_energy(*args)) == pytest.approx(local_rows + ghost_row, abs=1e-5)
    assert float(multi_hop_energy(*args)) == pytest.approx(local_rows, abs=1e-5)


@pytest.mark.parametrize(
    ("halo_hops", "should_match"),
    [(N_HOPS, True), (1, False)],
)
def test_ghost_halo_decomposition_matches_reference(halo_hops, should_match):
    """Emulate the ghost scheme end to end in JAX across two slab domains.

    Each domain masks energy to owned rows and sums ghost force rows back to
    owners; a halo of n_hops cutoffs must match the single-domain reference.
    """
    rng = np.random.default_rng(7)
    n_atoms = 96
    box = np.array([6.0 * CUTOFF, 2.0 * CUTOFF, 2.0 * CUTOFF])
    positions_np = (rng.random((n_atoms, 3)) * box).astype(np.float32)
    species_np = rng.integers(0, N_SPECIES, size=n_atoms)

    energy_fn = make_toy_mp_energy(cutoff=CUTOFF)
    positions = jnp.asarray(positions_np)
    species = jnp.asarray(species_np, dtype=jnp.int32)

    senders, receivers = edges_within_cutoff(positions_np, CUTOFF)
    ref_energy, ref_grad = jax.value_and_grad(
        lambda pos: jnp.sum(energy_fn(pos, species, Graph(senders, receivers)))
    )(positions)
    ref_forces = -ref_grad

    total_energy = 0.0
    total_forces = np.zeros((n_atoms, 3), dtype=np.float64)
    for present, n_owned in two_domains(
        positions_np, x_split=3.0 * CUTOFF, halo=halo_hops * CUTOFF
    ):
        owned_mask = jnp.asarray(np.arange(len(present)) < n_owned)
        senders, receivers = edges_within_cutoff(positions_np[present], CUTOFF)
        local_species = species[present]
        energy, grad = jax.value_and_grad(
            lambda pos: jnp.sum(jnp.where(
                owned_mask,
                energy_fn(pos, local_species, Graph(senders, receivers)),
                0.0,
            ))
        )(positions[present])
        forces = -grad
        total_energy += float(energy)
        np.add.at(total_forces, present, np.asarray(forces, dtype=np.float64))

    if should_match:
        assert total_energy == pytest.approx(float(ref_energy), abs=5e-4)
        np.testing.assert_allclose(total_forces, np.asarray(ref_forces), atol=5e-4)
    else:
        energy_close = total_energy == pytest.approx(float(ref_energy), abs=5e-4)
        forces_close = np.allclose(total_forces, np.asarray(ref_forces), atol=5e-4)
        assert not (energy_close and forces_close)


@pytest.mark.parametrize("model", ["toy_mp", "eam"])
@pytest.mark.parametrize(("exchange", "should_match"), [(True, True), (False, False)])
def test_comm_scheme_matches_reference(model, exchange, should_match):
    """One-cutoff shell with per-layer exchange against the full graph.

    The EAM cross term couples energy to exchanged densities; at kappa 0
    the exchange could not change the value."""
    positions_np = random_system(seed=3)
    species_np = np.random.default_rng(5).integers(0, 2, size=len(positions_np))
    if model == "toy_mp":
        energy_fn = make_toy_mp_energy(cutoff=CUTOFF, params=toy_mp_params(),
                                       communicating=True)
    else:
        energy_fn = make_eam_energy(cutoff=CUTOFF, communicating=True,
                                    pair_embedding=0.3)
    assert_comm_scheme_matches_reference(
        energy_fn,
        positions_np,
        species_np,
        cutoff=CUTOFF,
        exchange=exchange,
        should_match=should_match,
        atol=5e-4,
        scale_force_atol=model == "eam",
    )
