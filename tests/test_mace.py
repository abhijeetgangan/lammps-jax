"""Real MACE-MP-0 small via mace_jax on CPU, torch-free from the converted bundle.

Skipped when mace_jax or the converted bundle is absent.
"""

import os
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

mace_jax = pytest.importorskip("mace_jax")

REPO = Path(__file__).resolve().parent.parent
BUNDLE_DIR = Path(os.environ.get(
    "MACE_MP_BUNDLE_DIR", REPO.parent / "models" / "mace-mp-0-small-jax"))
if not (BUNDLE_DIR / "config.json").exists():
    pytest.skip("converted MACE-MP bundle not available", allow_module_level=True)

sys.path.insert(0, str(REPO / "examples"))
import export_mace  # installs the torch-free Wigner-3j cache shims

from flax import nnx
from mace_jax.tools.bundle import load_model_bundle

from helpers import Graph, assert_comm_scheme_matches_reference, edges_within_cutoff
from lammps_jax.mace import make_mace_energy


@pytest.fixture(scope="module")
def mp_setup():
    bundle = load_model_bundle(str(BUNDLE_DIR), "float32")
    model = nnx.merge(bundle.graphdef, bundle.params)
    rng = np.random.default_rng(7)
    cells, a = (6, 2, 2), 4.05
    base = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0],
                     [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]])
    lattice = np.stack(np.meshgrid(*[np.arange(c) for c in cells],
                                   indexing="ij"), -1).reshape(-1, 1, 3)
    positions = ((lattice + base) * a).reshape(-1, 3)
    positions = (positions + rng.normal(0.0, 0.05, positions.shape)).astype(np.float32)
    aluminum = [int(z) for z in bundle.config["atomic_numbers"]].index(13)
    species = np.full((len(positions),), aluminum, np.int32)
    return bundle.config, model, positions, species


def test_mace_adapter_matches_direct_mace_jax(mp_setup):
    config, model, positions_np, species_np = mp_setup
    cutoff = float(config["r_max"])
    num_elements = int(config["num_elements"])
    senders, receivers = edges_within_cutoff(positions_np, cutoff)
    n = len(positions_np)

    data = {
        "positions": jnp.asarray(positions_np),
        "cell": jnp.eye(3, dtype=jnp.float32)[None],
        "shifts": jnp.zeros((len(senders), 3), jnp.float32),
        "unit_shifts": jnp.zeros((len(senders), 3), jnp.float32),
        "edge_index": jnp.asarray(np.stack([receivers, senders]), jnp.int32),
        "node_attrs": jnp.eye(num_elements, dtype=jnp.float32)[species_np],
        "node_attrs_index": jnp.asarray(species_np),
        "batch": jnp.zeros((n,), jnp.int32),
        "ptr": jnp.asarray([0, n], jnp.int32),
    }
    reference = model(data, compute_force=False)
    reference = reference[0] if isinstance(reference, tuple) else reference

    energy_fn = make_mace_energy(config=config, model=model)
    ours = energy_fn(
        jnp.asarray(positions_np), jnp.asarray(species_np), Graph(senders, receivers)
    )
    np.testing.assert_allclose(
        np.asarray(ours), np.asarray(reference["node_energy"]), atol=1e-5
    )


@pytest.mark.parametrize(("exchange", "should_match"), [(True, True), (False, False)])
def test_mace_comm_scheme_matches_reference(mp_setup, exchange, should_match):
    config, model, positions_np, species_np = mp_setup
    energy_fn = make_mace_energy(config=config, model=model, communicating=True)
    assert_comm_scheme_matches_reference(
        energy_fn,
        positions_np,
        species_np,
        cutoff=float(config["r_max"]),
        exchange=exchange,
        should_match=should_match,
        atol=3e-4,
        scale_force_atol=True,
        x_split=12.15,
    )


def test_mace_mp_adapter_matches_torch_reference():
    """The converted MP-0 small bundle matches upstream mace-torch.

    Torch reference float64 energies/forces are embedded so the test needs no
    torch; regenerate with python tests/test_mace.py <checkpoint>.
    """
    import jax

    from lammps_jax.export import LammpsNeighborList

    torch_energy = -18.232465815561724
    torch_forces = np.array(
        [[0.22490264397581225, 0.9011733328866178, 0.6237052117802777],
         [0.7914569806252991, 0.7527125358250166, -0.6562661064647407],
         [1.0325735120904416, -0.8629742059040957, 0.8368805756870908],
         [0.3058794857397641, -0.818469962209097, -0.8126904117967411],
         [-0.3050421675873607, 0.772171490702066, 0.7629639895749367],
         [-0.7232901758712733, 0.7145467138429309, -0.7862673655358662],
         [-0.8879842492792909, -0.7647597015584646, 0.6511121720471862],
         [-0.4384960296933925, -0.6944002035849741, -0.6194380652921436]])

    bundle = load_model_bundle(str(BUNDLE_DIR), "float32")
    model = nnx.merge(bundle.graphdef, bundle.params)
    energy_fn = make_mace_energy(config=bundle.config, model=model,
                                 communicating=False)

    rng = np.random.default_rng(3)
    positions_np = (np.array([[i, j, k] for i in range(2) for j in range(2)
                              for k in range(2)], dtype=np.float64) * 2.86
                    + rng.normal(0, 0.08, (8, 3)))
    delta = positions_np[None] - positions_np[:, None]
    distance = np.sqrt((delta ** 2).sum(-1))
    senders, receivers = np.nonzero(
        (distance < float(bundle.config["r_max"])) & ~np.eye(8, dtype=bool))
    graph = LammpsNeighborList(
        senders=jnp.asarray(senders, jnp.int32),
        receivers=jnp.asarray(receivers, jnp.int32),
        edge_mask=jnp.ones(len(senders), bool))
    z_table = [int(z) for z in bundle.config["atomic_numbers"]]
    species = jnp.full((8,), z_table.index(13), jnp.int32)

    with jax.default_matmul_precision("highest"):
        energy, grad = jax.value_and_grad(
            lambda p: jnp.sum(energy_fn(p, species, graph)))(
            jnp.asarray(positions_np, jnp.float32))

    assert abs(float(energy) - torch_energy) / 8 < 5e-4
    assert np.max(np.abs(-np.asarray(grad) - torch_forces)) < 5e-3


if __name__ == "__main__":
    import torch
    from ase import Atoms
    from mace.calculators import MACECalculator

    rng = np.random.default_rng(3)
    positions = (np.array([[i, j, k] for i in range(2) for j in range(2)
                           for k in range(2)], dtype=np.float64) * 2.86
                 + rng.normal(0, 0.08, (8, 3)))
    atoms = Atoms("Al8", positions=positions, pbc=False)
    atoms.calc = MACECalculator(model_paths=sys.argv[1], device="cpu",
                                default_dtype="float64")
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    np.set_printoptions(precision=17, floatmode="unique")
    print(f"TORCH_ENERGY = {energy!r}")
    print("TORCH_FORCES = np.array(")
    print(repr(forces.tolist()) + ")")
