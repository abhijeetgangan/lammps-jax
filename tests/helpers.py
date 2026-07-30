"""Shared helpers for the CPU emulation tests.

Dense reference neighbor graphs, two-slab ghost decompositions, and a
gather/scatter comm emulation whose autodiff yields the reverse-comm adjoint.
"""

import json
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lammps_jax.export import LammpsNeighborList, export_model


def random_system(seed, n_atoms=96, box=(6.0, 2.0, 2.0)):
    rng = np.random.default_rng(seed)
    return (rng.random((n_atoms, 3)) * np.asarray(box)).astype(np.float32)


def edges_within_cutoff(positions_np, cutoff, senders_subset=None):
    """All directed edges shorter than cutoff; optionally restrict sender rows."""
    delta = positions_np[None, :, :] - positions_np[:, None, :]
    dist_sq = np.sum(delta * delta, axis=-1)
    n = len(positions_np)
    mask = (dist_sq < cutoff**2) & ~np.eye(n, dtype=bool)
    if senders_subset is not None:
        keep = np.zeros(n, dtype=bool)
        keep[senders_subset] = True
        mask &= keep[:, None]
    senders, receivers = np.nonzero(mask)
    return senders.astype(np.int32), receivers.astype(np.int32)


def two_domains(positions_np, x_split, halo):
    """Two slab domains along x. Yields tuples of present indices and n_owned, owned first."""
    for owned in (positions_np[:, 0] < x_split, positions_np[:, 0] >= x_split):
        ghost = ~owned & (np.abs(positions_np[:, 0] - x_split) < halo)
        yield np.concatenate([np.nonzero(owned)[0], np.nonzero(ghost)[0]]), int(owned.sum())


class Graph:
    """Minimal neighbor-graph duck type with every edge valid."""

    def __init__(self, senders, receivers):
        self.senders = jnp.asarray(senders)
        self.receivers = jnp.asarray(receivers)
        self.edge_mask = jnp.ones(len(senders), dtype=bool)


class RecordingComm:
    """Phase-A stand-in: pass features through, remembering the traced value."""

    def __init__(self):
        self.recorded = []

    def forward_comm(self, features):
        self.recorded.append(features)
        return features


class LookupComm:
    """Phase-B stand-in: return rows of the assembled global feature buffer.

    Ghost rows hold owner values, so jax.grad yields the reverse-comm adjoint.
    """

    def __init__(self, global_features, present):
        self.global_features = global_features
        self.present = jnp.asarray(present)

    def forward_comm(self, features):
        del features
        return self.global_features[self.present]


class NoComm:
    def forward_comm(self, features):
        return features


def call_model(model_fn, model_args):
    """Rebuild the graph from ABI args, the way the exported wrappers expect."""
    positions, species, nlocal, nghost, senders, receivers, edge_mask = model_args[:7]
    graph = LammpsNeighborList(senders=senders, receivers=receivers, edge_mask=edge_mask)
    return model_fn(positions, species, graph), graph, nlocal, nghost


def pair_energy(positions, species, graph):
    """Per-atom harmonic pair model: each masked edge adds 0.5*|rij|^2 to its sender."""
    del species
    safe_senders = jnp.where(graph.edge_mask, graph.senders, 0)
    safe_receivers = jnp.where(graph.edge_mask, graph.receivers, 0)
    rij = positions[safe_receivers] - positions[safe_senders]
    per_edge = jnp.sum(rij * rij, axis=-1)
    per_edge = 0.5 * jnp.where(graph.edge_mask, per_edge, 0.0)
    return jnp.zeros((positions.shape[0],), dtype=jnp.float32).at[safe_senders].add(per_edge)


def zero_force_fn(positions, species, graph):
    del species, graph
    return jnp.zeros_like(positions)


def export_and_load(path, **kwargs):
    """export_model to path, then parse and return the bundle JSON."""
    export_model(path=path, **kwargs)
    return json.loads(path.read_text())


def assert_comm_scheme_matches_reference(
    energy_fn,
    positions_np,
    species_np,
    *,
    cutoff,
    exchange,
    should_match,
    atol,
    scale_force_atol=False,
    x_split=3.0,
):
    """Check a one-cutoff ghost shell with optional per-layer exchange against the full graph.

    On a match, a 2-cutoff shell with no exchange must agree too: comm-vs-ghost equivalence.
    """
    n_atoms = len(positions_np)
    species = jnp.asarray(species_np, dtype=jnp.int32)

    full_senders, full_receivers = edges_within_cutoff(positions_np, cutoff)

    def reference_energy(positions):
        node = energy_fn(positions, species, Graph(full_senders, full_receivers), NoComm())
        return jnp.sum(node)

    ref_energy, ref_grad = jax.value_and_grad(reference_energy)(jnp.asarray(positions_np))
    ref_forces = -np.asarray(ref_grad)
    force_atol = atol * (float(np.max(np.abs(ref_forces))) + 1.0) if scale_force_atol else atol

    domains = []
    for present, n_owned in two_domains(positions_np, x_split, halo=cutoff):
        senders, receivers = edges_within_cutoff(
            positions_np[present], cutoff, senders_subset=np.arange(n_owned)
        )
        domains.append((present, n_owned, senders, receivers))

    def decomposed_energy(positions):
        if exchange:
            recorded = []
            for present, n_owned, senders, receivers in domains:
                recorder = RecordingComm()
                energy_fn(
                    positions[jnp.asarray(present)], species[jnp.asarray(present)],
                    Graph(senders, receivers), recorder,
                )
                recorded.append(recorder.recorded[0])
            global_features = jnp.zeros((n_atoms,) + recorded[0].shape[1:],
                                        recorded[0].dtype)
            for (present, n_owned, _s, _r), features in zip(domains, recorded):
                global_features = global_features.at[jnp.asarray(present[:n_owned])].set(
                    features[:n_owned]
                )
        total = jnp.float32(0.0)
        for present, n_owned, senders, receivers in domains:
            comm_obj = LookupComm(global_features, present) if exchange else NoComm()
            node = energy_fn(
                positions[jnp.asarray(present)], species[jnp.asarray(present)],
                Graph(senders, receivers), comm_obj,
            )
            total = total + jnp.sum(node[:n_owned])
        return total

    energy, grad = jax.value_and_grad(decomposed_energy)(jnp.asarray(positions_np))
    forces = -np.asarray(grad)


    if should_match:
        assert float(energy) == pytest.approx(float(ref_energy), abs=atol)
        np.testing.assert_allclose(forces, ref_forces, atol=force_atol)
    else:
        energy_close = float(energy) == pytest.approx(float(ref_energy), abs=atol)
        forces_close = np.allclose(forces, ref_forces, atol=force_atol)
        assert not (energy_close and forces_close)
        return

    def ghost_scheme_energy(positions):
        total = jnp.float32(0.0)
        for present, n_owned in two_domains(positions_np, x_split, halo=2 * cutoff):
            senders, receivers = edges_within_cutoff(positions_np[present], cutoff)
            node = energy_fn(
                positions[jnp.asarray(present)], species[jnp.asarray(present)],
                Graph(senders, receivers), NoComm(),
            )
            total = total + jnp.sum(node[:n_owned])
        return total

    mh_energy, mh_grad = jax.value_and_grad(ghost_scheme_energy)(jnp.asarray(positions_np))
    assert float(mh_energy) == pytest.approx(float(energy), abs=atol)
    np.testing.assert_allclose(-np.asarray(mh_grad), forces, atol=force_atol)


FEATURE_WIDTH = 4
N_SPECIES = 2


def toy_mp_params(seed: int = 0) -> dict[str, jax.Array]:
    key = jax.random.PRNGKey(seed)
    k0, k1, k2, k3 = jax.random.split(key, 4)
    scale = 1.0 / jnp.sqrt(jnp.float32(FEATURE_WIDTH))
    return {
        "embed": jax.random.normal(k0, (N_SPECIES, FEATURE_WIDTH), dtype=jnp.float32),
        "w1": scale * jax.random.normal(k1, (FEATURE_WIDTH, FEATURE_WIDTH), dtype=jnp.float32),
        "w2": scale * jax.random.normal(k2, (FEATURE_WIDTH, FEATURE_WIDTH), dtype=jnp.float32),
        "readout": scale * jax.random.normal(k3, (FEATURE_WIDTH,), dtype=jnp.float32),
    }


def aggregate_messages(n_atoms, senders, receivers, envelope, features):
    """Sender-side aggregation: sum neighbor features over an atom's own row."""
    messages = envelope[:, None] * features[receivers]
    return jnp.zeros((n_atoms, FEATURE_WIDTH), dtype=jnp.float32).at[senders].add(messages)


def make_toy_mp_energy(
    *, cutoff: float, params: dict[str, jax.Array] | None = None, communicating: bool = False
) -> Callable[..., Any]:
    """Build the per-atom energy callable in either form.

    Export the plain form with n_hops=2 and the communicating form with comm=True.
    """
    if params is None:
        params = toy_mp_params()

    def node_energies(positions, species, graph, comm=None):
        n_atoms = positions.shape[0]
        senders = jnp.where(graph.edge_mask, graph.senders, 0)
        receivers = jnp.where(graph.edge_mask, graph.receivers, 0)
        rij = positions[receivers] - positions[senders]
        r_sq = jnp.sum(rij * rij, axis=-1)
        cutoff_sq = jnp.float32(cutoff * cutoff)
        valid = graph.edge_mask & (r_sq > jnp.float32(1.0e-12)) & (r_sq < cutoff_sq)
        envelope = jnp.where(valid, (1.0 - jnp.where(valid, r_sq, 0.0) / cutoff_sq) ** 2, 0.0)
        h0 = params["embed"][jnp.clip(species, 0, N_SPECIES - 1)]
        m1 = aggregate_messages(n_atoms, senders, receivers, envelope, h0)
        h1 = jnp.tanh(h0 + m1 @ params["w1"])
        if comm is not None:
            h1 = comm.forward_comm(h1)
        m2 = aggregate_messages(n_atoms, senders, receivers, envelope, h1)
        return (h1 + m2 @ params["w2"]) @ params["readout"]

    if communicating:
        return node_energies

    def plain_energy(positions, species, graph):
        return node_energies(positions, species, graph)

    return plain_energy


def native_interpolate(values):
    """PairEAM::interpolate transcribed for reference coefficients.

    Kept independent of lammps_jax.eam.spline_coefficients."""
    f = np.asarray(values, dtype=np.float64)
    n = len(f)
    d = np.zeros(n)
    d[0] = f[1] - f[0]
    d[1] = 0.5 * (f[2] - f[0])
    d[n - 2] = 0.5 * (f[n - 1] - f[n - 3])
    d[n - 1] = f[n - 1] - f[n - 2]
    for m in range(2, n - 2):
        d[m] = ((f[m - 2] - f[m + 2]) + 8.0 * (f[m + 1] - f[m - 1])) / 12.0
    coefficients = np.zeros((n, 4))
    coefficients[:, 0] = f
    coefficients[:, 1] = d
    for m in range(n - 1):
        coefficients[m, 2] = 3.0 * (f[m + 1] - f[m]) - 2.0 * d[m] - d[m + 1]
        coefficients[m, 3] = d[m] + d[m + 1] - 2.0 * (f[m + 1] - f[m])
    return coefficients
