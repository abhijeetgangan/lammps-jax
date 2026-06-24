"""StableHLO exporter for CUDA-first LAMMPS/Kokkos execution."""

import base64
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

BUNDLE_FORMAT = "lammps-jax-json"
INPUT_LAYOUT = "sparse-edge"
ATOM_FORCE = "atom-force"
EDGE_FORCE = "edge-force"
FORCE_OUTPUTS = {ATOM_FORCE, EDGE_FORCE}
PRECISION = "float32"


class LammpsNeighborList(NamedTuple):
    """Sparse LAMMPS edge-list view exposed to exported models."""

    senders: jax.Array
    receivers: jax.Array
    edge_mask: jax.Array


def _default_compile_options() -> bytes:
    from jaxlib import xla_client

    compile_options = xla_client.CompileOptions()
    compile_options.num_replicas = 1
    compile_options.num_partitions = 1
    # Portable executables run on whichever device the plugin executes on
    # instead of assuming device 0 from compile time.
    compile_options.compile_portable_executable = True
    return compile_options.SerializeAsString()


def _abi_anchor(*values: jax.Array) -> jax.Array:
    """Keep fixed ABI inputs visible in exported StableHLO.

    JAX is allowed to remove unused function arguments during export. The native
    plugin, however, always supplies the full fixed ABI. A tiny stop-gradient
    term keeps all arguments in the StableHLO signature without changing forces.
    """

    anchor = jnp.float32(0.0)
    scale = jnp.finfo(jnp.float32).tiny
    for value in values:
        anchor += scale * jnp.sum(jax.lax.stop_gradient(jnp.asarray(value, dtype=jnp.float32)))
    return anchor


def wrap_energy_fn(
    energy_fn: Callable[..., Any],
    *,
    max_atoms: int,
    call_model: Callable[[Callable[..., Any], tuple[Any, ...]], tuple[Any, LammpsNeighborList, Any, Any]],
) -> tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any]]:
    """Build force, energy, and fused wrappers from an energy callable."""

    def export_energy(*args: Any) -> jax.Array:
        raw_energy, graph, nlocal, nghost = call_model(energy_fn, args)
        atom_index = jnp.arange(max_atoms)
        local_mask = atom_index < nlocal
        valid_mask = atom_index < (nlocal + nghost)
        raw_energy = jnp.asarray(raw_energy, dtype=jnp.float32)
        if raw_energy.shape == ():
            local_energy = raw_energy
            total_energy = raw_energy
        elif raw_energy.shape == valid_mask.shape:
            local_energy = jnp.sum(jnp.where(local_mask, raw_energy, jnp.float32(0.0)))
            total_energy = jnp.sum(jnp.where(valid_mask, raw_energy, jnp.float32(0.0)))
        else:
            raise ValueError(
                "energy_fn must return a scalar or per-atom array with "
                f"shape {valid_mask.shape}; got {raw_energy.shape}"
            )
        has_ghost_sender = jnp.any(graph.edge_mask & (graph.senders >= nlocal))
        energy = jnp.where(has_ghost_sender, total_energy, local_energy)
        return energy + _abi_anchor(*args)

    def export_energy_and_forces(*args: Any) -> tuple[jax.Array, jax.Array]:
        def neg_energy(pos: jax.Array) -> jax.Array:
            return -export_energy(pos, *args[1:])

        neg_energy_value, forces = jax.value_and_grad(neg_energy)(args[0])
        force_anchor = _abi_anchor(*args[1:])
        return -neg_energy_value, forces + force_anchor * jnp.ones_like(forces)

    def export_forces(*args: Any) -> jax.Array:
        return export_energy_and_forces(*args)[1]

    return jax.jit(export_forces), jax.jit(export_energy), jax.jit(export_energy_and_forces)


def export_model(
    *,
    energy_fn: Callable[..., Any] | None = None,
    force_fn: Callable[..., Any] | None = None,
    path: str | Path,
    max_atoms: int,
    max_edges: int,
    cutoff: float,
    unit_style: str = "real",
    uses_box: bool = False,
    force_output: str = ATOM_FORCE,
    newton: str = "any",
    disabled_checks: tuple[Any, ...] = (),
) -> dict[str, Any]:
    """Export a JAX model to a fixed-capacity sparse LAMMPS-JAX JSON bundle.

    Provide either:
    - `force_fn`, which returns forces with shape `(max_atoms, 3)` for
      atom-force output or `(max_edges, 3)` for edge-force output, or
    - `energy_fn`, which returns scalar or per-atom energies. If no direct
      `force_fn` is supplied, forces are generated with `jax.grad`.

    Energy-only autodiff exports require `newton on`. Edge-force bundles that
    also include energy must record the target `newton` mode because pair-energy
    normalization follows the neighbor-list convention.
    """
    if energy_fn is None and force_fn is None:
        raise ValueError("provide energy_fn or force_fn")
    if max_atoms <= 0:
        raise ValueError("max_atoms must be positive")
    if max_edges <= 0:
        raise ValueError("max_edges must be positive")
    if cutoff < 0:
        raise ValueError("cutoff must be non-negative")
    if force_output not in FORCE_OUTPUTS:
        raise ValueError(f"force_output must be one of {sorted(FORCE_OUTPUTS)}")
    if newton not in {"on", "off", "any"}:
        raise ValueError("newton must be 'on', 'off', or 'any'")
    if force_output == EDGE_FORCE and force_fn is None:
        raise ValueError("edge-force output requires a direct force_fn")
    if force_output == EDGE_FORCE and energy_fn is not None and newton == "any":
        raise ValueError(
            "edge-force bundles with an energy callable use newton-dependent "
            "per-edge energy normalization; export with newton='on' or newton='off'"
        )
    if force_fn is None:
        if newton == "off":
            raise ValueError(
                "energy-only exports produce autodiff forces that are only correct "
                "with LAMMPS newton on; export with newton='on' or provide a "
                "direct force_fn"
            )
        newton = "on"

    max_atoms = int(max_atoms)
    max_edges = int(max_edges)
    cutoff = float(cutoff)
    scalar_i32 = jax.ShapeDtypeStruct((), jnp.int32)
    edge_i32 = jax.ShapeDtypeStruct((max_edges,), jnp.int32)
    args: tuple[Any, ...] = (
        jax.ShapeDtypeStruct((max_atoms, 3), jnp.float32),  # positions
        jax.ShapeDtypeStruct((max_atoms,), jnp.int32),  # species
        scalar_i32,  # nlocal
        scalar_i32,  # nghost
        edge_i32,  # senders
        edge_i32,  # receivers
        jax.ShapeDtypeStruct((max_edges,), jnp.bool_),  # edge_mask
    )
    if uses_box:
        args += (jax.ShapeDtypeStruct((3, 3), jnp.float32),)

    def call_model(
        model_fn: Callable[..., Any], model_args: tuple[Any, ...]
    ) -> tuple[Any, LammpsNeighborList, Any, Any]:
        positions, species, nlocal, nghost, senders, receivers, edge_mask = model_args[:7]
        graph = LammpsNeighborList(senders=senders, receivers=receivers, edge_mask=edge_mask)
        if uses_box:
            value = model_fn(positions, species, graph, model_args[7])
        else:
            value = model_fn(positions, species, graph)
        return value, graph, nlocal, nghost

    force_program = None
    energy_program = None
    fused_program = None
    if energy_fn is not None:
        force_program, energy_program, fused_program = wrap_energy_fn(
            energy_fn, max_atoms=max_atoms, call_model=call_model
        )
    if force_fn is not None:
        def export_forces(*model_args: Any) -> jax.Array:
            forces, _graph, _nlocal, _nghost = call_model(force_fn, model_args)
            forces = jnp.asarray(forces, dtype=jnp.float32)
            expected_shape = (max_edges, 3) if force_output == EDGE_FORCE else (max_atoms, 3)
            if forces.shape != expected_shape:
                raise ValueError(
                    "force_fn must return an array with shape "
                    f"{expected_shape}; got {forces.shape}"
                )
            return forces + _abi_anchor(*model_args) * jnp.ones_like(forces)

        force_program = jax.jit(export_forces)
        if energy_program is not None:
            fused_program = jax.jit(lambda *a: (energy_program(*a), force_program(*a)))
    if force_program is None:
        raise AssertionError("validated export must have a force wrapper")

    def export_mlir(fn: Callable[..., Any] | None) -> str:
        if fn is None:
            return ""
        return jax.export.export(fn, platforms=("cuda",), disabled_checks=disabled_checks)(
            *args
        ).mlir_module()

    bundle = {
        "format": BUNDLE_FORMAT,
        "programs": {
            "force_mlir": export_mlir(force_program),
            "energy_mlir": export_mlir(energy_program),
            "energy_and_forces_mlir": export_mlir(fused_program),
        },
        "compile_options_b64": base64.b64encode(_default_compile_options()).decode("ascii"),
        "contract": {
            "input_layout": INPUT_LAYOUT,
            "max_atoms": max_atoms,
            "max_edges": max_edges,
            "cutoff": cutoff,
            "unit_style": unit_style,
            "precision": PRECISION,
            "force_output": force_output,
            "newton": newton,
            "uses_box": uses_box,
        },
    }
    Path(path).write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return bundle
