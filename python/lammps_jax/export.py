"""Export JAX energy or force callables as fixed-capacity VHLO bundles for pair_style jax/kk.

Padding edges carry senders = receivers = max_atoms with edge_mask false;
where-substitute masked indices and guard divisions or the gradient goes NaN.
"""

import base64
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from lammps_jax.comm import CUSTOM_CALL_TARGETS, Comm

BUNDLE_FORMAT = "lammps-jax-json"
# Distinct tag: plugins predating n_hops must reject, not mispack.
DISTRIBUTED_BUNDLE_FORMAT = "lammps-jax-json-distributed"
# Each pair packed once; a loader packing both directions double counts.
HALF_EDGE_BUNDLE_FORMAT = "lammps-jax-json-half-edge"
INPUT_LAYOUT = "sparse-edge"
ATOM_FORCE = "atom-force"
EDGE_FORCE = "edge-force"
FORCE_OUTPUTS = {ATOM_FORCE, EDGE_FORCE}
PRECISION = "float32"
PRECISIONS = ("float32", "float64")


class LammpsNeighborList(NamedTuple):
    """Sparse LAMMPS edge-list view exposed to exported models."""

    senders: jax.Array
    receivers: jax.Array
    edge_mask: jax.Array


def program_text(bundle: dict[str, Any], program: str) -> str:
    """Deserialize a bundle's portable VHLO bytecode back to StableHLO text.

    Returns an empty string when the program was not exported.
    """
    from jaxlib.mlir import ir
    from jaxlib.mlir.dialects import stablehlo

    blob = base64.b64decode(bundle["programs"][f"{program}_b64"])
    if not blob:
        return ""
    with ir.Context() as context:
        return str(stablehlo.deserialize_portable_artifact(context, blob))


def strip_debug_options(options: bytes) -> bytes:
    """Drop executable_build_options.debug_options from serialized CompileOptions.

    Shipped DebugOptions override the runtime's XLA flags; per
    compile_options.proto, executable_build_options = 3 and debug_options = 3.
    """

    def fields(buf: bytes):
        i = 0
        while i < len(buf):
            start = i
            tag, shift = 0, 0
            while True:
                byte = buf[i]
                i += 1
                tag |= (byte & 0x7F) << shift
                shift += 7
                if not byte & 0x80:
                    break
            wire = tag & 7
            if wire == 0:
                while buf[i] & 0x80:
                    i += 1
                i += 1
            elif wire == 1:
                i += 8
            elif wire == 5:
                i += 4
            elif wire == 2:
                length, shift = 0, 0
                while True:
                    byte = buf[i]
                    i += 1
                    length |= (byte & 0x7F) << shift
                    shift += 7
                    if not byte & 0x80:
                        break
                i += length
            else:
                raise ValueError(f"unsupported wire type {wire}")
            yield tag >> 3, wire, buf[start:i], buf[i - length:i] if wire == 2 else b""

    def encode_field(number: int, payload: bytes) -> bytes:
        tag, out = number << 3 | 2, bytearray()
        for value in (tag, len(payload)):
            while value > 0x7F:
                out.append(value & 0x7F | 0x80)
                value >>= 7
            out.append(value)
        return bytes(out) + payload

    result = bytearray()
    for number, wire, raw, payload in fields(options):
        if number == 3 and wire == 2:
            kept = b"".join(sub_raw for sub_number, _, sub_raw, _ in fields(payload)
                            if sub_number != 3)
            result += encode_field(3, kept)
        else:
            result += raw
    return bytes(result)


def abi_anchor(*values: jax.Array, dtype: Any = jnp.float32) -> jax.Array:
    """Keep fixed ABI inputs visible in exported StableHLO.

    JAX may prune unused arguments at export, but the native plugin always
    supplies the full fixed ABI.
    """

    anchor = jnp.zeros((), dtype=dtype)
    scale = jnp.finfo(dtype).tiny
    for value in values:
        anchor += scale * jnp.sum(jax.lax.stop_gradient(jnp.asarray(value, dtype=dtype)))
    return anchor


def wrap_energy_fn(
    energy_fn: Callable[..., Any],
    *,
    max_atoms: int,
    call_model: Callable[[Callable[..., Any], tuple[Any, ...]], tuple[Any, LammpsNeighborList, Any, Any]],
    owned_rows_only: bool = False,
    dtype: Any = jnp.float32,
) -> tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any]]:
    """Build force, energy, and fused wrappers from an energy callable.

    With owned_rows_only, energy_fn must return per-atom energies and only
    owned rows count toward the energy.
    """

    def export_energy(*args: Any) -> jax.Array:
        raw_energy, graph, nlocal, nghost = call_model(energy_fn, args)
        atom_index = jnp.arange(max_atoms)
        local_mask = atom_index < nlocal
        valid_mask = atom_index < (nlocal + nghost)
        raw_energy = jnp.asarray(raw_energy, dtype=dtype)
        zero = jnp.zeros((), dtype=dtype)
        if raw_energy.shape == ():
            if owned_rows_only:
                raise ValueError(
                    "ghost and communicating exports need per-atom energies "
                    "so ghost rows can be masked out"
                )
            local_energy = raw_energy
            total_energy = raw_energy
        elif raw_energy.shape == valid_mask.shape:
            local_energy = jnp.sum(jnp.where(local_mask, raw_energy, zero))
            total_energy = jnp.sum(jnp.where(valid_mask, raw_energy, zero))
        else:
            raise ValueError(
                "energy_fn must return a scalar or per-atom array with "
                f"shape {valid_mask.shape}; got {raw_energy.shape}"
            )
        if owned_rows_only:
            # Ghost rows duplicate neighbor-rank energies; count owned only.
            energy = local_energy
        else:
            has_ghost_sender = jnp.any(graph.edge_mask & (graph.senders >= nlocal))
            energy = jnp.where(has_ghost_sender, total_energy, local_energy)
        return energy + abi_anchor(*args, dtype=dtype)

    def export_energy_and_forces(*args: Any) -> tuple[jax.Array, jax.Array]:
        def neg_energy(pos: jax.Array) -> jax.Array:
            return -export_energy(pos, *args[1:])

        neg_energy_value, forces = jax.value_and_grad(neg_energy)(args[0])
        # Keeps the ABI traced; zero interactions must give zero force.
        force_anchor = abi_anchor(*args[1:], dtype=dtype)
        return -neg_energy_value, forces + force_anchor * jnp.zeros_like(forces)

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
    precision: str = PRECISION,
    uses_box: bool = False,
    force_output: str = ATOM_FORCE,
    newton: str = "any",
    n_hops: int = 1,
    comm: bool = False,
    half_edges: bool = False,
    custom_call_targets: tuple[str, ...] = (),
    n_species: int | None = None,
) -> dict[str, Any]:
    """Export a JAX model to a fixed-capacity sparse LAMMPS-JAX JSON bundle.

    Provide either energy_fn or force_fn; forces from an energy export are
    the negative position gradient of the summed energy.
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
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {list(PRECISIONS)}")
    if comm and precision == "float64":
        raise ValueError("communicating exports are float32-only")
    if n_species is not None and n_species < 1:
        raise ValueError("n_species must be positive when given")
    if precision == "float64" and not jax.config.jax_enable_x64:  # ty: ignore[unresolved-attribute]
        raise ValueError(
            "precision='float64' requires jax x64 mode, which is disabled; "
            "enable it with jax.config.update('jax_enable_x64', True) or the "
            "jax.enable_x64(True) context manager before calling export_model, "
            "otherwise the traced programs would silently truncate to float32"
        )
    if n_hops < 1:
        raise ValueError("n_hops must be at least 1")
    if n_hops > 1 and force_fn is not None:
        raise ValueError(
            "n_hops > 1 exports require autodiff forces from energy_fn; ghost-row "
            "force contributions of a direct force_fn have no defined convention"
        )
    if comm:
        if force_fn is not None and force_output != EDGE_FORCE:
            raise ValueError(
                "communicating exports take autodiff forces from energy_fn or "
                "per-edge forces; a direct atom force_fn has no ghost-row "
                "convention"
            )
        if n_hops != 1:
            raise ValueError(
                "communicating exports use a one-cutoff ghost shell; n_hops must be 1"
            )
    if half_edges and not (n_hops > 1 or comm):
        raise ValueError(
            "half_edges requires n_hops > 1 or a communicating export; the "
            "single-hop packer does not deduplicate edge directions"
        )
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
    n_hops = int(n_hops)
    dtype = jnp.float64 if precision == "float64" else jnp.float32
    scalar_i32 = jax.ShapeDtypeStruct((), jnp.int32)
    edge_i32 = jax.ShapeDtypeStruct((max_edges,), jnp.int32)
    args: tuple[Any, ...] = (
        jax.ShapeDtypeStruct((max_atoms, 3), dtype),  # positions
        jax.ShapeDtypeStruct((max_atoms,), jnp.int32),  # species
        scalar_i32,  # nlocal
        scalar_i32,  # nghost
        edge_i32,  # senders
        edge_i32,  # receivers
        jax.ShapeDtypeStruct((max_edges,), jnp.bool_),  # edge_mask
    )
    if uses_box:
        args += (jax.ShapeDtypeStruct((3, 3), dtype),)

    def call_model_with(
        model_fn: Callable[..., Any],
        model_args: tuple[Any, ...],
        comm_obj: Comm | None = None,
    ) -> tuple[Any, LammpsNeighborList, Any, Any]:
        positions, species, nlocal, nghost, senders, receivers, edge_mask = model_args[:7]
        graph = LammpsNeighborList(senders=senders, receivers=receivers, edge_mask=edge_mask)
        extra = (model_args[7],) if uses_box else ()
        if comm_obj is not None:
            value = model_fn(positions, species, graph, *extra, comm_obj)
            comm_obj.validate()
        else:
            value = model_fn(positions, species, graph, *extra)
        return value, graph, nlocal, nghost

    # Staging is sized from this width trace; per-callable schedules concatenate.
    comm_widths: tuple[int, ...] = ()
    if comm:
        for fn in (energy_fn, force_fn):
            if fn is None:
                continue
            recorder = Comm(enabled=False)
            jax.eval_shape(lambda *a, fn=fn: call_model_with(fn, a, recorder)[0], *args)  # ty: ignore[invalid-argument-type]
            comm_widths += tuple(recorder.widths)
        if not comm_widths:
            raise ValueError(
                "comm=True but the model never called comm.forward_comm; export "
                "without comm or add sites at the message-passing boundaries"
            )

    def call_model(
        model_fn: Callable[..., Any], model_args: tuple[Any, ...]
    ) -> tuple[Any, LammpsNeighborList, Any, Any]:
        if comm:
            # Fresh per trace: token and width record are trace-local.
            comm_obj = Comm(enabled=True, expected_widths=comm_widths)
            return call_model_with(model_fn, model_args, comm_obj)
        return call_model_with(model_fn, model_args)

    force_program = None
    energy_program = None
    fused_program = None
    if energy_fn is not None:
        force_program, energy_program, fused_program = wrap_energy_fn(
            energy_fn,
            max_atoms=max_atoms,
            call_model=call_model,
            owned_rows_only=n_hops > 1 or comm,
            dtype=dtype,
        )
    if force_fn is not None:
        def export_forces(*model_args: Any) -> jax.Array:
            forces, _graph, _nlocal, _nghost = call_model(force_fn, model_args)
            forces = jnp.asarray(forces, dtype=dtype)
            expected_shape = (max_edges, 3) if force_output == EDGE_FORCE else (max_atoms, 3)
            if forces.shape != expected_shape:
                raise ValueError(
                    "force_fn must return an array with shape "
                    f"{expected_shape}; got {forces.shape}"
                )
            return forces + abi_anchor(*model_args, dtype=dtype) * jnp.ones_like(forces)

        force_program = jax.jit(export_forces)
        if energy_program is not None:
            fused_program = jax.jit(lambda *a: (energy_program(*a), force_program(*a)))
    if force_program is None:
        raise AssertionError("validated export must have a force wrapper")

    disabled_checks = tuple(
        jax.export.DisabledSafetyCheck.custom_call(target)
        for target in (CUSTOM_CALL_TARGETS if comm else ()) + tuple(custom_call_targets)
    )

    # Portable VHLO bytecode; PJRT's mlir format auto-detects it.
    def export_mlir(fn: Callable[..., Any] | None) -> str:
        if fn is None:
            return ""
        # Highest matmul precision: TF32 dot_generals would soften forces.
        with jax.default_matmul_precision("highest"):
            exported = jax.export.export(fn, platforms=("cuda",), disabled_checks=disabled_checks)(  # ty: ignore[invalid-argument-type]
                *args
            )
        return base64.b64encode(exported.mlir_module_serialized).decode("ascii")

    from jaxlib import xla_client

    compile_options = xla_client.CompileOptions()
    compile_options.num_replicas = 1
    compile_options.num_partitions = 1
    # Portable: run on the plugin's device instead of compile-time device 0.
    compile_options.compile_portable_executable = True

    distributed = n_hops > 1 or comm
    bundle = {
        "format": HALF_EDGE_BUNDLE_FORMAT if half_edges
        else DISTRIBUTED_BUNDLE_FORMAT if distributed else BUNDLE_FORMAT,
        "programs": {
            "force_mlir_b64": export_mlir(force_program),
            "energy_mlir_b64": export_mlir(energy_program),
            "energy_and_forces_mlir_b64": export_mlir(fused_program),
        },
        "export_info": {
            "jax_version": jax.__version__,
        },
        "compile_options_b64": base64.b64encode(
            strip_debug_options(compile_options.SerializeAsString())).decode("ascii"),
        "contract": {
            "input_layout": INPUT_LAYOUT,
            "max_atoms": max_atoms,
            "max_edges": max_edges,
            "cutoff": cutoff,
            "unit_style": unit_style,
            "precision": precision,
            "force_output": force_output,
            "newton": newton,
            "n_hops": n_hops,
            "edge_pairing": "half" if half_edges else "full",
            "comm_widths": list(comm_widths),
            # "__gpu$" handlers ship in the plugin; recording them would demand a mapping.
            "custom_call_targets": sorted(
                target for target in custom_call_targets
                if not target.startswith("__gpu$")
            ),
            "uses_box": uses_box,
        },
    }
    if n_species is not None:
        bundle["contract"]["n_species"] = n_species
    Path(path).write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return bundle
