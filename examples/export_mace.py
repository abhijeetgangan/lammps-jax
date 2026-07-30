"""MACE-MP-0 small in one file: torch-free loader, exporter, and torch converter.

Import installs cg_cache.npz shims from $MACE_MP_BUNDLE_DIR when the cache
exists; misses fall back to torch-e3nn and convert writes the recorded cache
into its output bundle. Subcommands: export, convert, check.
"""

import argparse
import ctypes
import json
import os
import sys
import sysconfig
from pathlib import Path

import jax
import jax.numpy as jnp
import jraph
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE_DIR = os.path.normpath(os.environ.get(
    "MACE_MP_BUNDLE_DIR",
    os.path.join(HERE, "..", "..", "models", "mace-mp-0-small-jax"),
))

# Serve wigner_3j and the symmetric-contraction CG transform from
# cg_cache.npz; keys missing from the cache fall back to torch-e3nn.
import mace_jax.adapters.cuequivariance.symmetric_contraction as sc
import mace_jax.tools.cg as cg

cache_path = os.path.join(BUNDLE_DIR, "cg_cache.npz")
CG_CACHE = dict(np.load(cache_path)) if os.path.exists(cache_path) else {}
orig_wigner_3j = cg.wigner_3j
orig_full_cg = sc._cached_full_cg_transform


def cached_wigner_3j(l1, l2, l3, dtype=None):
    key = f"w:{int(l1)},{int(l2)},{int(l3)}"
    if key not in CG_CACHE:
        CG_CACHE[key] = np.asarray(orig_wigner_3j(l1, l2, l3, dtype=dtype))
    return jnp.asarray(CG_CACHE[key], dtype=dtype)


def cached_full_cg(irreps_in_str, irreps_out_str, correlation):
    key = f"f:{irreps_in_str}|{irreps_out_str}|{int(correlation)}"
    if key not in CG_CACHE:
        CG_CACHE[key] = np.asarray(
            orig_full_cg(irreps_in_str, irreps_out_str, correlation))
    return np.asarray(CG_CACHE[key])


cg.wigner_3j = cached_wigner_3j  # ty: ignore[invalid-assignment]
sc._cached_full_cg_transform = cached_full_cg  # ty: ignore[invalid-assignment]

from flax import nnx  # noqa: E402
from mace_jax.data.utils import (  # noqa: E402
    AtomicNumberTable,
    config_from_atoms,
    graph_from_configuration,
)
from mace_jax.tools import gin_model  # noqa: E402
from mace_jax.tools.bundle import load_model_bundle  # noqa: E402

from lammps_jax.export import LammpsNeighborList, export_model  # noqa: E402
from lammps_jax.mace import make_mace_energy  # noqa: E402


def load_model(dtype: str = "float64"):
    """Rebuild the ModelBundle from the bundle directory, torch-free.

    load_model_bundle toggles jax_enable_x64 process-wide to match dtype.
    """
    return load_model_bundle(BUNDLE_DIR, dtype)


def make_energy_forces_fn(bundle):
    """Build energy_forces(ase_atoms) -> (energy, forces) in eV and eV/A.

    Graphs use the model r_max and first head, jraph-padded; padding
    rows are stripped from the returned forces.
    """
    cfg = bundle.config
    num_species = len(cfg["atomic_numbers"])
    z_table = AtomicNumberTable([int(z) for z in cfg["atomic_numbers"]])
    r_max = float(cfg["r_max"])
    heads = [str(h) for h in (cfg.get("heads") or ["default"])]
    head_to_index = {h: i for i, h in enumerate(heads)}

    def energy_forces(atoms):
        conf = config_from_atoms(atoms, head_name=heads[0])
        graph = graph_from_configuration(
            conf, cutoff=r_max, z_table=z_table, head_to_index=head_to_index
        )
        n_node = int(np.asarray(graph.n_node).sum())
        n_edge = int(np.asarray(graph.n_edge).sum())
        padded = jraph.pad_with_graphs(
            graph, n_node=n_node + 1, n_edge=n_edge + 16, n_graph=2
        )
        data_dict = gin_model._graph_to_data(padded, num_species=num_species)
        outputs, _ = bundle.graphdef.apply(bundle.params)(
            data_dict, compute_force=True, compute_stress=False
        )
        energy = float(np.asarray(outputs["energy"])[0])
        forces = np.asarray(outputs["forces"])[:n_node]
        return energy, forces

    return energy_forces


def convert(args):
    # Convert in float64 to preserve the torch f64 weights.
    jax.config.update("jax_enable_x64", True)

    import torch
    from flax import serialization
    from mace.tools.scripts_utils import extract_config_mace_model

    import mace_jax.cli.mace_jax_from_torch as mjft
    from mace_jax.cli.mace_jax_from_torch import _serialize_for_json, convert_model

    # mace_jax 0.2.0 bug: _serialize_for_json references `torch`, which the
    # module only imports under TYPE_CHECKING.
    mjft.torch = torch  # ty: ignore[invalid-assignment]
    from mace_jax.nnx_utils import state_to_serializable_dict

    ckpt = Path(args.checkpoint)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch_model = torch.load(ckpt, map_location="cpu", weights_only=False)
    torch_model = torch_model.double()
    torch_model.eval()

    print("torch model class:", torch_model.__class__.__name__)
    p0 = next(torch_model.parameters())
    print("torch param dtype:", p0.dtype)

    config = extract_config_mace_model(torch_model)
    if "error" in config:
        raise RuntimeError(config["error"])
    config["torch_model_class"] = torch_model.__class__.__name__

    print("extracted config, key hyperparameters:")
    for key in (
        "r_max", "num_bessel", "num_polynomial_cutoff", "max_ell",
        "num_interactions", "hidden_irreps", "MLP_irreps", "correlation",
        "avg_num_neighbors", "interaction_cls", "interaction_cls_first",
        "radial_type", "radial_MLP", "gate", "distance_transform",
        "atomic_inter_scale", "atomic_inter_shift", "heads",
    ):
        if key in config:
            print(f"{key}: {config[key]}")
    print("num atomic_numbers (species):", len(config["atomic_numbers"]))
    print("atomic_numbers:", config["atomic_numbers"])
    ae = np.asarray(config["atomic_energies"])
    print("atomic_energies shape:", ae.shape, "first 5:", ae.ravel()[:5])

    # Same converter entry point chemtrain uses.
    if args.conv_fusion:
        from mace_jax.modules.wrapper_ops import CuEquivarianceConfig
        fused_cfg = CuEquivarianceConfig(enabled=False, optimize_channelwise=True,
                                         conv_fusion=True, layout="mul_ir")
        graphdef, state, template_data = convert_model(torch_model, config,
                                                       cueq_config=fused_cfg)
        config["cue_conv_fusion"] = True  # torch-free rebuild must construct the fused TP
    else:
        graphdef, state, template_data = convert_model(torch_model, config, cueq_config=None)

    variables = state_to_serializable_dict(state)

    leaf = next(((jax.tree_util.keystr(path), value.dtype)
                 for path, value in jax.tree_util.tree_leaves_with_path(variables)
                 if hasattr(value, "dtype") and jnp.issubdtype(value.dtype, jnp.floating)),
                None)
    print("example jax param leaf/dtype:", leaf)

    params_bytes = serialization.to_bytes(variables)
    (out_dir / "params.msgpack").write_bytes(params_bytes)

    # convert_model stored the torch normalize2mom constants in `config`, so
    # a torch-free rebuild matches the activation normalization.
    config_json = _serialize_for_json(config)
    (out_dir / "config.json").write_text(json.dumps(config_json, indent=2))

    # Model construction above ran through the recording CG shims.
    np.savez(out_dir / "cg_cache.npz", **CG_CACHE)

    print("normalize2mom_consts in config:", config_json.get("normalize2mom_consts"))
    print("wrote", out_dir / "params.msgpack", len(params_bytes), "bytes")
    print("wrote", out_dir / "config.json")
    print("wrote", out_dir / "cg_cache.npz", f"({len(CG_CACHE)} entries)")


def run_check():
    from ase.build import molecule

    bundle = load_model("float64")
    print("rebuilt mace-jax model:", bundle.config.get("torch_model_class"))
    print(
        "r_max=%s hidden_irreps=%s num_interactions=%s correlation=%s species=%d"
        % (
            bundle.config["r_max"],
            bundle.config["hidden_irreps"],
            bundle.config["num_interactions"],
            bundle.config["correlation"],
            len(bundle.config["atomic_numbers"]),
        )
    )

    h2o = molecule("H2O")
    h2o.set_cell([10.0, 10.0, 10.0])
    h2o.center()
    h2o.pbc = False

    e, f = make_energy_forces_fn(bundle)(h2o)
    print(f"H2O energy: {e:.8f} eV")
    print("H2O forces (eV/A):")
    np.set_printoptions(precision=6, suppress=True)
    print(np.asarray(f))

    # Check no torch/mace was imported.
    contraband = sorted(
        m for m in sys.modules if m.split(".")[0] in ("torch", "mace")
    )
    print("torch/mace modules imported:", contraband or "NONE")


def handler_offsets(targets):
    """Offset of each registered handler inside the extjax extension."""
    import openequivariance_extjax as extjax

    get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
    get_pointer.restype = ctypes.c_void_p
    get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    library = extjax.__file__
    base = None
    with open("/proc/self/maps") as maps:
        for line in maps:
            if os.path.basename(library) in line:
                base = int(line.split("-")[0], 16)
                break
    capsules = extjax.registrations()
    return library, {t: get_pointer(capsules[t], None) - base for t in targets}


def write_runtime_env(output, targets):
    """Write <output>.env: slot handlers, libpython preload, and offset guards."""
    library, offsets = handler_offsets(targets)
    libpython = os.path.join(sysconfig.get_config_var("LIBDIR"),
                             sysconfig.get_config_var("INSTSONAME"))
    shim = "/path/ffi_replay_shim.so"
    handlers = ";".join(f"{t}={shim}:ffi_handler_{i}" for i, t in enumerate(targets))
    size = os.path.getsize(library)
    lines = [f"export LAMMPS_JAX_FFI_HANDLERS='{handlers}'",
             f"export FFI_PRELOAD_LIBS={libpython}"]
    lines += [f"export FFI_HANDLER_{i}={library}+{offsets[t]:#x}@{size}"
              for i, t in enumerate(targets)]
    env_path = output + ".env"
    with open(env_path, "w") as env_file:
        env_file.write("\n".join(lines) + "\n")
    return env_path


def install_oeq_conv_tp():
    """Swap the channelwise conv TP for OpenEquivariance's CUDA kernel.

    Patches the adapter class method, so models built before or after the
    call pick up the kernel at trace time. CUDA-only; keep CPU checks
    ahead of this call.
    """
    os.environ.setdefault("OEQ_NOTORCH", "1")
    import mace_jax.adapters.cuequivariance.tensor_product as cue_tp
    from openequivariance.core.e3nn_lite import TPProblem
    from openequivariance.jax import TensorProduct as OeqTensorProduct
    from openequivariance.jax import TensorProductConv as OeqTensorProductConv

    kernels = {}

    def kernel_for(self, dtype, conv):
        instructions = self.instructions
        if instructions is None:
            _, instructions = cue_tp._expected_channelwise_instructions(
                self.irreps_in1_o3, self.irreps_in2_o3, self.irreps_out_o3)
        key = (str(self.irreps_in1_o3), str(self.irreps_in2_o3),
               str(self.irreps_out_o3), str(np.dtype(dtype)), conv)
        if key not in kernels:
            problem = TPProblem(
                key[0], key[1], key[2],
                [tuple(inst)[:5] for inst in instructions],
                shared_weights=False, internal_weights=False,
                irrep_dtype=np.dtype(dtype).type, weight_dtype=np.dtype(dtype).type)
            # Eager construction: the kernel holds jnp constants, and the e3nn
            # weight layout map is probed into a fixed permutation so each call
            # runs one gather instead of the reorder slice chain.
            with jax.ensure_compile_time_eval():
                if conv:
                    kernel = OeqTensorProductConv(problem, deterministic=False)
                else:
                    kernel = OeqTensorProduct(problem)
                probe = jnp.arange(problem.weight_numel, dtype=jnp.float32)[None]
                perm = np.asarray(kernel.reorder_weights_from_e3nn(probe))[0].astype(np.int64)
            # AssertionError: the adapter swallows RuntimeError into a fallback.
            assert np.array_equal(np.sort(perm), np.arange(problem.weight_numel)), \
                "OEQ weight reordering is not a pure permutation"
            kernels[key] = (kernel, perm)
        return kernels[key]

    def oeq_channelwise(self, x1, x2, weight_tensor, *, dtype):
        kernel, perm = kernel_for(self, dtype, conv=False)
        # The incoming per-edge weights use the e3nn flat layout.
        return kernel(x1, x2, weight_tensor[:, perm])

    def oeq_conv_fused(self, *, node_feats, edge_attrs, weights, sender, receiver,
                       num_nodes, dtype):
        kernel, perm = kernel_for(self, dtype, conv=True)
        # Atomic kernels skip XLA's out-of-range index protection; padded edge
        # slots get index 0 and zero weights.
        valid = sender < num_nodes
        rows = jnp.where(valid, receiver, 0).astype(jnp.int32)
        cols = jnp.where(valid, sender, 0).astype(jnp.int32)
        masked = jnp.where(valid[:, None], weights[:, perm], 0)
        return kernel(node_feats, edge_attrs, masked, rows, cols)

    cue_tp.TensorProduct._channelwise_apply = oeq_channelwise
    cue_tp.TensorProduct._conv_fused_apply = oeq_conv_fused


def run_export(args):
    if args.backend == "cueq":
        sys.path.insert(0, os.path.join(HERE, "..", "contrib", "ffi-replay"))
        import cuequivariance_ops_jax._common as cc
        import ffi_replay_record  # ty: ignore[unresolved-import]
        ffi_replay_record.install(cc.library, "compile_uniform_1d_ctypes")
        targets = ("uniform_1d_cuda",)
        default_dir = BUNDLE_DIR + "-fused"
    elif args.backend == "oeq":
        # The fused bundle routes the adapter through the conv path.
        targets = ("conv_forward", "conv_backward")
        default_dir = BUNDLE_DIR + "-fused"
    else:
        targets = ()
        default_dir = BUNDLE_DIR

    bundle_dir = args.bundle_dir or default_dir
    extra_cache = os.path.join(bundle_dir, "cg_cache.npz")
    if os.path.exists(extra_cache):
        CG_CACHE.update(np.load(extra_cache))

    model_bundle = load_model_bundle(bundle_dir, "float32")
    cfg = model_bundle.config
    model = nnx.merge(model_bundle.graphdef,
                      model_bundle.params)  # ty: ignore[no-matching-overload]

    z_table = [int(z) for z in cfg["atomic_numbers"]]
    type_table = jnp.asarray([z_table.index(z) for z in args.type_z], jnp.int32)

    if not args.skip_check:
        # Adapter vs mace_jax on a small free cluster of the first element;
        # on CPU the accelerated TPs fall back to naive, same math.
        from ase import Atoms
        ref = make_energy_forces_fn(model_bundle)
        rng = np.random.default_rng(3)
        pos = (np.array([[i, j, k] for i in range(2) for j in range(2) for k in range(2)],
                        dtype=np.float64) * 2.86 + rng.normal(0, 0.08, (8, 3)))
        from ase.data import chemical_symbols
        e_ref, f_ref = ref(Atoms(chemical_symbols[args.type_z[0]] * 8,
                                 positions=pos, pbc=False))
        fn = make_mace_energy(config=cfg, model=model, communicating=False)
        d = pos[None] - pos[:, None]
        r = np.sqrt((d ** 2).sum(-1))
        snd, rcv = np.nonzero((r < float(cfg["r_max"])) & ~np.eye(8, dtype=bool))

        graph = LammpsNeighborList(
            senders=jnp.asarray(snd, jnp.int32),
            receivers=jnp.asarray(rcv, jnp.int32),
            edge_mask=jnp.ones(len(snd), bool),
        )
        species = jnp.full((8,), z_table.index(args.type_z[0]), jnp.int32)
        e_ad, grad = jax.value_and_grad(
            lambda p: jnp.sum(fn(p, species, graph)))(jnp.asarray(pos, jnp.float32))
        df = float(np.max(np.abs(-np.asarray(grad) - f_ref)))
        print(f"preflight: dE/atom = {(float(e_ad) - e_ref) / 8:+.2e} eV, "
              f"max|dF| = {df:.3e} eV/A")
        assert abs(float(e_ad) - e_ref) / 8 < 5e-4 and df < 5e-3, "adapter parity failed"

    if args.backend == "oeq":
        install_oeq_conv_tp()

    communicating = args.mode == "comm"
    energy_fn = make_mace_energy(config=cfg, model=model, communicating=communicating)
    n_types = len(args.type_z)

    if communicating:
        def wrapped(p, s, graph, comm):
            return energy_fn(p, type_table[jnp.clip(s, 0, n_types - 1)], graph, comm)
    else:
        def wrapped(p, s, graph):
            return energy_fn(p, type_table[jnp.clip(s, 0, n_types - 1)], graph)

    export_model(
        energy_fn=wrapped,
        path=args.output,
        max_atoms=args.max_atoms,
        max_edges=args.max_edges,
        cutoff=float(cfg["r_max"]),
        unit_style="metal",
        comm=communicating,
        n_hops=1 if communicating else int(cfg["num_interactions"]),
        custom_call_targets=targets,
        n_species=n_types,
    )
    contract = json.load(open(args.output))["contract"]
    print(f"exported: targets={contract['custom_call_targets']} -> {args.output}")
    if args.backend == "cueq":
        # Keep the blob path distinct from the bundle path.
        if args.output.endswith(".lammps-jax.json"):
            blob = args.output[: -len(".lammps-jax.json")] + ".ffi-replay"
        else:
            blob = args.output + ".ffi-replay"
        n = ffi_replay_record.dump(blob, "compile_uniform_1d_ctypes",
                                   "execute_uniform_1d_cuda_handler")
        print(f"replay blob: {n} plans -> {blob}")
    elif args.backend == "oeq":
        env_path = write_runtime_env(args.output, targets)
        print(f"runtime environment -> {env_path}; edit the shim path, then source it")


def main():
    # Exports run on CPU unless the caller sets JAX_PLATFORMS.
    if "JAX_PLATFORMS" not in os.environ:
        jax.config.update("jax_platforms", "cpu")
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    exporter = sub.add_parser("export", help="write a pair_style jax/kk bundle")
    exporter.add_argument("backend", choices=("plain", "cueq", "oeq"),
                        help="plain: pure-XLA program; cueq: keep the NVIDIA "
                             "uniform_1d kernel and write a .ffi-replay sidecar "
                             "(contrib/ffi-replay); oeq: run the channelwise "
                             "tensor product through OpenEquivariance's fused "
                             "conv kernel and write a .env runtime sidecar")
    exporter.add_argument("output")
    exporter.add_argument("--mode", choices=("comm", "ghost"), default="comm",
                        help="comm: in-model feature exchange, one-cutoff ghost "
                             "shell; ghost: no exchange, extended ghost shell")
    exporter.add_argument("--type-z", type=int, nargs="+", default=[13],
                        help="atomic number per LAMMPS type, type 1 first; types "
                             "past the end of the list are clipped to the last "
                             "entry")
    exporter.add_argument("--max-atoms", type=int, default=2048,
                        help="per-rank row capacity, owned plus ghost atoms")
    exporter.add_argument("--max-edges", type=int, default=None,
                        help="per-rank packed-edge capacity; default "
                             "max-atoms * edges-per-atom")
    exporter.add_argument("--edges-per-atom", type=int, default=16,
                        help="edge capacity per atom row, used when --max-edges "
                             "is unset")
    exporter.add_argument("--bundle-dir", default=None,
                        help="converted-model dir; default $MACE_MP_BUNDLE_DIR or "
                             "<workspace>/models/mace-mp-0-small-jax, -fused "
                             "variant for cueq and oeq")
    exporter.add_argument("--skip-check", action="store_true",
                        help="skip the adapter-vs-mace_jax parity preflight")
    converter = sub.add_parser("convert", help="torch checkpoint to a mace-jax bundle dir")
    converter.add_argument("checkpoint", help="MACE-MP torch checkpoint")
    converter.add_argument("out", help="output bundle directory")
    converter.add_argument("--conv-fusion", action="store_true",
                           help="record cue_conv_fusion=True so the torch-free "
                                "rebuild constructs the fused tensor product")
    sub.add_parser("check", help="torch-free H2O evaluation of the loaded bundle")
    args = parser.parse_args()
    if args.command == "export":
        if args.max_edges is None:
            args.max_edges = args.max_atoms * args.edges_per_atom
        run_export(args)
    elif args.command == "convert":
        convert(args)
    else:
        run_check()


if __name__ == "__main__":
    main()
