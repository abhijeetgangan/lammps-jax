# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "lammps-jax",
#   "torchax==0.0.13",
#   "torch==2.8.0+cpu",
#   "jax[cuda12]==0.10.1",
#   "nequip==0.19.0",
#   "openequivariance==0.6.8",
# ]
#
# [tool.uv]
# index-strategy = "unsafe-best-match"
#
# [[tool.uv.index]]
# url = "https://download.pytorch.org/whl/cpu"
#
# [tool.uv.sources]
# lammps-jax = { path = "../..", editable = true }
# ///
"""NequIP comm bundles with the OpenEquivariance fused conv kernel.

Imports the nequip_export conversion and swaps each layer's interpreted
tensor product for the OEQ kernel, folding the static e3nn weight
permutation into the edge MLP. Writes an OUTPUT.env runtime sidecar with
slot offsets for the contrib/ffi-replay shim; openequivariance_extjax
must be built from the OpenEquivariance repo since the pip wheel does
not ship it.
"""

import copy
import ctypes
import os
import sysconfig

os.environ.setdefault("OEQ_NOTORCH", "1")

import numpy as np

import jax
import jax.numpy as jnp

import nequip_export as base
from torchax_utils import to_jax_fn

OUTPUT = "nequip-oam-oeq.lammps-jax.json"
TARGETS = ("conv_forward", "conv_backward")
SHIM = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "..", "contrib", "ffi-replay",
                       "ffi_replay_shim.so"))

from openequivariance.core.e3nn_lite import TPProblem
from openequivariance.jax import TensorProductConv

# Swap: OEQ kernels replace the interpreted conv in every layer.
for name, layer in zip(base.conv_names, base.LAYERS):
    block = base.blocks[name].conv
    torch_tp = block.tp_scatter.tp
    problem = TPProblem(
        str(torch_tp.irreps_in1), str(torch_tp.irreps_in2),
        str(torch_tp.irreps_out),
        [(inst.i_in1, inst.i_in2, inst.i_out, inst.connection_mode,
          inst.has_weight) for inst in torch_tp.instructions],
        shared_weights=False, internal_weights=False,
        irrep_dtype=np.float32, weight_dtype=np.float32)
    # The e3nn weight layout probes into one static gather.
    with jax.ensure_compile_time_eval():
        kernel = TensorProductConv(problem, deterministic=False)
        probe = jnp.arange(problem.weight_numel, dtype=jnp.float32)[None]
        perm = np.asarray(
            kernel.reorder_weights_from_e3nn(probe))[0].astype(np.int64)
    assert np.array_equal(np.sort(perm), np.arange(problem.weight_numel)), \
        "OEQ weight reordering is not a pure permutation"
    # Fold the static permutation into the edge MLP's last linear so the
    # runtime carries no gather and its grad carries no scatter.
    block = copy.deepcopy(block)
    last = block.edge_mlp.mlp[-1]
    if last.weight.shape[0] == problem.weight_numel:
        last.weight.data = last.weight.data[perm]
    else:
        last.weight.data = last.weight.data[:, perm]
    layer["edge_mlp"] = to_jax_fn(block.edge_mlp)

    def conv(h, edge_attrs, weights, centers, neighbors, kernel=kernel):
        return kernel(h, edge_attrs, weights,
                      centers.astype(jnp.int32), neighbors.astype(jnp.int32))

    layer["conv"] = conv

if not base.SKIP_CHECK:
    base.verify(base.model)
base.export(OUTPUT, custom_call_targets=TARGETS)

# Sidecar: slot offsets resolve the hidden kernel symbols at run time.
import openequivariance_extjax as extjax

get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
get_pointer.restype = ctypes.c_void_p
get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
library = extjax.__file__
base_addr = None
with open("/proc/self/maps") as maps:
    for line in maps:
        if os.path.basename(library) in line:
            base_addr = int(line.split("-")[0], 16)
            break
capsules = extjax.registrations()
offsets = {t: get_pointer(capsules[t], None) - base_addr for t in TARGETS}
libpython = os.path.join(sysconfig.get_config_var("LIBDIR"),
                         sysconfig.get_config_var("INSTSONAME"))
handlers = ";".join(f"{t}={SHIM}:ffi_handler_{i}"
                    for i, t in enumerate(TARGETS))
size = os.path.getsize(library)
lines = [f"export LAMMPS_JAX_FFI_HANDLERS='{handlers}'",
         f"export FFI_PRELOAD_LIBS={libpython}"]
lines += [f"export FFI_HANDLER_{i}={library}+{offsets[t]:#x}@{size}"
          for i, t in enumerate(TARGETS)]
with open(OUTPUT + ".env", "w") as env_file:
    env_file.write("\n".join(lines) + "\n")
print("runtime env", OUTPUT + ".env")
