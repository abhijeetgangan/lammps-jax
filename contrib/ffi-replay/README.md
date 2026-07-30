# ffi-replay

Record-and-replay for FFI kernels whose custom calls bake process-local plan
handles at trace time. `ffi_replay_record.py` records the library's
plan-compile calls during export; `ffi_replay_shim.c` reissues them through
libffi on first invocation, verifies the handles, and forwards call frames to
the real execute handler. Export one bundle per process. Trampolines
ffi_handler_0..3 serve kernels that need no replay, resolved from
FFI_HANDLER_<n> as lib.so:symbol or lib.so+0xOFFSET@FILESIZE.

```bash
gcc -shared -fPIC -O2 -o ffi_replay_shim.so ffi_replay_shim.c -ldl -lpthread -l:libffi.so.8
```

cuEquivariance uniform_1d replays; `export_mace.py export cueq` records it.
Run with `<venv>/site-packages/cuequivariance_ops/lib` on LD_LIBRARY_PATH and:

```bash
LAMMPS_JAX_FFI_HANDLERS='uniform_1d_cuda=/path/ffi_replay_shim.so:ffi_replay_handler'
FFI_REPLAY_FILE=/path/bundle.ffi-replay
FFI_REPLAY_LIB=<venv>/site-packages/cuequivariance_ops_jax/lib/libcue_ops_jax.so
```

OpenEquivariance needs only symbol resolution; `export_mace.py export oeq`
writes the slot specs, plus the FFI_PRELOAD_LIBS list that satisfies Python
extension relocations, to a <bundle>.env sidecar to source at run time.

Notes: replay stays lazy since dlopen of CUDA libraries under the loader lock
deadlocks. The cue handler compiles kernels per OS thread, 40 ms per plan,
which is why cpp/pjrt/runtime.cpp keeps a persistent execution worker. Set
CUEQUIVARIANCE_OPS_NVRTC_CACHE_DIR on GPUs without shipped SASS. A teardown
SIGSEGV after MPI_Finalize is harmless.
