# LAMMPS-JAX

[![DOI](https://zenodo.org/badge/1278620240.svg)](https://doi.org/10.5281/zenodo.20838693)

[LAMMPS](https://www.lammps.org) plugin for running JAX forcefields with
`jax.export` and a PJRT runtime.

## Setup

```bash
uv venv .venv --python python3
source .venv/bin/activate
uv pip install -e '.[test]'
uv pip install -U 'jax[cuda12]'
```

## Build

The pair style needs the KOKKOS precision layer from LAMMPS 10 Sep 2025 or
newer; stable releases do not carry it yet, and older trees stop at a
compile-time error. Build the plugin against the same LAMMPS source tree as
the `lmp` binary that loads it, and rebuild both together after pulling.

```bash
cmake -S cpp -B build-plugin-gpu-pjrt \
  -D CMAKE_CXX_COMPILER="$KOKKOS_NVCC_WRAPPER" \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_CXX_FLAGS="-fno-lto" \
  -D CMAKE_SHARED_LINKER_FLAGS="-fno-lto" \
  -D CMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF \
  -D LAMMPS_HEADER_DIR="$LAMMPS_SRC/src" \
  -D JAXLIB_INCLUDE_DIR="$JAXLIB_INCLUDE_DIR" \
  -D KOKKOS_CONFIG_INCLUDE_DIR="$LAMMPS_BUILD/lib/kokkos"
cmake --build build-plugin-gpu-pjrt -j4
```

## Run

```bash
python examples/export_model.py lj examples/lj.lammps-jax.json \
  --max-atoms 1024 --edges-per-atom 96
```

The bundle loads as a pair style:

```
pair_style jax/kk ${pjrt}
pair_coeff * * examples/lj.lammps-jax.json
```

```bash
LAMMPS_PLUGIN_PATH=build-plugin-gpu-pjrt \
  "$LAMMPS_INSTALL/bin/lmp" -k on g 1 -pk kokkos newton off neigh full -sf kk \
  -var pjrt /absolute/path/xla_cuda_plugin.so -in examples/in.lj_jax
```

## Test

```bash
JAX_PLATFORMS=cpu python -m pytest
```

Integration tests, which require a GPU and LAMMPS:

```bash
LAMMPS_BIN=$LAMMPS_INSTALL/bin/lmp \
PJRT_PLUGIN=/absolute/path/xla_cuda_plugin.so \
LAMMPS_PLUGIN_PATH=$PWD/build-plugin-gpu-pjrt \
  python -m pytest tests/test_lammps.py -v
```

## Design notes

Models are exported with `jax.export` into a JSON bundle holding the
serialized program and the settings the pair style enforces at load: atom and
edge capacities, cutoff, precision, unit style, and the distribution scheme.
Exported programs have static shapes, so positions and the edge list are
padded to capacity and an edge mask marks the live entries.

The pair style executes the bundle through the PJRT C API, loading the same
plugin library jax uses on the GPU, so the build never needs XLA. Kokkos kernels
pack the LAMMPS neighbor list into sender, receiver, and mask arrays on
device, the program runs on the LAMMPS CUDA stream, and forces come back as
device buffers, so array data never passes through the host.

Multi-rank runs pick one of two schemes at export. A ghost export widens the
ghost shell to n_hops cutoffs and stays communication-free inside the
program; a comm export keeps the one-cutoff shell and exchanges per-layer
features through LAMMPS forward and reverse communication, called from inside
the program as an FFI callback. Exchanges are linear JAX primitives whose
transposes swap forward and reverse, so models may call either direction and
every differentiation mode traces through them. Communicating half-edge
exports pack each pair on one rank, matching the native half neighbor list,
and reverse-communicate ghost density partials into their owners. Exchanged
rows move in place on device under the Kokkos brick and tiled comm styles and
stage through pinned host memory otherwise.

cuEquivariance and OpenEquivariance kernels stay in the exported program as
custom call targets, resolved at run time from LAMMPS_JAX_FFI_HANDLERS;
contrib/ffi-replay handles libraries whose compiled kernels only exist inside
the exporting process.

## Citation

If you use `lammps-jax` or any algorithms implemented here, please cite
the archived Zenodo release: https://doi.org/10.5281/zenodo.20838694

## Acknowledgements

Thanks to @mitkotak for API discussions and @wcwitt for guidance on
distributed inference. See
[lammps/lammps#4691](https://github.com/lammps/lammps/pull/4691) for the
original discussion on running JAX models in LAMMPS, and
[openmm-jax](https://github.com/atomicarchitects/openmm-jax) for the OpenMM
counterpart.

## License

MIT. See `LICENSE`.
