# LAMMPS-JAX

CUDA-first external LAMMPS/Kokkos plugin for running exported JAX models through
`pair_style jax/kk`.

## Setup

```bash
uv venv .venv --python python3
uv pip install -e '.[test]'
uv pip install -U 'jax[cuda12]'
```

## Build

```bash
cmake -S cpp -B build-plugin-gpu-pjrt \
  -D CMAKE_CXX_COMPILER="$KOKKOS_NVCC_WRAPPER" \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_CXX_FLAGS="-fno-lto" \
  -D CMAKE_SHARED_LINKER_FLAGS="-fno-lto" \
  -D CMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF \
  -D LAMMPS_HEADER_DIR="$LAMMPS_SRC/src" \
  -D JAXLIB_INCLUDE_DIR="$JAXLIB_INCLUDE_DIR" \
  -D PJRT_C_API_INCLUDE_DIR="$PJRT_C_API_INCLUDE_DIR" \
  -D KOKKOS_CONFIG_INCLUDE_DIR="$LAMMPS_BUILD/lib/kokkos"

cmake --build build-plugin-gpu-pjrt -j4
```

## Run

```bash
.venv/bin/python examples/export_lj.py examples/lj.lammps-jax.json \
  --max-atoms 1024 \
  --edges-per-atom 96

LAMMPS_PLUGIN_PATH=build-plugin-gpu-pjrt \
  "$LAMMPS_INSTALL/bin/lmp" -k on g 1 -pk kokkos newton off neigh full -sf kk \
  -var pjrt /absolute/path/xla_cuda_plugin.so \
  -in examples/in.lj_jax
```

## Test

```bash
.venv/bin/python -m pytest
```
