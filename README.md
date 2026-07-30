# LAMMPS-JAX

[![DOI](https://zenodo.org/badge/1278620240.svg)](https://doi.org/10.5281/zenodo.20838693)

LAMMPS plugin for running JAX forcefields with `jax.export` and a PJRT runtime

## Setup

```bash
uv venv .venv --python python3
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
.venv/bin/python examples/export_model.py lj examples/lj.lammps-jax.json \
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
JAX_PLATFORMS=cpu .venv/bin/python -m pytest
```

Integration tests, which require a GPU and LAMMPS:

```bash
LAMMPS_BIN=$LAMMPS_INSTALL/bin/lmp \
PJRT_PLUGIN=/absolute/path/xla_cuda_plugin.so \
LAMMPS_PLUGIN_PATH=$PWD/build-plugin-gpu-pjrt \
  .venv/bin/python -m pytest tests/test_lammps.py -v
```

## Citation

If you use `lammps-jax` or any algorithms implemented here, please cite
the archived Zenodo release: https://doi.org/10.5281/zenodo.20838694

## Acknowledgements

Thanks to @mitkotak for API discussions and @wcwitt for guidance on
distributed inference.

## License

MIT. See `LICENSE`.
