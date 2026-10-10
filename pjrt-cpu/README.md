# lammps-jax-pjrt-cpu

XLA's CPU PJRT plugin for [lammps-jax](https://github.com/abhijeetgangan/lammps-jax)
on host Kokkos builds of LAMMPS, built from the XLA of the jaxlib with the same
version; it runs bundles from that jax or older.

```bash
pip install lammps-jax-pjrt-cpu
python -c 'import lammps_jax_pjrt_cpu as p; print(p.PLUGIN_PATH)'
```
