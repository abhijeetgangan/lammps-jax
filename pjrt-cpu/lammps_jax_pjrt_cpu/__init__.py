"""XLA CPU PJRT plugin for pair_style jax/kk on host Kokkos builds of LAMMPS."""

from pathlib import Path

# The pair_style jax/kk argument.
PLUGIN_PATH = Path(__file__).resolve().parent / "pjrt_c_api_cpu_plugin.so"
