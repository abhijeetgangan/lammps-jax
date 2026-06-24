#include "lammpsplugin.h"
#include "version.h"

#include "pair_jax_kokkos.h"

using namespace LAMMPS_NS;

static Pair *jax_kokkos_creator(LAMMPS *lmp)
{
  return new PairJaxKokkos(lmp);
}

extern "C" void lammpsplugin_init(void *lmp, void *handle, void *regfunc)
{
  lammpsplugin_t plugin;
  auto register_plugin = (lammpsplugin_regfunc) regfunc;

  plugin.version = LAMMPS_VERSION;
  plugin.style = "pair";
  plugin.name = "jax/kk";
  plugin.info = "CUDA PJRT-backed JAX pair style for Kokkos";
  plugin.author = "LAMMPS-JAX";
  plugin.creator.v1 = (lammpsplugin_factory1 *) &jax_kokkos_creator;
  plugin.handle = handle;
  (*register_plugin)(&plugin, lmp);
}
