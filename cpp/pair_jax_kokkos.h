#ifndef LMP_PAIR_JAX_KOKKOS_H
#define LMP_PAIR_JAX_KOKKOS_H

#include "lammps_jax_model.h"

#include "pjrt/runtime.h"

#include "pair.h"

#include "kokkos_type.h"

#include <cuda.h>

#include <memory>
#include <string>

namespace LAMMPS_NS {

class AtomKokkos;
template <class DeviceType> class NeighListKokkos;

class PairJaxKokkos : public Pair {
 public:
  PairJaxKokkos(class LAMMPS *);
  ~PairJaxKokkos() override;

  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;

 protected:
  void allocate();
  void allocate_device_buffers();
  bool edge_force_enabled() const;

  AtomKokkos *atomKK = nullptr;
  // Empty means: resolve via LAMMPS_JAX_PJRT_PLUGIN_PATH, then the soname.
  std::string pjrt_plugin_path;
  lammps_jax::ModelBundle bundle;
  std::unique_ptr<pjrt::Runtime> runtime;
  double scale = 1.0;
  bool model_loaded = false;
  bool edge_cache_valid = false;
  int cached_edge_count = 0;

#ifdef KOKKOS_ENABLE_CUDA
  using device_type = LMPDeviceType;
  using x_view = typename ArrayTypes<device_type>::t_kkfloat_1d_3_lr_randomread;
  using type_view = typename ArrayTypes<device_type>::t_int_1d_randomread;
  using f_view = typename ArrayTypes<device_type>::t_kkacc_1d_3;
  using float_positions_view =
      Kokkos::View<float *[3], Kokkos::LayoutRight, device_type>;
  using int_view = Kokkos::View<int *, device_type>;
  using bool_view = Kokkos::View<bool *, device_type>;
  using scalar_int_view = Kokkos::View<int, device_type>;
  using box_view = Kokkos::View<float[3][3], Kokkos::LayoutRight, device_type>;
  using host_box_view = Kokkos::View<float[3][3], Kokkos::LayoutRight, Kokkos::HostSpace>;

  void pack_atoms(int nall, const x_view &x, const type_view &type);
  void pack_edges(NeighListKokkos<device_type> *klist, const x_view &x, bool rebuild_edges);
  void pack_box();
  pjrt::ExecutionRequest make_request(CUstream stream, CUevent ready_event);
  void add_model_forces(CUdeviceptr force_output, const f_view &f, int nlocal, int nall);

  device_type exec;
  float_positions_view d_positions;
  int_view d_species;
  scalar_int_view d_nlocal;
  scalar_int_view d_nghost;
  scalar_int_view d_edge_count;
  scalar_int_view d_edge_overflow;
  int_view d_senders;
  int_view d_receivers;
  bool_view d_edge_mask;
  box_view d_box;
  host_box_view h_box;
#endif
};

} // namespace LAMMPS_NS

#endif
