// pair_style jax/kk: runs a jax.export'ed bundle through a PJRT plugin on
// the LAMMPS Kokkos CUDA stream; array data never passes through the host.

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

  // Per-layer model comm: pack/unpack move feature rows between pinned staging and comm buffers.
  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;
  int pack_reverse_comm(int, int, double *) override;
  void unpack_reverse_comm(int, int *, double *) override;

 protected:
  void allocate();
  void allocate_device_buffers();
  bool edge_force_enabled() const;
  bool comm_enabled() const;
  bool f64_enabled() const;
  void service_model_comm(const pjrt::ModelCommRequest &request);

  AtomKokkos *atomKK = nullptr;
  // Empty means: resolve via LAMMPS_JAX_PJRT_PLUGIN_PATH, then the soname.
  std::string pjrt_plugin_path;
  lammps_jax::ModelBundle bundle;
  std::unique_ptr<pjrt::Runtime> runtime;
  // Optional fourth pair_coeff argument; multiplies model energy and forces.
  double scale = 1.0;
  bool model_loaded = false;
  // The packed edge graph persists between reneighbor steps; cached count is the launch extent.
  bool edge_cache_valid = false;
  int cached_edge_count = 0;
  // Active model-comm site state, valid only inside service_model_comm.
  float *comm_rows = nullptr;
  int comm_width = 0;

#ifdef KOKKOS_ENABLE_CUDA
  using device_type = LMPDeviceType;
  using x_view = typename ArrayTypes<device_type>::t_kkfloat_1d_3_lr_randomread;
  using type_view = typename ArrayTypes<device_type>::t_int_1d_randomread;
  using f_view = typename ArrayTypes<device_type>::t_kkacc_1d_3;
  // Positions and box stage at contract precision; only the matching f32 or f64 view is allocated.
  template <typename Scalar>
  using positions_view = Kokkos::View<Scalar *[3], Kokkos::LayoutRight, device_type>;
  using int_view = Kokkos::View<int *, device_type>;
  using bool_view = Kokkos::View<bool *, device_type>;
  using scalar_int_view = Kokkos::View<int, device_type>;
  template <typename Scalar>
  using box_view = Kokkos::View<Scalar[3][3], Kokkos::LayoutRight, device_type>;
  template <typename Scalar>
  using host_box_view = Kokkos::View<Scalar[3][3], Kokkos::LayoutRight, Kokkos::HostSpace>;

  void pack_atoms(int nall, const x_view &x, const type_view &type);
  void pack_edges(NeighListKokkos<device_type> *klist, bool rebuild_edges);
  void pack_box();
  pjrt::ExecutionRequest make_request(CUstream stream, CUevent ready_event);
  void add_model_forces(CUdeviceptr force_output, const f_view &f, int nlocal, int nall);
  template <typename Scalar>
  void add_model_forces_as(CUdeviceptr force_output, const f_view &f, int nlocal, int nall);

  device_type exec;
  positions_view<float> d_positions;
  positions_view<double> d_positions_f64;
  int_view d_species;
  scalar_int_view d_nlocal;
  scalar_int_view d_nghost;
  scalar_int_view d_edge_count;
  scalar_int_view d_edge_overflow;
  int_view d_senders;
  int_view d_receivers;
  bool_view d_edge_mask;
  box_view<float> d_box;
  host_box_view<float> h_box;
  box_view<double> d_box_f64;
  host_box_view<double> h_box_f64;
#endif
};

} // namespace LAMMPS_NS

#endif
