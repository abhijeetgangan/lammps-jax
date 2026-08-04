// pair_style jax/kk: runs a jax.export'ed bundle through a PJRT plugin on
// the LAMMPS Kokkos CUDA stream; array data never passes through the host.

#ifndef LMP_PAIR_JAX_KOKKOS_H
#define LMP_PAIR_JAX_KOKKOS_H

#include "lammps_jax_model.h"

#include "pjrt/runtime.h"

#include "kokkos_base.h"
#include "pair.h"

#include <type_traits>

#include "kokkos_type.h"

#if defined(KOKKOS_ENABLE_CUDA) && !defined(LMP_KOKKOS_DOUBLE_DOUBLE) && \
    !defined(LMP_KOKKOS_SINGLE_SINGLE) && !defined(LMP_KOKKOS_SINGLE_DOUBLE)
#error "pair jax/kk needs the KOKKOS precision layer: LAMMPS 10 Sep 2025 or newer"
#endif

#include <cuda.h>

#include <memory>
#include <string>

namespace LAMMPS_NS {

class AtomKokkos;
template <class DeviceType> class NeighListKokkos;

class PairJaxKokkos : public Pair, public KokkosBase {
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

#ifdef KOKKOS_ENABLE_CUDA
  // Device variants exchange rows in place through Kokkos comm buffers.
  int pack_forward_comm_kokkos(int, DAT::tdual_int_1d, DAT::tdual_double_1d &, int,
                               int *) override;
  void unpack_forward_comm_kokkos(int, int, DAT::tdual_double_1d &) override;
  int pack_reverse_comm_kokkos(int, int, DAT::tdual_double_1d &) override;
  void unpack_reverse_comm_kokkos(int, DAT::tdual_int_1d, DAT::tdual_double_1d &) override;
#endif

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
  float *d_comm_rows = nullptr;
  int comm_width = 0;

#ifdef KOKKOS_ENABLE_CUDA
  using device_type = LMPDeviceType;
  using x_view = typename ArrayTypes<device_type>::t_kkfloat_1d_3_lr_randomread;
  using type_view = typename ArrayTypes<device_type>::t_int_1d_randomread;
  using f_view = typename ArrayTypes<device_type>::t_kkacc_1d_3;
  template <typename Scalar>
  using positions_view = Kokkos::View<Scalar *[3], Kokkos::LayoutRight, device_type>;
  template <typename Scalar>
  using box_view = Kokkos::View<Scalar[3][3], Kokkos::LayoutRight, device_type>;
  template <typename Scalar>
  using host_box_view = Kokkos::View<Scalar[3][3], Kokkos::LayoutRight, Kokkos::HostSpace>;
  // Positions and box stage at contract precision; only the matching f32 or f64 view is allocated.
  template <typename Scalar>
  struct Staging {
    positions_view<Scalar> positions;
    box_view<Scalar> box;
    host_box_view<Scalar> host_box;
  };
  using int_view = Kokkos::View<int *, device_type>;
  using bool_view = Kokkos::View<bool *, device_type>;
  using scalar_int_view = Kokkos::View<int, device_type>;

  void pack_atoms(int nall, const x_view &x, const type_view &type);
  template <typename Scalar>
  void pack_atoms_as(int span, const x_view &x, const type_view &type);
  void pack_edges(NeighListKokkos<device_type> *klist, bool rebuild_edges,
                  const x_view &x);
  void pack_box();
  template <typename Scalar>
  void pack_box_as();
  pjrt::ExecutionRequest make_request(CUstream stream, CUevent ready_event);
  template <typename Scalar>
  pjrt::ExecutionRequest make_request_as(CUstream stream, CUevent ready_event);
  template <typename Scalar>
  void allocate_staging_as(int max_atoms);
  void add_model_forces(CUdeviceptr force_output, const f_view &f, int nlocal, int nall);
  template <typename Scalar>
  void add_model_forces_as(CUdeviceptr force_output, const f_view &f, int nlocal, int nall);

  device_type exec;
  Staging<float> staging32;
  Staging<double> staging64;
  template <typename Scalar>
  Staging<Scalar> &staging()
  {
    if constexpr (std::is_same_v<Scalar, double>)
      return staging64;
    else
      return staging32;
  }
  // Calls f with a float or double value per the contract precision; f deduces Scalar from it.
  template <typename F>
  auto dispatch_by_precision(F &&f)
  {
    if (f64_enabled()) return f(double{});
    return f(float{});
  }
  int_view d_species;
  scalar_int_view d_nlocal;
  scalar_int_view d_nghost;
  scalar_int_view d_edge_count;
  scalar_int_view d_edge_overflow;
  int_view d_senders;
  int_view d_receivers;
  bool_view d_edge_mask;
#endif
};

} // namespace LAMMPS_NS

#endif
