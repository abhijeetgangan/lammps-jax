#include "pair_jax_kokkos.h"

#include "atom.h"
#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "fix.h"
#include "force.h"
#include "kokkos.h"
#include "memory.h"
#include "modify.h"
#include "neigh_list.h"
#include "neigh_list_kokkos.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "update.h"
#include "utils.h"

#include <cuda.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

using namespace LAMMPS_NS;

namespace {

#ifdef KOKKOS_ENABLE_CUDA
template <class DeviceType>
struct PackAtomsFunctor {
  using AT = ArrayTypes<DeviceType>;
  typename AT::t_kkfloat_1d_3_lr_randomread x;
  typename AT::t_int_1d_randomread type;
  Kokkos::View<float *[3], Kokkos::LayoutRight, DeviceType> positions;
  Kokkos::View<int *, DeviceType> species;

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const
  {
    positions(i, 0) = static_cast<float>(x(i, 0));
    positions(i, 1) = static_cast<float>(x(i, 1));
    positions(i, 2) = static_cast<float>(x(i, 2));
    species(i) = type(i) - 1;
  }
};

template <class DeviceType>
struct PackNeighborFunctor {
  NeighListKokkos<DeviceType> list;
  Kokkos::View<int *, DeviceType> senders_out;
  Kokkos::View<int *, DeviceType> receivers_out;
  Kokkos::View<bool *, DeviceType> edge_mask_out;
  Kokkos::View<int, DeviceType> edge_count;
  Kokkos::View<int, DeviceType> edge_overflow;
  int max_atoms;
  int max_edges;
  int max_neighbors_per_atom;
  int inum;
  bool duplicate_reverse_edges;

  ~PackNeighborFunctor() { list.copymode = 1; }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int flat) const
  {
    const int ii = flat / max_neighbors_per_atom;
    const int jj = flat - ii * max_neighbors_per_atom;
    if (ii >= inum) return;
    const int i = list.d_ilist(ii);
    if (i >= max_atoms || jj >= list.d_numneigh(i)) return;
    const int j = list.d_neighbors(i, jj) & NEIGHMASK;
    if (j >= max_atoms) return;

    const int edges_to_add = duplicate_reverse_edges ? 2 : 1;
    const int edge = Kokkos::atomic_fetch_add(&edge_count(), edges_to_add);
    if (edge + edges_to_add > max_edges) {
      edge_overflow() = 1;
      return;
    }
    senders_out(edge) = i;
    receivers_out(edge) = j;
    edge_mask_out(edge) = true;
    if (duplicate_reverse_edges) {
      senders_out(edge + 1) = j;
      receivers_out(edge + 1) = i;
      edge_mask_out(edge + 1) = true;
    }
  }
};

template <class DeviceType>
struct AddForcesFunctor {
  using AT = ArrayTypes<DeviceType>;
  typename AT::t_kkacc_1d_3 f;
  Kokkos::View<const float *[3], Kokkos::LayoutRight, DeviceType,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      model_forces;
  double scale;

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const
  {
    f(i, 0) += static_cast<KK_ACC_FLOAT>(scale * model_forces(i, 0));
    f(i, 1) += static_cast<KK_ACC_FLOAT>(scale * model_forces(i, 1));
    f(i, 2) += static_cast<KK_ACC_FLOAT>(scale * model_forces(i, 2));
  }
};

// Launched over [0, cached_edge_count): the packed edge rows are contiguous,
// so no mask test is needed here (only the padding rows past the count are
// invalid, and they are never in range).
template <class DeviceType>
struct AddEdgeForcesFunctor {
  using AT = ArrayTypes<DeviceType>;
  typename AT::t_kkacc_1d_3 f;
  Kokkos::View<const float *[3], Kokkos::LayoutRight, DeviceType,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      edge_forces;
  Kokkos::View<int *, DeviceType> senders;
  Kokkos::View<int *, DeviceType> receivers;
  bool newton_pair;
  double scale;

  KOKKOS_INLINE_FUNCTION
  void operator()(const int edge) const
  {
    // Senders are always owned atoms: edges come from rows 0..inum-1 of a
    // local-atom neighbor list in both newton modes.
    const int i = senders(edge);
    const KK_ACC_FLOAT fx = static_cast<KK_ACC_FLOAT>(scale * edge_forces(edge, 0));
    const KK_ACC_FLOAT fy = static_cast<KK_ACC_FLOAT>(scale * edge_forces(edge, 1));
    const KK_ACC_FLOAT fz = static_cast<KK_ACC_FLOAT>(scale * edge_forces(edge, 2));
    Kokkos::atomic_add(&f(i, 0), fx);
    Kokkos::atomic_add(&f(i, 1), fy);
    Kokkos::atomic_add(&f(i, 2), fz);
    if (newton_pair) {
      const int j = receivers(edge);
      Kokkos::atomic_add(&f(j, 0), -fx);
      Kokkos::atomic_add(&f(j, 1), -fy);
      Kokkos::atomic_add(&f(j, 2), -fz);
    }
  }
};

template <class DeviceType>
struct VirialFDotRFunctor {
  using AT = ArrayTypes<DeviceType>;
  using value_type = EV_FLOAT;
  typename AT::t_kkfloat_1d_3_lr_randomread x;
  typename AT::t_kkacc_1d_3 f;

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, value_type &virial) const
  {
    virial.v[0] += f(i, 0) * static_cast<KK_ACC_FLOAT>(x(i, 0));
    virial.v[1] += f(i, 1) * static_cast<KK_ACC_FLOAT>(x(i, 1));
    virial.v[2] += f(i, 2) * static_cast<KK_ACC_FLOAT>(x(i, 2));
    virial.v[3] += f(i, 1) * static_cast<KK_ACC_FLOAT>(x(i, 0));
    virial.v[4] += f(i, 2) * static_cast<KK_ACC_FLOAT>(x(i, 0));
    virial.v[5] += f(i, 2) * static_cast<KK_ACC_FLOAT>(x(i, 1));
  }
};

CUevent record_ready_event(CUstream stream)
{
  CUevent event = nullptr;
  if (cuEventCreate(&event, CU_EVENT_DISABLE_TIMING) != CUDA_SUCCESS)
    throw std::runtime_error("Failed to create CUDA event for LAMMPS-JAX inputs");
  if (cuEventRecord(event, stream) != CUDA_SUCCESS) {
    cuEventDestroy(event);
    throw std::runtime_error("Failed to record CUDA event for LAMMPS-JAX inputs");
  }
  return event;
}

#endif

} // namespace

PairJaxKokkos::PairJaxKokkos(LAMMPS *lmp) : Pair(lmp)
{
#ifdef KOKKOS_ENABLE_CUDA
  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<LMPDeviceType>::space;
  datamask_read = X_MASK | F_MASK | TYPE_MASK;
  datamask_modify = F_MASK | ENERGY_MASK | VIRIAL_MASK;
  manybody_flag = 1;
  one_coeff = 1;
  restartinfo = 0;
  single_enable = 0;
#else
  error->all(FLERR, "pair_style jax/kk requires LAMMPS KOKKOS built with CUDA");
#endif
}

PairJaxKokkos::~PairJaxKokkos()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }
}

void PairJaxKokkos::allocate()
{
  allocated = 1;
  const int n = atom->ntypes;
  memory->create(setflag, n + 1, n + 1, "pair_jax:setflag");
  memory->create(cutsq, n + 1, n + 1, "pair_jax:cutsq");
  for (int i = 1; i <= n; ++i)
    for (int j = 1; j <= n; ++j) setflag[i][j] = 0;
}

void PairJaxKokkos::allocate_device_buffers()
{
#ifdef KOKKOS_ENABLE_CUDA
  const int max_atoms = bundle.contract.max_atoms;
  const int max_edges = bundle.contract.max_edges;
  // Kokkos zero-initializes these views, which is the padding contract:
  // pack_atoms only writes the live rows.
  d_positions = float_positions_view("lammps_jax_positions", max_atoms);
  d_species = int_view("lammps_jax_species", max_atoms);
  d_nlocal = scalar_int_view("lammps_jax_nlocal");
  d_nghost = scalar_int_view("lammps_jax_nghost");
  d_edge_count = scalar_int_view("lammps_jax_edge_count");
  d_edge_overflow = scalar_int_view("lammps_jax_edge_overflow");
  d_senders = int_view("lammps_jax_senders", max_edges);
  d_receivers = int_view("lammps_jax_receivers", max_edges);
  d_edge_mask = bool_view("lammps_jax_edge_mask", max_edges);
  if (bundle.contract.uses_box) {
    d_box = box_view("lammps_jax_box");
    h_box = host_box_view("lammps_jax_host_box");
  }
#endif
}

bool PairJaxKokkos::edge_force_enabled() const
{
  return bundle.contract.force_layout == lammps_jax::ForceLayout::Edge;
}

void PairJaxKokkos::settings(int narg, char **arg)
{
  // pair_style jax/kk [plugin-path]
  if (narg > 1) error->all(FLERR, "Illegal pair_style jax/kk command");
  if (narg == 1) {
    pjrt_plugin_path = arg[0];
  }
}

void PairJaxKokkos::coeff(int narg, char **arg)
{
  if (narg < 3 || narg > 4) error->all(FLERR, "Illegal pair_coeff jax/kk command");
  if (!allocated) allocate();

  int ilo, ihi, jlo, jhi;
  utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
  utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);
  if (narg == 4) scale = utils::numeric(FLERR, arg[3], false, lmp);

  try {
    bundle = lammps_jax::load_bundle_file(arg[2]);
    if (bundle.contract.unit_style != update->unit_style)
      error->all(FLERR, "LAMMPS-JAX model unit style does not match current units");
    pjrt::ClientOptions client_options;
#ifdef KOKKOS_ENABLE_CUDA
    client_options.visible_device = exec.cuda_device();
#endif
    if (const char *fraction = std::getenv("LAMMPS_JAX_MEM_FRACTION"))
      client_options.memory_fraction = std::atof(fraction);
    runtime = std::make_unique<pjrt::Runtime>();
    runtime->initialize(
        pjrt::resolve_plugin_path(pjrt_plugin_path, "LAMMPS_JAX_PJRT_PLUGIN_PATH"),
        {bundle.programs.force_mlir, bundle.programs.energy_mlir,
         bundle.programs.energy_and_forces_mlir},
        bundle.compile_options, client_options);
    allocate_device_buffers();
    model_loaded = true;
  } catch (const std::exception &e) {
    error->all(FLERR, "Failed to load LAMMPS-JAX model: {}", e.what());
  }

  int count = 0;
  for (int i = ilo; i <= ihi; ++i) {
    for (int j = MAX(jlo, i); j <= jhi; ++j) {
      setflag[i][j] = 1;
      ++count;
    }
  }
  if (count == 0) error->all(FLERR, "Incorrect args for pair coefficients");
}

void PairJaxKokkos::init_style()
{
  if (!model_loaded) error->all(FLERR, "LAMMPS-JAX model has not been loaded");
  if (bundle.contract.newton == lammps_jax::NewtonMode::On && !force->newton_pair)
    error->all(FLERR,
               "LAMMPS-JAX bundle was exported for newton pair on; this run uses newton off");
  if (bundle.contract.newton == lammps_jax::NewtonMode::Off && force->newton_pair)
    error->all(FLERR,
               "LAMMPS-JAX bundle was exported for newton pair off; this run uses newton on");
  if (atom->molecular != Atom::ATOMIC) {
    // The packed graph applies full weight to every listed pair. Fully
    // excluded pairs (lj 0, coul 0) never enter the neighbor list; any other
    // special_bonds factor would be silently ignored.
    for (int n = 1; n <= 3; ++n) {
      if (force->special_lj[n] != 1.0 &&
          !(force->special_lj[n] == 0.0 && force->special_coul[n] == 0.0))
        error->all(FLERR,
                   "pair jax/kk does not apply special_bonds factors; use 1 1 1 or full exclusion");
    }
  }
  if (!force->newton_pair) {
    // Pair virial is available only via device f-dot-r, which needs the
    // ghost force rows of the newton-on path. Barostats would silently
    // regulate against a pressure missing the pair contribution.
    for (int i = 0; i < modify->nfix; ++i) {
      const char *style = modify->fix[i]->style;
      if (strstr(style, "npt") || strstr(style, "nph") || strstr(style, "press"))
        error->all(FLERR,
                   "Fix {} requires pair virial; run pair jax/kk with newton pair on",
                   style);
    }
  }
  if (comm->me == 0) {
    if (force->newton_pair)
      utils::logmesg(lmp,
                     "LAMMPS-JAX: using newton pair on with a Kokkos half neighbor list; "
                     "ghost force rows are accumulated for LAMMPS reverse communication.\n");
    if (bundle.programs.energy_and_forces_mlir.empty())
      utils::logmesg(lmp,
                     "LAMMPS-JAX: force-only bundle; pair energy is reported as zero.\n");
    if (!force->newton_pair)
      utils::logmesg(lmp,
                     "LAMMPS-JAX: pair virial is computed only with newton pair on; "
                     "pressure output excludes pair contributions in this run.\n");
  }
  // Newton on: default half list (owned-atom rows). Newton off: full list;
  // ghost rows are not consumed, so no REQ_GHOST.
  auto request = force->newton_pair ? neighbor->add_request(this)
                                    : neighbor->add_request(this, NeighConst::REQ_FULL);
  request->set_kokkos_device(true);
  request->set_kokkos_host(false);
}

double PairJaxKokkos::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR, "All pair coeffs are not set");
  setflag[j][i] = setflag[i][j];
  return bundle.contract.cutoff;
}

#ifdef KOKKOS_ENABLE_CUDA
void PairJaxKokkos::pack_atoms(int nall, const PairJaxKokkos::x_view &x,
                               const PairJaxKokkos::type_view &type)
{
  // Padding rows keep their allocation-time zeros; rows past a shrinking
  // nall hold stale data that only masked-out edges can reference. The span
  // is clamped so the capacity-exceeded step (which errors out right after
  // packing) cannot write out of bounds.
  const int span = std::min(nall, bundle.contract.max_atoms);
  Kokkos::parallel_for("LAMMPSJAX::pack_atoms", Kokkos::RangePolicy<LMPDeviceType>(exec, 0, span),
                       PackAtomsFunctor<LMPDeviceType>{x, type, d_positions, d_species});
}

void PairJaxKokkos::pack_edges(NeighListKokkos<PairJaxKokkos::device_type> *klist,
                               const PairJaxKokkos::x_view &x, bool rebuild_edges)
{
  if (rebuild_edges) {
    Kokkos::deep_copy(exec, d_edge_count, 0);
    Kokkos::deep_copy(exec, d_edge_overflow, 0);
    Kokkos::deep_copy(exec, d_edge_mask, false);
    Kokkos::deep_copy(exec, d_senders, bundle.contract.max_atoms);
    Kokkos::deep_copy(exec, d_receivers, bundle.contract.max_atoms);

    const int max_neighbors_per_atom = std::max(1, klist->maxneighs);
    Kokkos::parallel_for(
        "LAMMPSJAX::pack_neighbors",
        Kokkos::RangePolicy<LMPDeviceType>(exec, 0, klist->inum * max_neighbors_per_atom),
        PackNeighborFunctor<LMPDeviceType>{*klist, d_senders, d_receivers, d_edge_mask,
                                           d_edge_count, d_edge_overflow,
                                           bundle.contract.max_atoms, bundle.contract.max_edges,
                                           max_neighbors_per_atom, klist->inum,
                                           static_cast<bool>(force->newton_pair && !edge_force_enabled())});
  }
}

void PairJaxKokkos::pack_box()
{
  // Row-vector cell matrix [[xprd,0,0],[xy,yprd,0],[xz,yz,zprd]], repacked
  // every step so box-changing runs (NPT) stay consistent.
  h_box(0, 0) = static_cast<float>(domain->xprd);
  h_box(0, 1) = 0.0f;
  h_box(0, 2) = 0.0f;
  h_box(1, 0) = static_cast<float>(domain->xy);
  h_box(1, 1) = static_cast<float>(domain->yprd);
  h_box(1, 2) = 0.0f;
  h_box(2, 0) = static_cast<float>(domain->xz);
  h_box(2, 1) = static_cast<float>(domain->yz);
  h_box(2, 2) = static_cast<float>(domain->zprd);
  Kokkos::deep_copy(exec, d_box, h_box);
}

pjrt::ExecutionRequest PairJaxKokkos::make_request(CUstream stream, CUevent ready_event)
{
  pjrt::ExecutionRequest request;
  request.stream = stream;
  request.input_ready_event = ready_event;
  request.inputs = {
      {"positions", reinterpret_cast<CUdeviceptr>(d_positions.data()),
       {bundle.contract.max_atoms, 3}, pjrt::ElementType::F32},
      {"species", reinterpret_cast<CUdeviceptr>(d_species.data()), {bundle.contract.max_atoms},
       pjrt::ElementType::S32},
      {"nlocal", reinterpret_cast<CUdeviceptr>(d_nlocal.data()), {}, pjrt::ElementType::S32},
      {"nghost", reinterpret_cast<CUdeviceptr>(d_nghost.data()), {}, pjrt::ElementType::S32},
      {"senders", reinterpret_cast<CUdeviceptr>(d_senders.data()), {bundle.contract.max_edges},
       pjrt::ElementType::S32},
      {"receivers", reinterpret_cast<CUdeviceptr>(d_receivers.data()), {bundle.contract.max_edges},
       pjrt::ElementType::S32},
      {"edge_mask", reinterpret_cast<CUdeviceptr>(d_edge_mask.data()), {bundle.contract.max_edges},
       pjrt::ElementType::Pred},
  };
  if (bundle.contract.uses_box) {
    request.inputs.emplace_back("box", reinterpret_cast<CUdeviceptr>(d_box.data()),
                                std::vector<int64_t>{3, 3}, pjrt::ElementType::F32);
  }
  return request;
}

void PairJaxKokkos::add_model_forces(CUdeviceptr force_output, const PairJaxKokkos::f_view &f,
                                     int nlocal, int nall)
{
  using ForceView = Kokkos::View<const float *[3], Kokkos::LayoutRight, LMPDeviceType,
                                Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  if (edge_force_enabled()) {
    ForceView edge_forces(reinterpret_cast<const float *>(force_output), bundle.contract.max_edges);
    Kokkos::parallel_for(
        "LAMMPSJAX::add_edge_forces",
        Kokkos::RangePolicy<LMPDeviceType>(exec, 0, cached_edge_count),
        AddEdgeForcesFunctor<LMPDeviceType>{f, edge_forces, d_senders, d_receivers,
                                            static_cast<bool>(force->newton_pair), scale});
  } else {
    ForceView model_forces(reinterpret_cast<const float *>(force_output), bundle.contract.max_atoms);
    const int limit = force->newton_pair ? nall : nlocal;
    Kokkos::parallel_for("LAMMPSJAX::add_forces", Kokkos::RangePolicy<LMPDeviceType>(exec, 0, limit),
                         AddForcesFunctor<LMPDeviceType>{f, model_forces, scale});
  }
}
#endif

void PairJaxKokkos::compute(int eflag, int vflag)
{
#ifndef KOKKOS_ENABLE_CUDA
  error->all(FLERR, "pair_style jax/kk requires CUDA");
#else
  ev_init(eflag, vflag, 0);
  if (!model_loaded || runtime == nullptr) error->all(FLERR, "LAMMPS-JAX runtime is not initialized");
  // The global energy/virial flags are requested generously (every setup and
  // thermo step, whatever the thermo style prints), so they cannot trigger
  // hard errors. Unavailable quantities stay zero (ev_init cleared them) and
  // init_style logs that once. Per-atom flags and ENERGY_ONLY are precise
  // consumer signals (per-atom arrays are never allocated; ENERGY_ONLY is set
  // manually by MC-style fixes), so those must abort when unavailable.
  if (eflag_atom) error->all(FLERR, "pair jax/kk does not support per-atom energy output");
  if (vflag_atom) error->all(FLERR, "pair jax/kk does not support per-atom virial output");
  if (eflag_only && bundle.programs.energy_mlir.empty())
    error->all(FLERR,
               "LAMMPS-JAX bundle does not provide the energy program required for "
               "energy-only (MC) evaluation");
  const bool energy_only_step = eflag_only != 0;
  const bool include_energy =
      energy_only_step || (eflag_global && !bundle.programs.energy_and_forces_mlir.empty());

  const int nlocal = atom->nlocal;
  const int nghost = atom->nghost;
  const int nall = nlocal + nghost;

  auto *klist = dynamic_cast<NeighListKokkos<LMPDeviceType> *>(list);
  if (klist == nullptr) error->all(FLERR, "LAMMPS-JAX requires a Kokkos device neighbor list");
  const bool rebuild_edges = !edge_cache_valid || (neighbor->ago == 0);

  atomKK->sync(execution_space, datamask_read);
  atomKK->modified(execution_space, datamask_modify);

  using AT = ArrayTypes<LMPDeviceType>;
  typename AT::t_kkfloat_1d_3_lr_randomread x = atomKK->k_x.view<LMPDeviceType>();
  typename AT::t_int_1d_randomread type = atomKK->k_type.view<LMPDeviceType>();
  typename AT::t_kkacc_1d_3 f = atomKK->k_f.view<LMPDeviceType>();
  CUstream stream = reinterpret_cast<CUstream>(exec.cuda_stream());

  pack_atoms(nall, x, type);
  pack_edges(klist, x, rebuild_edges);
  if (bundle.contract.uses_box) pack_box();

  Kokkos::deep_copy(exec, d_nlocal, nlocal);
  Kokkos::deep_copy(exec, d_nghost, nghost);
  if (rebuild_edges) {
    // Atom and edge counts only change on reneighbor steps, so capacity is
    // checked here. One max-allreduce keeps the overflow error collective:
    // every rank reaches the same error->all instead of deadlocking when only
    // one rank exceeds capacity.
    int edge_overflow = 0;
    int edge_count = 0;
    Kokkos::deep_copy(edge_overflow, d_edge_overflow);
    Kokkos::deep_copy(edge_count, d_edge_count);
    const int local_counts[3] = {nall, edge_count, edge_overflow};
    int global_counts[3] = {nall, edge_count, edge_overflow};
    MPI_Allreduce(local_counts, global_counts, 3, MPI_INT, MPI_MAX, world);
    if (global_counts[0] > bundle.contract.max_atoms)
      error->all(FLERR, "LAMMPS-JAX atom capacity exceeded: global max {} atoms, capacity {}",
                 global_counts[0], bundle.contract.max_atoms);
    if (global_counts[2])
      error->all(FLERR, "LAMMPS-JAX edge capacity exceeded: global max {} edges, capacity {}",
                 global_counts[1], bundle.contract.max_edges);
    cached_edge_count = edge_count;
    edge_cache_valid = true;
  }

  CUevent ready_event = nullptr;
  try {
    ready_event = record_ready_event(stream);
    pjrt::ExecutionRequest request = make_request(stream, ready_event);
    pjrt::ExecutionResult result = energy_only_step ? runtime->execute_energy(request)
        : include_energy ? runtime->execute_energy_force(request)
                         : runtime->execute_force(request);
    if (include_energy) eng_vdwl += scale * result.energy;
    if (!energy_only_step) {
      runtime->consume_force_output(stream, [&](CUdeviceptr force_output) {
        add_model_forces(force_output, f, nlocal, nall);
      });
      if (vflag_fdotr) {
        // Device f-dot-r over owned + ghost rows; only reachable with newton
        // on (LAMMPS uses per-pair tallying for newton off, which this style
        // cannot provide). The reduce is stream-ordered after the force add.
        EV_FLOAT virial_acc;
        Kokkos::parallel_reduce("LAMMPSJAX::virial_fdotr",
                                Kokkos::RangePolicy<LMPDeviceType>(exec, 0, nall),
                                VirialFDotRFunctor<LMPDeviceType>{x, f}, virial_acc);
        for (int n = 0; n < 6; ++n) virial[n] += static_cast<double>(virial_acc.v[n]);
        vflag_fdotr = 0;
      }
    }
  } catch (const std::exception &e) {
    if (ready_event != nullptr) cuEventDestroy(ready_event);
    error->all(FLERR, "LAMMPS-JAX compute failed: {}", e.what());
  }
  if (ready_event != nullptr) cuEventDestroy(ready_event);
#endif
}
