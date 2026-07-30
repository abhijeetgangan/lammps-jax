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
#include <limits>
#include <stdexcept>

using namespace LAMMPS_NS;

namespace {

#ifdef KOKKOS_ENABLE_CUDA
// Functors template on the contract precision Scalar: f32 pack narrows, f64 pack copies.
template <class DeviceType, typename Scalar>
struct PackAtomsFunctor {
  using AT = ArrayTypes<DeviceType>;
  typename AT::t_kkfloat_1d_3_lr_randomread x;
  typename AT::t_int_1d_randomread type;
  Kokkos::View<Scalar *[3], Kokkos::LayoutRight, DeviceType> positions;
  Kokkos::View<int *, DeviceType> species;

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const
  {
    positions(i, 0) = static_cast<Scalar>(x(i, 0));
    positions(i, 1) = static_cast<Scalar>(x(i, 1));
    positions(i, 2) = static_cast<Scalar>(x(i, 2));
    // The model's species input is 0-based; LAMMPS types are 1-based.
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
  // Local-atom rows, plus ghost-atom rows when n_hops exceeds 1.
  int num_rows;
  bool duplicate_reverse_edges;
  bool half_edges;

  // copymode makes each per-launch functor copy's ~NeighListKokkos skip freeing shared storage.
  ~PackNeighborFunctor() { list.copymode = 1; }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int flat) const
  {
    const int ii = flat / max_neighbors_per_atom;
    const int jj = flat - ii * max_neighbors_per_atom;
    if (ii >= num_rows) return;
    const int i = list.d_ilist(ii);
    if (i >= max_atoms || jj >= list.d_numneigh(i)) return;
    const int j = list.d_neighbors(i, jj) & NEIGHMASK;
    if (j >= max_atoms) return;
    // Half-edge bundles keep one direction per pair; symmetrized models would double count.
    if (half_edges && j < i) return;

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

// Adds model force rows into f: nall rows with newton on, nlocal rows with newton off.
template <class DeviceType, typename Scalar>
struct AddForcesFunctor {
  using AT = ArrayTypes<DeviceType>;
  typename AT::t_kkacc_1d_3 f;
  Kokkos::View<const Scalar *[3], Kokkos::LayoutRight, DeviceType,
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

// Launched over [0, cached_edge_count): packed rows are contiguous, padding never in range.
template <class DeviceType, typename Scalar>
struct AddEdgeForcesFunctor {
  using AT = ArrayTypes<DeviceType>;
  typename AT::t_kkacc_1d_3 f;
  Kokkos::View<const Scalar *[3], Kokkos::LayoutRight, DeviceType,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      edge_forces;
  Kokkos::View<int *, DeviceType> senders;
  Kokkos::View<int *, DeviceType> receivers;
  bool newton_pair;
  double scale;

  KOKKOS_INLINE_FUNCTION
  void operator()(const int edge) const
  {
    // Senders are owned atoms: edges come from local-list rows 0..inum-1 in both newton modes.
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

// Device tally matching Pair::virial_fdotr_compute; runs over owned and ghost rows.
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
  // Kokkos zero-init is the padding contract; only the contract-precision views are allocated.
  dispatch_by_precision([&](auto p) { allocate_staging_as<decltype(p)>(max_atoms); });
  d_species = int_view("lammps_jax_species", max_atoms);
  d_nlocal = scalar_int_view("lammps_jax_nlocal");
  d_nghost = scalar_int_view("lammps_jax_nghost");
  d_edge_count = scalar_int_view("lammps_jax_edge_count");
  d_edge_overflow = scalar_int_view("lammps_jax_edge_overflow");
  d_senders = int_view("lammps_jax_senders", max_edges);
  d_receivers = int_view("lammps_jax_receivers", max_edges);
  d_edge_mask = bool_view("lammps_jax_edge_mask", max_edges);
#endif
}

#ifdef KOKKOS_ENABLE_CUDA
template <typename Scalar>
void PairJaxKokkos::allocate_staging_as(int max_atoms)
{
  auto &views = staging<Scalar>();
  views.positions = positions_view<Scalar>("lammps_jax_positions", max_atoms);
  if (bundle.contract.uses_box) {
    views.box = box_view<Scalar>("lammps_jax_box");
    views.host_box = host_box_view<Scalar>("lammps_jax_host_box");
  }
}
#endif

bool PairJaxKokkos::edge_force_enabled() const
{
  return bundle.contract.force_layout == lammps_jax::ForceLayout::Edge;
}

bool PairJaxKokkos::comm_enabled() const
{
  return !bundle.contract.comm_widths.empty();
}

bool PairJaxKokkos::f64_enabled() const
{
  return bundle.contract.precision == lammps_jax::Precision::Float64;
}

// Pair comm buffers are opaque doubles end to end; two f32 features bit-pack per double slot.
static inline int comm_double_slots(int width)
{
  return (width + 1) / 2;
}

// Runs on the LAMMPS MPI thread while execution waits in the FFI handler; staging is pinned host.
void PairJaxKokkos::service_model_comm(const pjrt::ModelCommRequest &request)
{
  comm_rows = request.host_rows;
  comm_width = request.width;
  const ExecutionSpace saved_space = execution_space;
  execution_space = Host;
  if (request.forward)
    comm->forward_comm(this, comm_double_slots(comm_width));
  else
    comm->reverse_comm(this, comm_double_slots(comm_width));
  execution_space = saved_space;
  comm_rows = nullptr;
  comm_width = 0;
}

int PairJaxKokkos::pack_forward_comm(int n, int *list, double *buf, int /*pbc_flag*/, int * /*pbc*/)
{
  // Feature rows are not coordinates, so periodic image shifts do not apply.
  const int slots = comm_double_slots(comm_width);
  for (int i = 0; i < n; ++i) {
    const float *row = comm_rows + static_cast<size_t>(list[i]) * comm_width;
    float *packed = reinterpret_cast<float *>(buf + static_cast<size_t>(i) * slots);
    for (int w = 0; w < comm_width; ++w) packed[w] = row[w];
    if (comm_width & 1) packed[comm_width] = 0.0f;
  }
  return n * slots;
}

void PairJaxKokkos::unpack_forward_comm(int n, int first, double *buf)
{
  const int slots = comm_double_slots(comm_width);
  for (int i = 0; i < n; ++i) {
    float *row = comm_rows + static_cast<size_t>(first + i) * comm_width;
    const float *packed = reinterpret_cast<const float *>(buf + static_cast<size_t>(i) * slots);
    for (int w = 0; w < comm_width; ++w) row[w] = packed[w];
  }
}

int PairJaxKokkos::pack_reverse_comm(int n, int first, double *buf)
{
  const int slots = comm_double_slots(comm_width);
  for (int i = 0; i < n; ++i) {
    const float *row = comm_rows + static_cast<size_t>(first + i) * comm_width;
    float *packed = reinterpret_cast<float *>(buf + static_cast<size_t>(i) * slots);
    for (int w = 0; w < comm_width; ++w) packed[w] = row[w];
    if (comm_width & 1) packed[comm_width] = 0.0f;
  }
  return n * slots;
}

void PairJaxKokkos::unpack_reverse_comm(int n, int *list, double *buf)
{
  // Adjoint accumulation: ghost-row cotangents sum into their owner rows.
  const int slots = comm_double_slots(comm_width);
  for (int i = 0; i < n; ++i) {
    float *row = comm_rows + static_cast<size_t>(list[i]) * comm_width;
    const float *packed = reinterpret_cast<const float *>(buf + static_cast<size_t>(i) * slots);
    for (int w = 0; w < comm_width; ++w) row[w] += packed[w];
  }
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
  } catch (const std::exception &e) {
    // error->one: a single rank can fail the load, and error->all would barrier-deadlock.
    error->one(FLERR, "Failed to load LAMMPS-JAX model: {}", e.what());
  }

  // Outside any try: error->all throws, and a catch would nest it into an MPI_Abort.
  if (bundle.contract.unit_style != update->unit_style)
    error->all(FLERR, "LAMMPS-JAX model unit style does not match current units");
  if (bundle.contract.n_species > 0 && atom->ntypes > bundle.contract.n_species)
    error->all(FLERR,
               "LAMMPS-JAX model distinguishes {} species but the system defines {} atom "
               "types; types beyond the model's range would silently map to its last species",
               bundle.contract.n_species, atom->ntypes);

  try {
    pjrt::ClientOptions client_options;
#ifdef KOKKOS_ENABLE_CUDA
    client_options.visible_device = exec.cuda_device();
#endif
    if (const char *fraction = std::getenv("LAMMPS_JAX_MEM_FRACTION"))
      client_options.memory_fraction = std::atof(fraction);
    pjrt::CommConfig comm_config;
    if (comm_enabled()) {
      comm_config.max_atoms = bundle.contract.max_atoms;
      comm_config.widths = bundle.contract.comm_widths;
      comm_config.callback =
          [this](const pjrt::ModelCommRequest &request) { service_model_comm(request); };
    }
    runtime = std::make_unique<pjrt::Runtime>();
    runtime->initialize(
        pjrt::resolve_plugin_path(pjrt_plugin_path, "LAMMPS_JAX_PJRT_PLUGIN_PATH"),
        bundle.programs.force_mlir, bundle.programs.energy_mlir,
        bundle.programs.energy_and_forces_mlir,
        bundle.compile_options, client_options, comm_config,
        bundle.contract.custom_call_targets);
    if (comm_enabled()) {
      // LAMMPS sizes pair comm buffers once at init, in doubles; two f32 features per slot.
      int max_width = 0;
      for (const int width : bundle.contract.comm_widths) max_width = MAX(max_width, width);
      comm_forward = (max_width + 1) / 2;
      comm_reverse = (max_width + 1) / 2;
    }
    allocate_device_buffers();
    model_loaded = true;
  } catch (const std::exception &e) {
    // error->one: driver state and device memory make PJRT setup rank-local.
    error->one(FLERR, "Failed to initialize LAMMPS-JAX runtime: {}", e.what());
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
    // The graph applies full weight to every listed pair; partial special_bonds would be ignored.
    for (int n = 1; n <= 3; ++n) {
      if (force->special_lj[n] != 1.0 &&
          !(force->special_lj[n] == 0.0 && force->special_coul[n] == 0.0))
        error->all(FLERR,
                   "pair jax/kk does not apply special_bonds factors; use 1 1 1 or full exclusion");
    }
  }
  if (!force->newton_pair) {
    // Pair virial needs newton-on ghost force rows; a barostat would see incomplete pressure.
    for (int i = 0; i < modify->nfix; ++i) {
      const char *style = modify->fix[i]->style;
      if (strstr(style, "npt") || strstr(style, "nph") || strstr(style, "press") ||
          strstr(style, "box/relax") || strstr(style, "msst"))
        error->all(FLERR,
                   "Fix {} requires pair virial; run pair jax/kk with newton pair on",
                   style);
    }
  }
  const bool multi_hop = bundle.contract.n_hops > 1;
  if (comm->me == 0) {
    if (comm_enabled()) {
      int max_width = 0;
      for (const int width : bundle.contract.comm_widths) max_width = MAX(max_width, width);
      utils::logmesg(lmp,
                     "LAMMPS-JAX: communicating bundle ({} exchange site(s), max width {}); "
                     "node features are synchronized per layer through LAMMPS "
                     "forward/reverse communication with a one-cutoff ghost shell.\n",
                     bundle.contract.comm_widths.size(), max_width);
    }
    else if (multi_hop)
      utils::logmesg(lmp,
                     "LAMMPS-JAX: n_hops = {}; extending the ghost shell "
                     "and evaluating node features redundantly on ghost atoms.\n",
                     bundle.contract.n_hops);
    else if (force->newton_pair)
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
  if (comm_enabled()) {
    // Full list: owned atoms need complete rows; ghost features arrive via the exchange.
    if (atom->molecular != Atom::ATOMIC) {
      for (int n = 1; n <= 3; ++n) {
        if (force->special_lj[n] != 1.0)
          error->all(FLERR,
                     "Communicating LAMMPS-JAX bundles require special_bonds 1 1 1");
      }
    }
    auto request = neighbor->add_request(this, NeighConst::REQ_FULL);
    request->set_kokkos_device(true);
    request->set_kokkos_host(false);
    return;
  }
  if (multi_hop) {
    if (atom->molecular != Atom::ATOMIC) {
      // Kokkos ghost builds skip special-bonds, making ghost-row features decomposition-dependent.
      for (int n = 1; n <= 3; ++n) {
        if (force->special_lj[n] != 1.0)
          error->all(FLERR,
                     "LAMMPS-JAX bundles with n_hops > 1 require special_bonds 1 1 1; "
                     "ghost-atom neighbor rows do not apply exclusions");
      }
    }
    // T hops need ghosts to T*r_cut plus skin; ghost forces return by reverse communication.
    const double needed_cutoff =
        bundle.contract.n_hops * bundle.contract.cutoff + neighbor->skin;
    if (comm->get_comm_cutoff() < needed_cutoff) {
      const std::string cutoff_val = std::to_string(needed_cutoff);
      char *args[2];
      args[0] = (char *) "cutoff";
      args[1] = const_cast<char *>(cutoff_val.c_str());
      comm->modify_params(2, args);
      if (comm->me == 0)
        error->warning(FLERR, "pair jax/kk is setting the communication cutoff to {}",
                       cutoff_val);
    }
    auto request = neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_GHOST);
    request->set_kokkos_device(true);
    request->set_kokkos_host(false);
    return;
  }
  // Newton on: half list over owned rows. Newton off: full list, no REQ_GHOST.
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
  // Stale rows past a shrinking nall reach only masked edges; the clamp bounds writes.
  const int span = std::min(nall, bundle.contract.max_atoms);
  dispatch_by_precision([&](auto p) { pack_atoms_as<decltype(p)>(span, x, type); });
}

template <typename Scalar>
void PairJaxKokkos::pack_atoms_as(int span, const PairJaxKokkos::x_view &x,
                                  const PairJaxKokkos::type_view &type)
{
  Kokkos::parallel_for("LAMMPSJAX::pack_atoms", Kokkos::RangePolicy<LMPDeviceType>(exec, 0, span),
                       PackAtomsFunctor<LMPDeviceType, Scalar>{x, type, staging<Scalar>().positions,
                                                               d_species});
}

void PairJaxKokkos::pack_edges(NeighListKokkos<PairJaxKokkos::device_type> *klist,
                               bool rebuild_edges)
{
  if (rebuild_edges) {
    Kokkos::deep_copy(exec, d_edge_count, 0);
    Kokkos::deep_copy(exec, d_edge_overflow, 0);
    Kokkos::deep_copy(exec, d_edge_mask, false);
    Kokkos::deep_copy(exec, d_senders, bundle.contract.max_atoms);
    Kokkos::deep_copy(exec, d_receivers, bundle.contract.max_atoms);

    const int max_neighbors_per_atom = std::max(1, klist->maxneighs);
    // Ghost features feed owned energies, so every ghost needs its full neighborhood.
    const bool multi_hop = bundle.contract.n_hops > 1;
    // The collective capacity check errors this step; the clamp bounds the flat index.
    const int num_rows = std::min(klist->inum + (multi_hop ? klist->gnum : 0),
                                  bundle.contract.max_atoms);
    const long long flat_extent =
        static_cast<long long>(num_rows) * max_neighbors_per_atom;
    if (flat_extent > static_cast<long long>(std::numeric_limits<int>::max()))
      error->one(FLERR,
                 "LAMMPS-JAX neighbor pack index overflows int; reduce the bundle "
                 "max_atoms or the neigh_modify one setting");
    // Ghost and comm full lists already carry both edge directions.
    const bool duplicate_reverse_edges =
        force->newton_pair && !edge_force_enabled() && !multi_hop && !comm_enabled();
    Kokkos::parallel_for(
        "LAMMPSJAX::pack_neighbors",
        Kokkos::RangePolicy<LMPDeviceType>(exec, 0, num_rows * max_neighbors_per_atom),
        PackNeighborFunctor<LMPDeviceType>{*klist, d_senders, d_receivers, d_edge_mask,
                                           d_edge_count, d_edge_overflow,
                                           bundle.contract.max_atoms, bundle.contract.max_edges,
                                           max_neighbors_per_atom, num_rows,
                                           duplicate_reverse_edges,
                                           bundle.contract.half_edges});
  }
}

namespace {

// Row-vector cell [[xprd,0,0],[xy,yprd,0],[xz,yz,zprd]] at the contract precision.
template <class HostBoxView>
void fill_host_box(const HostBoxView &h_box, const Domain *domain)
{
  using Scalar = typename HostBoxView::non_const_value_type;
  h_box(0, 0) = static_cast<Scalar>(domain->xprd);
  h_box(0, 1) = Scalar(0);
  h_box(0, 2) = Scalar(0);
  h_box(1, 0) = static_cast<Scalar>(domain->xy);
  h_box(1, 1) = static_cast<Scalar>(domain->yprd);
  h_box(1, 2) = Scalar(0);
  h_box(2, 0) = static_cast<Scalar>(domain->xz);
  h_box(2, 1) = static_cast<Scalar>(domain->yz);
  h_box(2, 2) = static_cast<Scalar>(domain->zprd);
}

} // namespace

void PairJaxKokkos::pack_box()
{
  // Repacked every step so box-changing runs such as NPT stay consistent.
  dispatch_by_precision([&](auto p) { pack_box_as<decltype(p)>(); });
}

template <typename Scalar>
void PairJaxKokkos::pack_box_as()
{
  auto &views = staging<Scalar>();
  fill_host_box(views.host_box, domain);
  Kokkos::deep_copy(exec, views.box, views.host_box);
}

pjrt::ExecutionRequest PairJaxKokkos::make_request(CUstream stream, CUevent ready_event)
{
  return dispatch_by_precision(
      [&](auto p) { return make_request_as<decltype(p)>(stream, ready_event); });
}

template <typename Scalar>
pjrt::ExecutionRequest PairJaxKokkos::make_request_as(CUstream stream, CUevent ready_event)
{
  auto &views = staging<Scalar>();
  const pjrt::ElementType real_type =
      std::is_same_v<Scalar, double> ? pjrt::ElementType::F64 : pjrt::ElementType::F32;
  const CUdeviceptr positions_pointer = reinterpret_cast<CUdeviceptr>(views.positions.data());
  pjrt::ExecutionRequest request;
  request.stream = stream;
  request.input_ready_event = ready_event;
  request.scalar_type = real_type;
  request.inputs = {
      {"positions", positions_pointer, {bundle.contract.max_atoms, 3}, real_type},
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
    const CUdeviceptr box_pointer = reinterpret_cast<CUdeviceptr>(views.box.data());
    request.inputs.emplace_back("box", box_pointer, std::vector<int64_t>{3, 3}, real_type);
  }
  return request;
}

template <typename Scalar>
void PairJaxKokkos::add_model_forces_as(CUdeviceptr force_output, const PairJaxKokkos::f_view &f,
                                        int nlocal, int nall)
{
  using ForceView = Kokkos::View<const Scalar *[3], Kokkos::LayoutRight, LMPDeviceType,
                                Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  if (edge_force_enabled()) {
    ForceView edge_forces(reinterpret_cast<const Scalar *>(force_output), bundle.contract.max_edges);
    Kokkos::parallel_for(
        "LAMMPSJAX::add_edge_forces",
        Kokkos::RangePolicy<LMPDeviceType>(exec, 0, cached_edge_count),
        AddEdgeForcesFunctor<LMPDeviceType, Scalar>{f, edge_forces, d_senders, d_receivers,
                                                    static_cast<bool>(force->newton_pair), scale});
  } else {
    ForceView model_forces(reinterpret_cast<const Scalar *>(force_output), bundle.contract.max_atoms);
    const int limit = force->newton_pair ? nall : nlocal;
    Kokkos::parallel_for("LAMMPSJAX::add_forces", Kokkos::RangePolicy<LMPDeviceType>(exec, 0, limit),
                         AddForcesFunctor<LMPDeviceType, Scalar>{f, model_forces, scale});
  }
}

void PairJaxKokkos::add_model_forces(CUdeviceptr force_output, const PairJaxKokkos::f_view &f,
                                     int nlocal, int nall)
{
  // Model force rows are f32 or f64 per the contract precision.
  dispatch_by_precision(
      [&](auto p) { add_model_forces_as<decltype(p)>(force_output, f, nlocal, nall); });
}
#endif

void PairJaxKokkos::compute(int eflag, int vflag)
{
#ifndef KOKKOS_ENABLE_CUDA
  error->all(FLERR, "pair_style jax/kk requires CUDA");
#else
  ev_init(eflag, vflag, 0);
  if (!model_loaded || runtime == nullptr) error->all(FLERR, "LAMMPS-JAX runtime is not initialized");
  // Global energy/virial flags fire every thermo step, so unavailable values
  // stay zero; per-atom flags and ENERGY_ONLY are real signals and must abort.
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
  pack_edges(klist, rebuild_edges);
  if (bundle.contract.uses_box) pack_box();

  Kokkos::deep_copy(exec, d_nlocal, nlocal);
  Kokkos::deep_copy(exec, d_nghost, nghost);
  if (rebuild_edges) {
    // Counts change only at reneighbor; the max-allreduce keeps the overflow error collective.
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
    // The catch below destroys the event on any failure past creation.
    if (cuEventCreate(&ready_event, CU_EVENT_DISABLE_TIMING) != CUDA_SUCCESS)
      throw std::runtime_error("Failed to create CUDA event for LAMMPS-JAX inputs");
    if (cuEventRecord(ready_event, stream) != CUDA_SUCCESS)
      throw std::runtime_error("Failed to record CUDA event for LAMMPS-JAX inputs");
    pjrt::ExecutionRequest request = make_request(stream, ready_event);
    // Execution runs on a worker while this thread services exchanges through LAMMPS comm.
    request.nlocal = nlocal;
    request.nghost = nghost;
    pjrt::ExecutionResult result = energy_only_step ? runtime->execute_energy(request)
        : include_energy ? runtime->execute_energy_force(request)
                         : runtime->execute_force(request);
    if (include_energy) eng_vdwl += scale * result.energy;
    if (!energy_only_step) {
      runtime->consume_force_output(stream, [&](CUdeviceptr force_output) {
        add_model_forces(force_output, f, nlocal, nall);
      });
      if (vflag_fdotr) {
        // Device f-dot-r over owned and ghost rows, stream-ordered after the force add.
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
    // error->one: execution failures are rank-local and cannot collectivize.
    error->one(FLERR, "LAMMPS-JAX compute failed: {}", e.what());
  }
  if (ready_event != nullptr) cuEventDestroy(ready_event);
#endif
}
