// Parsed model bundle: exported programs plus the contract the pair style validates.

#ifndef LAMMPS_JAX_MODEL_H
#define LAMMPS_JAX_MODEL_H

#include <string>
#include <vector>

namespace lammps_jax {

// Atom rows add per-atom forces; Edge rows hold dU/d(rij), sender plus, receiver minus.
enum class ForceLayout {
  Atom,
  Edge,
};

// newton pair setting at export; On and Off abort on mismatch, Any accepts either.
enum class NewtonMode {
  Any,
  On,
  Off,
};

// Width of positions, box, energy, forces; index inputs stay integer. Comm bundles are f32-only.
enum class Precision {
  Float32,
  Float64,
};

// Portable VHLO bytecode or legacy StableHLO text; PJRT accepts either. Empty: not exported.
struct ProgramSet {
  std::string force_mlir;
  std::string energy_mlir;
  std::string energy_and_forces_mlir;
};

struct ModelContract {
  // Fixed input capacities, checked collectively at reneighbor steps, never resized.
  int max_atoms = 0;
  int max_edges = 0;
  // Model cutoff; init_one reports it to LAMMPS as the pair cutoff.
  double cutoff = 0.0;
  // Must match the run's unit style when pair_coeff loads the bundle.
  std::string unit_style = "real";
  Precision precision = Precision::Float32;
  ForceLayout force_layout = ForceLayout::Atom;
  NewtonMode newton = NewtonMode::Any;
  // Receptive field in cutoffs; above 1 ghosts extend to n_hops*cutoff and enter the graph.
  int n_hops = 1;
  // Half-edge bundles carry each pair once; the model scatters per-edge terms to both endpoints.
  // Communicating half-edge bundles also pack each boundary pair on one rank only.
  bool half_edges = false;
  // Feature width of each exchange in forward order; ModelComm validates. Empty: non-communicating.
  std::vector<int> comm_widths;
  // External FFI custom-call targets; handlers register via LAMMPS_JAX_FFI_HANDLERS before compile.
  std::vector<std::string> custom_call_targets;
  // Model takes the cell as a trailing [3,3] row-vector matrix at the contract precision.
  bool uses_box = false;
  // Species the model distinguishes; more atom types abort pair_coeff. 0 means species-blind.
  int n_species = 0;
  // Static owned-row bound for models that truncate per-node work. 0 means untruncated.
  int max_owned = 0;
};

struct ModelBundle {
  // Format tag; semantics at load_bundle_file.
  std::string format;
  ProgramSet programs;
  // Serialized xla.CompileOptions proto, handed to PJRT_Client_Compile unchanged.
  std::string compile_options;
  ModelContract contract;
};

ModelBundle load_bundle_file(const std::string &path);

} // namespace lammps_jax

#endif
