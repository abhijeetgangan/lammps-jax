#ifndef LAMMPS_JAX_MODEL_H
#define LAMMPS_JAX_MODEL_H

#include <string>
#include <vector>

namespace lammps_jax {

enum class InputLayout {
  SparseEdge,
};

enum class ForceLayout {
  Atom,
  Edge,
};

// LAMMPS `newton pair` setting the bundle was exported for.
enum class NewtonMode {
  Any,
  On,
  Off,
};

struct ProgramSet {
  std::string force_mlir;
  std::string energy_mlir;
  std::string energy_and_forces_mlir;
};

struct ModelContract {
  InputLayout input_layout = InputLayout::SparseEdge;
  int max_atoms = 0;
  int max_edges = 0;
  double cutoff = 0.0;
  std::string unit_style = "real";
  std::string precision = "float32";
  ForceLayout force_layout = ForceLayout::Atom;
  NewtonMode newton = NewtonMode::Any;
  // Model receives the current cell as a trailing f32[3,3] row-vector matrix.
  bool uses_box = false;
};

struct ModelBundle {
  std::string format;
  ProgramSet programs;
  std::string compile_options;
  ModelContract contract;
};

ModelBundle load_bundle_file(const std::string &path);

} // namespace lammps_jax

#endif
