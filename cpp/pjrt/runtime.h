#ifndef PJRT_RUNTIME_H
#define PJRT_RUNTIME_H

#include "pjrt/client_session.h"
#include "pjrt/handles.h"
#include "pjrt/output_lifetime.h"
#include "pjrt/plugin.h"

#include <cuda.h>

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace pjrt {

enum class ElementType {
  Pred,
  S32,
  F32,
};

struct DeviceBufferSpec {
  std::string name;
  CUdeviceptr pointer = 0;
  std::vector<int64_t> dims;
  ElementType element_type = ElementType::F32;
};

struct ExecutionRequest {
  std::vector<DeviceBufferSpec> inputs;
  CUstream stream = nullptr;
  CUevent input_ready_event = nullptr;
};

struct ExecutionResult {
  double energy = 0.0;
  double pjrt_total_ms = 0.0;
  double pjrt_view_ms = 0.0;
  double pjrt_execute_ms = 0.0;
  double pjrt_await_ms = 0.0;
  double pjrt_output_ms = 0.0;
};

struct ProgramSet {
  std::string force_mlir;
  std::string energy_mlir;
  std::string energy_and_forces_mlir;
};

// Executes exported JAX programs against device buffers owned by the host MD
// engine, on the engine's CUDA stream, without host copies for array data.
// Empty programs leave the corresponding capability unavailable; executing an
// unavailable capability throws.
class Runtime {
 public:
  Runtime() = default;
  Runtime(const Runtime &) = delete;
  Runtime &operator=(const Runtime &) = delete;
  ~Runtime();

  void initialize(const std::string &plugin_path, const ProgramSet &programs,
                  const std::string &compile_options, const ClientOptions &client_options = {});
  void close();

  ExecutionResult execute_force(const ExecutionRequest &request);
  ExecutionResult execute_energy(const ExecutionRequest &request);
  ExecutionResult execute_energy_force(const ExecutionRequest &request);

  // Hands the force output pointer of the latest execution to the consumer;
  // the buffer is retired behind an event recorded on consumer_stream.
  void consume_force_output(CUstream consumer_stream,
                            const std::function<void(CUdeviceptr)> &consumer);

 private:
  PluginLibrary library_;
  ClientSession session_;
  ExecutablePtr force_executable_{nullptr, ExecutableDeleter{}};
  ExecutablePtr energy_executable_{nullptr, ExecutableDeleter{}};
  ExecutablePtr energy_force_executable_{nullptr, ExecutableDeleter{}};
  OutputLifetime output_lifetime_;
};

} // namespace pjrt

#endif
