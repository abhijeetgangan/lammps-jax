#ifndef PJRT_RUNTIME_H
#define PJRT_RUNTIME_H

#include "pjrt/client_session.h"
#include "pjrt/model_comm.h"
#include "pjrt/handles.h"
#include "pjrt/output_lifetime.h"
#include "pjrt/plugin.h"

#include <cuda.h>

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace pjrt {

enum class ElementType {
  Pred,
  S32,
  F32,
  F64,
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
  // Element type of energy and force outputs; validated before readback.
  ElementType scalar_type = ElementType::F32;
  // Owned/ghost row split for communicating exchange handlers; ignored otherwise.
  int nlocal = 0;
  int nghost = 0;
};

struct ExecutionResult {
  double energy = 0.0;
};

// Communication schedule from comm_widths; empty disables the exchange machinery.
struct CommConfig {
  int max_atoms = 0;
  std::vector<int> widths;
  // Services exchange requests on the engine's MPI thread.
  ModelComm::ServiceCallback callback;
};

// Executes exported programs against engine-owned device buffers on the
// engine's stream; communicating runs use the worker, the caller services MPI.
class Runtime {
 public:
  Runtime() = default;
  Runtime(const Runtime &) = delete;
  Runtime &operator=(const Runtime &) = delete;
  ~Runtime();

  void initialize(const std::string &plugin_path, const std::string &force_mlir,
                  const std::string &energy_mlir, const std::string &energy_and_forces_mlir,
                  const std::string &compile_options, const ClientOptions &client_options = {},
                  const CommConfig &comm_config = {},
                  const std::vector<std::string> &custom_call_targets = {});
  void close();

  ExecutionResult execute_force(const ExecutionRequest &request);
  ExecutionResult execute_energy(const ExecutionRequest &request);
  ExecutionResult execute_energy_force(const ExecutionRequest &request);

  // Lends the latest force output pointer; the buffer retires behind a consumer-stream event.
  void consume_force_output(CUstream consumer_stream,
                            const std::function<void(CUdeviceptr)> &consumer);

  // Null for bundles without communication.
  ModelComm *model_comm() const { return model_comm_.get(); }

 private:
  ExecutionResult run_with_comm(const ExecutionRequest &request,
                                const std::function<ExecutionResult()> &execute);
  void worker_loop();
  void stop_worker();

  // Model kernels keep thread-local state, so executions share one long-lived worker thread.
  std::thread worker_;
  std::mutex worker_mutex_;
  std::condition_variable worker_cv_;
  std::packaged_task<ExecutionResult()> worker_task_;
  bool worker_has_task_ = false;
  bool worker_stop_ = false;

  PluginLibrary library_;
  ClientSession session_;
  ExecutablePtr force_executable_{nullptr, ExecutableDeleter{}};
  ExecutablePtr energy_executable_{nullptr, ExecutableDeleter{}};
  ExecutablePtr energy_force_executable_{nullptr, ExecutableDeleter{}};
  OutputLifetime output_lifetime_;
  std::unique_ptr<ModelComm> model_comm_;
};

} // namespace pjrt

#endif
