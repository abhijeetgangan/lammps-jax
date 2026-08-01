// Per-layer forward/reverse communication for communicating bundles.

#ifndef PJRT_MODEL_COMM_H
#define PJRT_MODEL_COMM_H

#include <cuda.h>

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

struct PJRT_Api;
typedef struct PJRT_ExecuteContext PJRT_ExecuteContext;

namespace pjrt {

struct ModelCommUserData;

// One request, serviced on the engine's MPI thread; host_rows is pinned f32
// staging, device_rows the in-place device buffer when device comm is active.
struct ModelCommRequest {
  bool forward = true;
  float *host_rows = nullptr;
  int width = 0;
  int nlocal = 0;
  int nghost = 0;
  float *device_rows = nullptr;
};

class ModelComm {
 public:
  using ServiceCallback = std::function<void(const ModelCommRequest &)>;

  ModelComm() = default;
  ModelComm(const ModelComm &) = delete;
  ModelComm &operator=(const ModelComm &) = delete;
  ~ModelComm();

  // Registers handlers and allocates pinned staging; must precede compiling communicating programs.
  void initialize(const PJRT_Api *api, int max_atoms, const std::vector<int> &widths);
  void close(const PJRT_Api *api);

  PJRT_ExecuteContext *execute_context() const { return execute_context_; }

  // The callback runs on the engine thread and performs the LAMMPS comm.
  void set_service_callback(ServiceCallback callback);

  // Per-step row counts; call before launching an execution.
  void begin_step(int nlocal, int nghost);

  // Exchange in place on device rows, skipping the pinned staging.
  void set_device_rows(bool enabled) { device_rows_ = enabled; }

  // Protocol: begin_service; worker executes and marks done; service_loop; worker.get.
  void begin_service();
  void mark_execution_done();
  void service_loop();

  // Called from FFI threads; blocks until serviced. Empty on success, else a rank-identical error.
  std::string comm_from_handler(bool forward, CUstream stream, const void *input,
                                void *output, int64_t rows, int64_t width,
                                const void *token_input, void *token_output);

 private:
  std::string validate_site(bool forward, int64_t rows, int64_t width);

  // Registration + execute-context state.
  ModelCommUserData *user_data_ = nullptr;
  PJRT_ExecuteContext *execute_context_ = nullptr;
  const PJRT_Api *api_ = nullptr;

  // Static capacity/schedule from the bundle contract.
  int max_atoms_ = 0;
  std::vector<int> widths_;

  // Pinned staging shared by all sites; the token chain serializes communications.
  float *pinned_ = nullptr;
  CUevent staged_event_ = nullptr;

  std::mutex mutex_;
  std::condition_variable condition_;
  ServiceCallback service_callback_;
  ModelCommRequest request_;
  bool request_pending_ = false;
  std::string service_error_;
  bool servicing_ = false;
  bool device_rows_ = false;
  bool done_ = false;
  int nlocal_ = 0;
  int nghost_ = 0;
  int forward_site_ = 0;
  int reverse_site_ = 0;
};

} // namespace pjrt

#endif
