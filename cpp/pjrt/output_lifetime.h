#ifndef PJRT_OUTPUT_LIFETIME_H
#define PJRT_OUTPUT_LIFETIME_H

#include "pjrt/handles.h"

#include <cuda.h>

#include <functional>
#include <vector>

namespace pjrt {

// Tracks the lifetime of PJRT force-output buffers handed to the host MD
// engine by raw device pointer. PJRT owns the allocation; the
// consumer enqueues work that reads it on its own stream, so destruction is
// deferred until a CUDA event recorded after that work completes.
class OutputLifetime {
 public:
  ~OutputLifetime();

  // Collects retired outputs whose consumer work completed and drops any
  // unconsumed pending output. Call at the start of every execution.
  void cleanup_before_execution();

  // Stashes the force output of the latest execution for one consumer call.
  void retain_force_output(BufferPtr buffer, CUdeviceptr pointer);

  // Invokes the consumer with the force pointer. Afterwards the buffer is
  // retired behind an event recorded on consumer_stream, even if the consumer
  // throws.
  void consume_force_output(CUstream consumer_stream,
                            const std::function<void(CUdeviceptr)> &consumer);

  // Synchronously releases everything. Call before tearing down the client.
  void reset();

 private:
  struct RetiredOutput {
    BufferPtr buffer;
    CUevent ready_event = nullptr;
  };

  void retire_pending(CUstream consumer_stream);
  void collect_retired(bool wait);

  BufferPtr pending_force_output_{nullptr, BufferDeleter{}};
  CUdeviceptr pending_force_pointer_ = 0;
  std::vector<RetiredOutput> retired_outputs_;
};

} // namespace pjrt

#endif
