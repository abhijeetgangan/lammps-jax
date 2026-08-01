#include "pjrt/output_lifetime.h"

#include <stdexcept>
#include <utility>

namespace pjrt {

OutputLifetime::~OutputLifetime()
{
  try {
    reset();
  } catch (...) {
  }
}

void OutputLifetime::cleanup_before_execution()
{
  collect_retired(false);
  pending_force_output_.reset();
  pending_force_pointer_ = 0;
}

void OutputLifetime::retain_force_output(BufferPtr buffer, CUdeviceptr pointer)
{
  pending_force_output_ = std::move(buffer);
  pending_force_pointer_ = pointer;
}

void OutputLifetime::consume_force_output(CUstream consumer_stream,
                                          const std::function<void(CUdeviceptr)> &consumer)
{
  if (!pending_force_output_ || pending_force_pointer_ == 0)
    throw std::runtime_error("No PJRT force output is available");
  try {
    consumer(pending_force_pointer_);
  } catch (...) {
    retire_pending(consumer_stream);
    throw;
  }
  retire_pending(consumer_stream);
  collect_retired(false);
}

void OutputLifetime::reset()
{
  collect_retired(true);
  pending_force_output_.reset();
  pending_force_pointer_ = 0;
}

void OutputLifetime::retire_pending(CUstream consumer_stream)
{
  // Reserve and record before moving: a throw must leave the pending buffer alive.
  retired_outputs_.reserve(retired_outputs_.size() + 1);
  CUevent ready_event = nullptr;
  check_cuda(cuEventCreate(&ready_event, CU_EVENT_DISABLE_TIMING), "create CUDA retirement event");
  const CUresult record = cuEventRecord(ready_event, consumer_stream);
  if (record != CUDA_SUCCESS) {
    cuEventDestroy(ready_event);
    check_cuda(record, "record CUDA retirement event");
  }
  retired_outputs_.push_back({std::move(pending_force_output_), ready_event});
  pending_force_pointer_ = 0;
}

void OutputLifetime::collect_retired(bool wait)
{
  // Unprovable completion releases the buffer, an accepted leak: destroy
  // under in-flight kernels is a device use-after-free. First failure rethrows last.
  std::vector<RetiredOutput> keep;
  CUresult first_failure = CUDA_SUCCESS;
  const char *failure_context = nullptr;
  for (auto &item : retired_outputs_) {
    if (wait) {
      const CUresult sync = cuEventSynchronize(item.ready_event);
      if (sync == CUDA_SUCCESS) {
        cuEventDestroy(item.ready_event);
      } else {
        item.buffer.release();
        cuEventDestroy(item.ready_event);
        if (first_failure == CUDA_SUCCESS) {
          first_failure = sync;
          failure_context = "wait for retired PJRT output";
        }
      }
      continue;
    }
    const CUresult query = cuEventQuery(item.ready_event);
    if (query == CUDA_SUCCESS) {
      cuEventDestroy(item.ready_event);
    } else if (query == CUDA_ERROR_NOT_READY) {
      keep.push_back(std::move(item));
    } else {
      item.buffer.release();
      cuEventDestroy(item.ready_event);
      if (first_failure == CUDA_SUCCESS) {
        first_failure = query;
        failure_context = "query retired PJRT output";
      }
    }
  }
  retired_outputs_.swap(keep);
  if (first_failure != CUDA_SUCCESS) check_cuda(first_failure, failure_context);
}

} // namespace pjrt
