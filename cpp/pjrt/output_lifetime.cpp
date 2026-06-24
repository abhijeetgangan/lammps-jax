#include "pjrt/output_lifetime.h"

#include <stdexcept>
#include <utility>

namespace pjrt {

namespace {

CUevent record_event(CUstream stream)
{
  CUevent event = nullptr;
  check_cuda(cuEventCreate(&event, CU_EVENT_DISABLE_TIMING), "create CUDA retirement event");
  const CUresult record = cuEventRecord(event, stream);
  if (record != CUDA_SUCCESS) {
    cuEventDestroy(event);
    check_cuda(record, "record CUDA retirement event");
  }
  return event;
}

} // namespace

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
  // Reserve and record before moving the buffer: if either throws, the
  // pending buffer must stay alive, not be destroyed under in-flight
  // consumer kernels.
  retired_outputs_.reserve(retired_outputs_.size() + 1);
  CUevent ready_event = record_event(consumer_stream);
  retired_outputs_.push_back({std::move(pending_force_output_), ready_event});
  pending_force_pointer_ = 0;
}

void OutputLifetime::collect_retired(bool wait)
{
  std::vector<RetiredOutput> keep;
  for (auto &item : retired_outputs_) {
    if (wait) {
      check_cuda(cuEventSynchronize(item.ready_event), "wait for retired PJRT output");
      cuEventDestroy(item.ready_event);
      continue;
    }
    const CUresult query = cuEventQuery(item.ready_event);
    if (query == CUDA_SUCCESS) {
      cuEventDestroy(item.ready_event);
    } else if (query == CUDA_ERROR_NOT_READY) {
      keep.push_back(std::move(item));
    } else {
      check_cuda(query, "query retired PJRT output");
    }
  }
  retired_outputs_.swap(keep);
}

} // namespace pjrt
