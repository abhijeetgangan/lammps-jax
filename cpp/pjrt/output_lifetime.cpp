#include "pjrt/output_lifetime.h"

#include <stdexcept>
#include <string>
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

void OutputLifetime::retain_force_output(BufferPtr buffer, device::Pointer pointer)
{
  pending_force_output_ = std::move(buffer);
  pending_force_pointer_ = pointer;
}

void OutputLifetime::consume_force_output(device::Stream consumer_stream,
                                          const std::function<void(device::Pointer)> &consumer)
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

void OutputLifetime::retire_pending(device::Stream consumer_stream)
{
  // Reserve and record before moving: a throw must leave the pending buffer alive.
  retired_outputs_.reserve(retired_outputs_.size() + 1);
  device::Event ready_event = device::create_event();
  try {
    device::record_event(ready_event, consumer_stream);
  } catch (...) {
    device::destroy_event(ready_event);
    throw;
  }
  retired_outputs_.push_back({std::move(pending_force_output_), ready_event});
  pending_force_pointer_ = 0;
}

void OutputLifetime::collect_retired(bool wait)
{
  // Unprovable completion releases the buffer, an accepted leak: destroy
  // under in-flight kernels is a device use-after-free. First failure rethrows last.
  std::vector<RetiredOutput> keep;
  std::string first_failure;
  for (auto &item : retired_outputs_) {
    try {
      if (wait) device::synchronize_event(item.ready_event);
      if (!wait && !device::event_complete(item.ready_event)) {
        keep.push_back(std::move(item));
        continue;
      }
      device::destroy_event(item.ready_event);
    } catch (const std::exception &error) {
      item.buffer.release();
      device::destroy_event(item.ready_event);
      if (first_failure.empty())
        first_failure = std::string(wait ? "wait for" : "query") + " retired PJRT output: " +
                        error.what();
    }
  }
  retired_outputs_.swap(keep);
  if (!first_failure.empty()) throw std::runtime_error(first_failure);
}

} // namespace pjrt
