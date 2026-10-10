// CUDA driver backend: work is ordered on streams and synchronized through events.

#include "pjrt/device.h"

#include <cuda.h>

#include <sstream>
#include <stdexcept>
#include <type_traits>

namespace pjrt::device {

static_assert(std::is_same_v<Pointer, CUdeviceptr>, "device::Pointer must match CUdeviceptr");

namespace {

void check(CUresult result, const char *operation)
{
  if (result == CUDA_SUCCESS) return;
  const char *name = nullptr;
  const char *message = nullptr;
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &message);
  std::ostringstream out;
  out << operation << " failed";
  if (name) out << ": " << name;
  if (message) out << " (" << message << ")";
  throw std::runtime_error(out.str());
}

CUstream stream_of(Stream stream)
{
  return static_cast<CUstream>(stream);
}

CUevent event_of(Event event)
{
  return static_cast<CUevent>(event);
}

} // namespace

bool has_streams()
{
  return true;
}

const char *platform_name()
{
  return "CUDA";
}

void wait_event(Stream stream, Event event)
{
  check(cuStreamWaitEvent(stream_of(stream), event_of(event), 0), "CUDA stream wait");
}

void synchronize(Stream stream)
{
  check(cuStreamSynchronize(stream_of(stream)), "CUDA stream synchronize");
}

Event create_event()
{
  CUevent event = nullptr;
  check(cuEventCreate(&event, CU_EVENT_DISABLE_TIMING), "create CUDA event");
  return event;
}

void destroy_event(Event event)
{
  if (event != nullptr) cuEventDestroy(event_of(event));
}

void record_event(Event event, Stream stream)
{
  check(cuEventRecord(event_of(event), stream_of(stream)), "record CUDA event");
}

void synchronize_event(Event event)
{
  check(cuEventSynchronize(event_of(event)), "synchronize CUDA event");
}

bool event_complete(Event event)
{
  const CUresult query = cuEventQuery(event_of(event));
  if (query == CUDA_ERROR_NOT_READY) return false;
  check(query, "query CUDA event");
  return true;
}

void copy_to_host(void *destination, Pointer source, size_t bytes, Stream stream)
{
  check(cuMemcpyDtoHAsync(destination, source, bytes, stream_of(stream)), "copy device to host");
}

void copy_to_device(Pointer destination, const void *source, size_t bytes, Stream stream)
{
  check(cuMemcpyHtoDAsync(destination, source, bytes, stream_of(stream)), "copy host to device");
}

void copy_on_device(Pointer destination, Pointer source, size_t bytes, Stream stream)
{
  check(cuMemcpyDtoDAsync(destination, source, bytes, stream_of(stream)), "copy on device");
}

void zero(Pointer destination, size_t bytes, Stream stream)
{
  check(cuMemsetD8Async(destination, 0, bytes, stream_of(stream)), "zero device memory");
}

void *allocate_pinned(size_t bytes)
{
  void *memory = nullptr;
  check(cuMemHostAlloc(&memory, bytes, 0), "allocate pinned host memory");
  return memory;
}

void free_pinned(void *memory)
{
  if (memory != nullptr) cuMemFreeHost(memory);
}

Context current_context()
{
  CUcontext context = nullptr;
  cuCtxGetCurrent(&context);
  return context;
}

void set_current_context(Context context)
{
  if (context != nullptr)
    check(cuCtxSetCurrent(static_cast<CUcontext>(context)), "adopt CUDA context");
}

} // namespace pjrt::device
