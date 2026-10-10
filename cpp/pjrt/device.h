// Device operations of the build's backend: device_cuda.cpp or device_host.cpp, chosen by CMake.

#ifndef PJRT_DEVICE_H
#define PJRT_DEVICE_H

#include <cstddef>

namespace pjrt::device {

// Same width and type as CUdeviceptr; on the host backend an address in host memory.
using Pointer = unsigned long long;
// Opaque CUstream, CUevent and CUcontext; null on the host backend.
using Stream = void *;
using Event = void *;
using Context = void *;

// True when work is ordered on streams; false when operations finish before returning.
bool has_streams();
// PJRT platform name, used for FFI handler registration.
const char *platform_name();

// Makes stream wait on device for event.
void wait_event(Stream stream, Event event);
void synchronize(Stream stream);

Event create_event();
void destroy_event(Event event);
void record_event(Event event, Stream stream);
void synchronize_event(Event event);
// True once the work before the event finished; throws if it failed.
bool event_complete(Event event);

void copy_to_host(void *destination, Pointer source, size_t bytes, Stream stream);
void copy_to_device(Pointer destination, const void *source, size_t bytes, Stream stream);
void copy_on_device(Pointer destination, Pointer source, size_t bytes, Stream stream);
void zero(Pointer destination, size_t bytes, Stream stream);

// Page-locked on CUDA so stream copies run asynchronously.
void *allocate_pinned(size_t bytes);
void free_pinned(void *memory);

// Worker threads adopt the caller's CUDA context; no-ops on the host backend.
Context current_context();
void set_current_context(Context context);

} // namespace pjrt::device

#endif
