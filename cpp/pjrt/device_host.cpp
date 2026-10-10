// CPU PJRT plugin backend: buffers are host memory and all work finishes before returning.

#include "pjrt/device.h"

#include <cstdlib>
#include <cstring>
#include <new>

namespace pjrt::device {

bool has_streams()
{
  return false;
}

const char *platform_name()
{
  return "Host";
}

void wait_event(Stream, Event) {}

void synchronize(Stream) {}

Event create_event()
{
  return nullptr;
}

void destroy_event(Event) {}

void record_event(Event, Stream) {}

void synchronize_event(Event) {}

bool event_complete(Event)
{
  return true;
}

void copy_to_host(void *destination, Pointer source, size_t bytes, Stream)
{
  std::memcpy(destination, reinterpret_cast<void *>(source), bytes);
}

void copy_to_device(Pointer destination, const void *source, size_t bytes, Stream)
{
  std::memcpy(reinterpret_cast<void *>(destination), source, bytes);
}

void copy_on_device(Pointer destination, Pointer source, size_t bytes, Stream)
{
  std::memmove(reinterpret_cast<void *>(destination), reinterpret_cast<void *>(source), bytes);
}

void zero(Pointer destination, size_t bytes, Stream)
{
  std::memset(reinterpret_cast<void *>(destination), 0, bytes);
}

void *allocate_pinned(size_t bytes)
{
  void *memory = std::malloc(bytes);
  if (memory == nullptr && bytes != 0) throw std::bad_alloc();
  return memory;
}

void free_pinned(void *memory)
{
  std::free(memory);
}

Context current_context()
{
  return nullptr;
}

void set_current_context(Context) {}

} // namespace pjrt::device
