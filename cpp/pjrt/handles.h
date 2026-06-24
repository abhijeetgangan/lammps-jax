#ifndef PJRT_HANDLES_H
#define PJRT_HANDLES_H

#include "pjrt/api.h"

#include <memory>
#include <string>

namespace pjrt {

// Throws std::runtime_error("<operation> failed: <message>") when error is
// non-null, destroying the error object first.
void check(const PJRT_Api *api, PJRT_Error *error, const std::string &operation);

// Throws std::runtime_error for CUDA driver failures, with name and message.
void check_cuda(int cu_result, const std::string &operation);

struct BufferDeleter {
  const PJRT_Api *api = nullptr;
  void operator()(PJRT_Buffer *buffer) const;
};

struct EventDeleter {
  const PJRT_Api *api = nullptr;
  void operator()(PJRT_Event *event) const;
};

struct ExecutableDeleter {
  const PJRT_Api *api = nullptr;
  void operator()(PJRT_LoadedExecutable *executable) const;
};

using BufferPtr = std::unique_ptr<PJRT_Buffer, BufferDeleter>;
using EventPtr = std::unique_ptr<PJRT_Event, EventDeleter>;
using ExecutablePtr = std::unique_ptr<PJRT_LoadedExecutable, ExecutableDeleter>;

} // namespace pjrt

#endif
