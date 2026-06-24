#include "pjrt/handles.h"

#include <cuda.h>

#include <sstream>
#include <stdexcept>

namespace pjrt {

namespace {

// Extracts the message from a PJRT_Error and destroys the error object.
// Returns "unknown PJRT error" when the message cannot be read. A null API is
// tolerated because this path is already reporting a failure.
std::string error_message_and_destroy(const PJRT_Api *api, PJRT_Error *error)
{
  if (error == nullptr) return "";
  std::string message = "unknown PJRT error";
  if (api != nullptr && api->PJRT_Error_Message != nullptr) {
    PJRT_Error_Message_Args args{};
    args.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
    args.error = error;
    api->PJRT_Error_Message(&args);
    if (args.message != nullptr) message.assign(args.message, args.message_size);
  }
  if (api != nullptr && api->PJRT_Error_Destroy != nullptr) {
    PJRT_Error_Destroy_Args args{};
    args.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
    args.error = error;
    api->PJRT_Error_Destroy(&args);
  }
  return message;
}

} // namespace

void check(const PJRT_Api *api, PJRT_Error *error, const std::string &operation)
{
  if (error == nullptr) return;
  throw std::runtime_error(operation + " failed: " + error_message_and_destroy(api, error));
}

void check_cuda(int cu_result, const std::string &operation)
{
  const CUresult result = static_cast<CUresult>(cu_result);
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

void BufferDeleter::operator()(PJRT_Buffer *buffer) const
{
  if (!buffer) return;
  PJRT_Buffer_Destroy_Args args{};
  args.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
  args.buffer = buffer;
  api->PJRT_Buffer_Destroy(&args);
}

void EventDeleter::operator()(PJRT_Event *event) const
{
  if (!event) return;
  PJRT_Event_Destroy_Args args{};
  args.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
  args.event = event;
  api->PJRT_Event_Destroy(&args);
}

void ExecutableDeleter::operator()(PJRT_LoadedExecutable *executable) const
{
  if (!executable) return;
  PJRT_LoadedExecutable_Destroy_Args args{};
  args.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
  args.executable = executable;
  api->PJRT_LoadedExecutable_Destroy(&args);
}

} // namespace pjrt
