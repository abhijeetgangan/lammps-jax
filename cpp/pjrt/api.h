#ifndef PJRT_API_H
#define PJRT_API_H

// Prefer a real PJRT header from jaxlib or XLA, then a bare copy, then the vendored fallback.
#if __has_include("xla/pjrt/c/pjrt_c_api.h")
#include "xla/pjrt/c/pjrt_c_api.h"
#elif __has_include("pjrt_c_api.h")
#include "pjrt_c_api.h"
#elif __has_include("third_party/pjrt_c_api.h")
#include "third_party/pjrt_c_api.h"
#else
#error "pjrt requires the PJRT C API header on the include path"
#endif

#include <cstddef>
#include <cstdint>

namespace pjrt {

// The CUDA stream extension is absent from the public header; this mirrors the CUDA plugin.
struct PJRT_Get_Stream_For_External_Ready_Events_Args_Local {
  size_t struct_size;
  PJRT_Device *device;
  intptr_t stream;
};

using PJRT_Get_Stream_For_External_Ready_Events_Fn_Local =
    PJRT_Error *(*)(PJRT_Get_Stream_For_External_Ready_Events_Args_Local *args);

struct PJRT_Stream_Extension_Local {
  PJRT_Extension_Base base;
  PJRT_Get_Stream_For_External_Ready_Events_Fn_Local get_stream;
};

} // namespace pjrt

#endif
