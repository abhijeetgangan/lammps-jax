#ifndef PJRT_API_H
#define PJRT_API_H

// The consuming project provides the PJRT C API header on its include path:
// either the real jaxlib header or a vendored copy named pjrt_c_api.h.
#if __has_include("xla/pjrt/c/pjrt_c_api.h")
#include "xla/pjrt/c/pjrt_c_api.h"
#elif __has_include("pjrt_c_api.h")
#include "pjrt_c_api.h"
#else
#error "pjrt requires xla/pjrt/c/pjrt_c_api.h or pjrt_c_api.h on the include path"
#endif

#include <cstddef>
#include <cstdint>

namespace pjrt {

// The CUDA stream extension is not part of the public C API header; these
// mirror the layout exposed by the CUDA PJRT plugin.
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
