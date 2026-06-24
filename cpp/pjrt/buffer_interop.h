#ifndef PJRT_BUFFER_INTEROP_H
#define PJRT_BUFFER_INTEROP_H

#include "pjrt/plugin.h"
#include "pjrt/runtime.h"

#include <cuda.h>

namespace pjrt {

BufferPtr create_input_view(const PluginLibrary &library, PJRT_Client *client,
                            PJRT_Device *device, const DeviceBufferSpec &spec,
                            CUstream stream);
CUdeviceptr output_pointer(const PluginLibrary &library, PJRT_Buffer *buffer);

} // namespace pjrt

#endif
