#ifndef PJRT_LOADED_EXECUTABLE_H
#define PJRT_LOADED_EXECUTABLE_H

#include "pjrt/output_lifetime.h"
#include "pjrt/plugin.h"
#include "pjrt/runtime.h"

#include <cuda.h>

#include <string>

namespace pjrt {

ExecutablePtr compile_program(const PluginLibrary &library, PJRT_Client *client,
                              const std::string &stablehlo,
                              const std::string &compile_options,
                              const std::string &label, size_t expected_outputs);

ExecutionResult execute_loaded(const PluginLibrary &library, PJRT_Client *client,
                               PJRT_Device *device, CUstream input_stream,
                               PJRT_LoadedExecutable *executable,
                               const ExecutionRequest &request, bool with_energy,
                               bool with_forces, OutputLifetime &output_lifetime);

} // namespace pjrt

#endif
