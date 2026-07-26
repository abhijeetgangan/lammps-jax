#include "pjrt/plugin.h"

#include "pjrt/handles.h"

#if __has_include("xla/pjrt/c/pjrt_c_api_ffi_extension.h")
#include "xla/pjrt/c/pjrt_c_api_ffi_extension.h"
#else
#include "third_party/pjrt_c_api_ffi_extension.h"
#endif

#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace pjrt {

namespace {
using GetPjrtApiFn = const PJRT_Api *(*)();
}

std::string resolve_plugin_path(const std::string &explicit_path, const char *env_var)
{
  if (!explicit_path.empty()) return explicit_path;
  if (env_var != nullptr) {
    const char *env_value = std::getenv(env_var);
    if (env_value != nullptr && env_value[0] != '\0') return env_value;
  }
  return "xla_cuda_plugin.so";
}

PluginLibrary::~PluginLibrary()
{
  close();
}

void PluginLibrary::open(const std::string &path)
{
  close();
  // RTLD_NODELETE: ModelComm's once-per-process registration must survive re-initialization.
  library_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE);
  if (library_ == nullptr)
    throw std::runtime_error("Failed to load PJRT plugin '" + path + "': " + dlerror());
  auto get_api = reinterpret_cast<GetPjrtApiFn>(dlsym(library_, "GetPjrtApi"));
  if (get_api == nullptr) throw std::runtime_error("PJRT plugin is missing GetPjrtApi");
  api_ = get_api();
  if (api_ == nullptr) throw std::runtime_error("GetPjrtApi returned null");
  // Compatible only within a major version; minor skew is guarded by struct_size at use sites.
  const int plugin_major = api_->pjrt_api_version.major_version;
  const int plugin_minor = api_->pjrt_api_version.minor_version;
  if (plugin_major != PJRT_API_MAJOR)
    throw std::runtime_error("PJRT plugin '" + path + "' implements C API major version " +
                             std::to_string(plugin_major) + "." + std::to_string(plugin_minor) +
                             "; this build requires major version " +
                             std::to_string(PJRT_API_MAJOR));
  if (plugin_minor < PJRT_API_MINOR)
    std::fprintf(stderr,
                 "lammps-jax: PJRT plugin '%s' implements C API v%d.%d, older than the "
                 "v%d.%d headers this plugin was built against; newer features may be "
                 "unavailable.\n",
                 path.c_str(), plugin_major, plugin_minor, PJRT_API_MAJOR, PJRT_API_MINOR);
  if (api_->PJRT_Plugin_Initialize != nullptr) {
    PJRT_Plugin_Initialize_Args args{};
    args.struct_size = PJRT_Plugin_Initialize_Args_STRUCT_SIZE;
    check(api_->PJRT_Plugin_Initialize(&args), "PJRT_Plugin_Initialize");
  }
}

void PluginLibrary::close()
{
  api_ = nullptr;
  if (library_ != nullptr) dlclose(library_);
  library_ = nullptr;
}

PJRT_Extension_Base *PluginLibrary::find_extension(PJRT_Extension_Type type) const
{
  if (api_ == nullptr) return nullptr;
  PJRT_Extension_Base *starts[] = {api_->extension_start, api_->pjrt_api_version.extension_start};
  for (PJRT_Extension_Base *start : starts)
    for (PJRT_Extension_Base *extension = start; extension != nullptr; extension = extension->next)
      if (extension->type == type) return extension;
  return nullptr;
}

void PluginLibrary::check(PJRT_Error *error, const std::string &operation) const
{
  pjrt::check(api_, error, operation);
}

void register_external_ffi_handlers(const PluginLibrary &library,
                                    const std::vector<std::string> &targets)
{
  if (targets.empty()) return;

  // LAMMPS_JAX_FFI_HANDLERS: target=/path/lib.so:symbol[;...]
  std::map<std::string, std::pair<std::string, std::string>> mapping;
  if (const char *spec = std::getenv("LAMMPS_JAX_FFI_HANDLERS")) {
    const std::string text(spec);
    size_t start = 0;
    while (start < text.size()) {
      size_t end = text.find(';', start);
      if (end == std::string::npos) end = text.size();
      const std::string entry = text.substr(start, end - start);
      start = end + 1;
      if (entry.empty()) continue;
      const size_t equals = entry.find('=');
      const size_t colon = entry.rfind(':');
      if (equals == std::string::npos || colon == std::string::npos || colon < equals)
        throw std::runtime_error(
            "Malformed LAMMPS_JAX_FFI_HANDLERS entry '" + entry +
            "'; expected target=/path/lib.so:symbol");
      mapping[entry.substr(0, equals)] = {
          entry.substr(equals + 1, colon - equals - 1), entry.substr(colon + 1)};
    }
  }

  auto *extension = reinterpret_cast<PJRT_FFI_Extension *>(
      library.find_extension(PJRT_Extension_Type_FFI));
  if (extension == nullptr ||
      extension->base.struct_size < PJRT_STRUCT_SIZE(PJRT_FFI_Extension, register_handler))
    throw std::runtime_error(
        "Bundle requires external custom kernels but the PJRT plugin lacks FFI "
        "handler registration; upgrade the jax CUDA plugin");

  for (const std::string &target : targets) {
    const auto entry = mapping.find(target);
    if (entry == mapping.end())
      throw std::runtime_error(
          "Bundle requires FFI custom-call target '" + target +
          "' but LAMMPS_JAX_FFI_HANDLERS does not map it; set e.g. "
          "LAMMPS_JAX_FFI_HANDLERS='" + target + "=/path/to/handlers.so:symbol'");
    const auto &[library_path, symbol] = entry->second;
    // Handler libraries stay resident for the process lifetime.
    void *handle = dlopen(library_path.c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE);
    if (handle == nullptr)
      throw std::runtime_error("Failed to load FFI handler library '" + library_path +
                               "' for target '" + target + "': " + dlerror());
    void *handler = dlsym(handle, symbol.c_str());
    if (handler == nullptr)
      throw std::runtime_error("FFI handler symbol '" + symbol + "' not found in '" +
                               library_path + "' for target '" + target + "'");

    PJRT_FFI_Register_Handler_Args args{};
    args.struct_size = PJRT_FFI_Register_Handler_Args_STRUCT_SIZE;
    args.target_name = target.c_str();
    args.target_name_size = target.size();
    args.handler = handler;
    args.platform_name = "CUDA";
    args.platform_name_size = 4;
    args.traits = static_cast<PJRT_FFI_Handler_TraitsBits>(0);
    library.check(extension->register_handler(&args),
                  "PJRT_FFI_Register_Handler(" + target + ")");
  }
}

} // namespace pjrt
