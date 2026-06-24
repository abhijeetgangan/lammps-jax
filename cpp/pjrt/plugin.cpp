#include "pjrt/plugin.h"

#include "pjrt/handles.h"

#include <dlfcn.h>

#include <cstdlib>
#include <stdexcept>

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
  library_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (library_ == nullptr)
    throw std::runtime_error("Failed to load PJRT plugin '" + path + "': " + dlerror());
  auto get_api = reinterpret_cast<GetPjrtApiFn>(dlsym(library_, "GetPjrtApi"));
  if (get_api == nullptr) throw std::runtime_error("PJRT plugin is missing GetPjrtApi");
  api_ = get_api();
  if (api_ == nullptr) throw std::runtime_error("GetPjrtApi returned null");
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

} // namespace pjrt
