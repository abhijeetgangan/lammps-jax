#ifndef PJRT_PLUGIN_H
#define PJRT_PLUGIN_H

#include "pjrt/api.h"

#include <string>

namespace pjrt {

// Resolution order: explicit path, then the named environment variable, then
// the bare plugin soname through the loader search path.
std::string resolve_plugin_path(const std::string &explicit_path, const char *env_var);

class PluginLibrary {
 public:
  PluginLibrary() = default;
  PluginLibrary(const PluginLibrary &) = delete;
  PluginLibrary &operator=(const PluginLibrary &) = delete;
  ~PluginLibrary();

  void open(const std::string &path);
  void close();
  const PJRT_Api *api() const { return api_; }

  PJRT_Extension_Base *find_extension(PJRT_Extension_Type type) const;

  // Throws with the extracted PJRT error message when error is non-null.
  void check(PJRT_Error *error, const std::string &operation) const;

 private:
  void *library_ = nullptr;
  const PJRT_Api *api_ = nullptr;
};

} // namespace pjrt

#endif
