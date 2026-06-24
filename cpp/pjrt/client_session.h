#ifndef PJRT_CLIENT_SESSION_H
#define PJRT_CLIENT_SESSION_H

#include "pjrt/plugin.h"

#include <cuda.h>

namespace pjrt {

// The client is restricted to a single CUDA device so executables compiled
// with the default client device assignment run on the engine's GPU. BFC
// preallocation is disabled so per-rank clients do not reserve most of every
// visible device.
struct ClientOptions {
  int visible_device = 0;        // CUDA device ordinal this client may use
  double memory_fraction = 0.0;  // 0 keeps the plugin default
};

// Owns the PJRT client and the single CUDA device visible to this LAMMPS rank.
class ClientSession {
 public:
  ClientSession() = default;
  ClientSession(const ClientSession &) = delete;
  ClientSession &operator=(const ClientSession &) = delete;

  void initialize(const PluginLibrary &library, const ClientOptions &options);
  void close(const PluginLibrary &library);

  PJRT_Client *client() const { return client_; }
  PJRT_Device *device() const { return device_; }
  CUstream input_stream_for(const PluginLibrary &library) const;

 private:
  PJRT_Client *client_ = nullptr;
  PJRT_Device *device_ = nullptr;
  PJRT_Stream_Extension_Local *stream_extension_ = nullptr;
};

} // namespace pjrt

#endif
