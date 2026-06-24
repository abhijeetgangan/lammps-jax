#include "pjrt/client_session.h"

#include <cstring>
#include <stdexcept>
#include <vector>

namespace pjrt {

namespace {

PJRT_NamedValue named_option(const char *name, PJRT_NamedValue_Type type)
{
  PJRT_NamedValue option{};
  option.struct_size = PJRT_NamedValue_STRUCT_SIZE;
  option.name = name;
  option.name_size = std::strlen(name);
  option.type = type;
  option.value_size = 1;
  return option;
}

} // namespace

void ClientSession::initialize(const PluginLibrary &library, const ClientOptions &options)
{
  close(library);

  const PJRT_Api *api = library.api();
  if (api == nullptr) throw std::runtime_error("PJRT plugin is not loaded");

  // Restrict the client to the engine's device and keep BFC preallocation
  // off: every MPI rank creates its own client, and the plugin's defaults
  // The default behavior uses all visible devices and preallocates most free
  // memory on each, which can starve the host engine and other ranks.
  const int64_t visible_device = options.visible_device;
  std::vector<PJRT_NamedValue> create_options;
  PJRT_NamedValue visible = named_option("visible_devices", PJRT_NamedValue_kInt64List);
  visible.int64_array_value = &visible_device;
  create_options.push_back(visible);
  PJRT_NamedValue preallocate = named_option("preallocate", PJRT_NamedValue_kBool);
  preallocate.bool_value = false;
  create_options.push_back(preallocate);
  PJRT_NamedValue memory_fraction = named_option("memory_fraction", PJRT_NamedValue_kFloat);
  if (options.memory_fraction > 0.0) {
    memory_fraction.float_value = static_cast<float>(options.memory_fraction);
    create_options.push_back(memory_fraction);
  }

  PJRT_Client_Create_Args args{};
  args.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
  args.create_options = create_options.data();
  args.num_options = create_options.size();
  library.check(api->PJRT_Client_Create(&args), "PJRT_Client_Create");
  client_ = args.client;
  if (client_ == nullptr) throw std::runtime_error("PJRT_Client_Create returned null");

  PJRT_Client_AddressableDevices_Args device_args{};
  device_args.struct_size = PJRT_Client_AddressableDevices_Args_STRUCT_SIZE;
  device_args.client = client_;
  library.check(api->PJRT_Client_AddressableDevices(&device_args),
                "PJRT_Client_AddressableDevices");
  if (device_args.num_addressable_devices == 0)
    throw std::runtime_error("PJRT client has no addressable devices");
  device_ = device_args.addressable_devices[0];

  stream_extension_ = reinterpret_cast<PJRT_Stream_Extension_Local *>(
      library.find_extension(PJRT_Extension_Type_Stream));
  if (stream_extension_ == nullptr || stream_extension_->get_stream == nullptr)
    throw std::runtime_error("PJRT CUDA stream extension is required for device stream handoff");
}

void ClientSession::close(const PluginLibrary &library)
{
  stream_extension_ = nullptr;
  device_ = nullptr;
  if (client_ != nullptr && library.api() != nullptr) {
    PJRT_Client_Destroy_Args args{};
    args.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
    args.client = client_;
    library.check(library.api()->PJRT_Client_Destroy(&args), "PJRT_Client_Destroy");
  }
  client_ = nullptr;
}

CUstream ClientSession::input_stream_for(const PluginLibrary &library) const
{
  if (stream_extension_ == nullptr || device_ == nullptr)
    throw std::runtime_error("PJRT client session is not initialized");
  PJRT_Get_Stream_For_External_Ready_Events_Args_Local args{};
  args.struct_size = sizeof(PJRT_Get_Stream_For_External_Ready_Events_Args_Local);
  args.device = device_;
  library.check(stream_extension_->get_stream(&args),
                "PJRT_Get_Stream_For_External_Ready_Events");
  return reinterpret_cast<CUstream>(args.stream);
}

} // namespace pjrt
