#include "pjrt/buffer_interop.h"

#include <vector>

namespace pjrt {

namespace {

PJRT_Buffer_Type pjrt_type(ElementType type)
{
  switch (type) {
    case ElementType::Pred:
      return PJRT_Buffer_Type_PRED;
    case ElementType::S32:
      return PJRT_Buffer_Type_S32;
    case ElementType::F32:
      return PJRT_Buffer_Type_F32;
  }
  return PJRT_Buffer_Type_F32;
}

} // namespace

BufferPtr create_input_view(const PluginLibrary &library, PJRT_Client *client,
                            PJRT_Device *device, const DeviceBufferSpec &spec,
                            CUstream stream)
{
  std::vector<int64_t> minor_to_major(spec.dims.size());
  for (size_t i = 0; i < spec.dims.size(); ++i)
    minor_to_major[i] = static_cast<int64_t>(spec.dims.size() - 1 - i);

  PJRT_Buffer_MemoryLayout layout{};
  layout.struct_size = PJRT_Buffer_MemoryLayout_STRUCT_SIZE;
  layout.tiled.struct_size = PJRT_Buffer_MemoryLayout_Tiled_STRUCT_SIZE;
  layout.tiled.minor_to_major = minor_to_major.data();
  layout.tiled.minor_to_major_size = minor_to_major.size();
  layout.type = PJRT_Buffer_MemoryLayout_Type_Tiled;

  PJRT_Client_CreateViewOfDeviceBuffer_Args args{};
  args.struct_size = PJRT_Client_CreateViewOfDeviceBuffer_Args_STRUCT_SIZE;
  args.client = client;
  args.device_buffer_ptr = reinterpret_cast<void *>(spec.pointer);
  args.dims = spec.dims.data();
  args.num_dims = spec.dims.size();
  args.element_type = pjrt_type(spec.element_type);
  args.layout = &layout;
  args.device = device;
  // The engine owns the memory; the view must not free it.
  args.on_delete_callback = [](void *, void *) {};
  args.stream = reinterpret_cast<intptr_t>(stream);
  library.check(library.api()->PJRT_Client_CreateViewOfDeviceBuffer(&args),
                "PJRT_Client_CreateViewOfDeviceBuffer(" + spec.name + ")");
  return BufferPtr(args.buffer, BufferDeleter{library.api()});
}

CUdeviceptr output_pointer(const PluginLibrary &library, PJRT_Buffer *buffer)
{
  PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args args{};
  args.struct_size = PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args_STRUCT_SIZE;
  args.buffer = buffer;
  library.check(library.api()->PJRT_Buffer_OpaqueDeviceMemoryDataPointer(&args),
                "PJRT_Buffer_OpaqueDeviceMemoryDataPointer");
  return reinterpret_cast<CUdeviceptr>(args.device_memory_ptr);
}

} // namespace pjrt
