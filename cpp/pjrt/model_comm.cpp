#include "pjrt/model_comm.h"

#include "pjrt/api.h"
#include "pjrt/handles.h"

#if __has_include("xla/pjrt/c/pjrt_c_api_ffi_extension.h")
#include "xla/pjrt/c/pjrt_c_api_ffi_extension.h"
#else
#include "third_party/pjrt_c_api_ffi_extension.h"
#endif

// Header-only XLA FFI API from the jaxlib wheel; the per-call XLA_FFI_Api table avoids XLA linkage.
#include "xla/ffi/api/ffi.h"

#include <algorithm>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace ffi = xla::ffi;

namespace pjrt {

namespace {

const char kForwardTarget[] = "lammps_jax.forward_comm";
const char kReverseTarget[] = "lammps_jax.reverse_comm";
const char kUserDataTypeName[] = "lammps_jax.ModelCommUserData";

const PJRT_FFI_Extension *find_ffi_extension(const PJRT_Api *api)
{
  for (PJRT_Extension_Base *extension = api->extension_start; extension != nullptr;
       extension = extension->next) {
    if (extension->type == PJRT_Extension_Type_FFI)
      return reinterpret_cast<const PJRT_FFI_Extension *>(extension);
  }
  return nullptr;
}

std::string describe_buffer_error(const char *what, int64_t got, int64_t expected)
{
  return std::string("model comm ") + what + ": got " + std::to_string(got) +
         ", expected " + std::to_string(expected);
}

} // namespace

// UserData decoding needs a static TypeId, assigned once per process by the plugin registry.
struct ModelCommUserData {
  static XLA_FFI_TypeId id;
  ModelComm *exchange = nullptr;
};

XLA_FFI_TypeId ModelCommUserData::id = {0};

namespace {

ffi::Error model_comm_impl(bool forward, ffi::AnyBuffer features, ffi::AnyBuffer token,
                              ffi::Result<ffi::AnyBuffer> features_out,
                              ffi::Result<ffi::AnyBuffer> token_out, CUstream stream,
                              ModelCommUserData *user_data)
{
  if (user_data == nullptr || user_data->exchange == nullptr)
    return ffi::Error::Internal("model comm user data is not attached");
  if (features.element_type() != ffi::DataType::F32)
    return ffi::Error::InvalidArgument("model comm features must be float32");
  if (features.dimensions().size() != 2)
    return ffi::Error::InvalidArgument("model comm features must be rank 2");
  const int64_t rows = features.dimensions()[0];
  const int64_t width = features.dimensions()[1];
  if (features_out->dimensions().size() != 2 || features_out->dimensions()[0] != rows ||
      features_out->dimensions()[1] != width)
    return ffi::Error::InvalidArgument("model comm output shape differs from input");

  // Exceptions must not cross the FFI boundary.
  std::string error;
  try {
    error = user_data->exchange->comm_from_handler(
        forward, stream, features.untyped_data(), features_out->untyped_data(), rows, width,
        token.untyped_data(), token_out->untyped_data());
  } catch (const std::exception &exception) {
    error = exception.what();
  } catch (...) {
    error = "model comm failed";
  }
  if (!error.empty()) return ffi::Error::Internal(error);
  return ffi::Error::Success();
}

ffi::Error forward_comm_impl(ffi::AnyBuffer features, ffi::AnyBuffer token,
                             ffi::Result<ffi::AnyBuffer> features_out,
                             ffi::Result<ffi::AnyBuffer> token_out, CUstream stream,
                             ModelCommUserData *user_data)
{
  return model_comm_impl(true, features, token, features_out, token_out, stream, user_data);
}

ffi::Error reverse_comm_impl(ffi::AnyBuffer features, ffi::AnyBuffer token,
                             ffi::Result<ffi::AnyBuffer> features_out,
                             ffi::Result<ffi::AnyBuffer> token_out, CUstream stream,
                             ModelCommUserData *user_data)
{
  return model_comm_impl(false, features, token, features_out, token_out, stream, user_data);
}

XLA_FFI_DEFINE_HANDLER(kForwardCommHandler, forward_comm_impl,
                       ffi::Ffi::Bind()
                           .Arg<ffi::AnyBuffer>()
                           .Arg<ffi::AnyBuffer>()
                           .Ret<ffi::AnyBuffer>()
                           .Ret<ffi::AnyBuffer>()
                           .Ctx<ffi::PlatformStream<CUstream>>()
                           .Ctx<ffi::UserData<ModelCommUserData>>());

XLA_FFI_DEFINE_HANDLER(kReverseCommHandler, reverse_comm_impl,
                       ffi::Ffi::Bind()
                           .Arg<ffi::AnyBuffer>()
                           .Arg<ffi::AnyBuffer>()
                           .Ret<ffi::AnyBuffer>()
                           .Ret<ffi::AnyBuffer>()
                           .Ctx<ffi::PlatformStream<CUstream>>()
                           .Ctx<ffi::UserData<ModelCommUserData>>());

} // namespace

ModelComm::~ModelComm()
{
  try {
    close(api_);
  } catch (...) {
  }
}

void ModelComm::initialize(const PJRT_Api *api, int max_atoms, const std::vector<int> &widths)
{
  if (widths.empty()) throw std::runtime_error("ModelComm requires a non-empty width schedule");
  const PJRT_FFI_Extension *extension = find_ffi_extension(api);
  // The struct_size check guards register_handler, appended to the extension
  // after its introduction; an older plugin's extension ends before it.
  if (extension == nullptr ||
      extension->base.struct_size < PJRT_STRUCT_SIZE(PJRT_FFI_Extension, register_handler))
    throw std::runtime_error(
        "PJRT plugin does not expose the FFI handler registration required by "
        "communicating bundles; use a newer jax CUDA plugin or a non-communicating bundle");

  // Execute-context support is late C API; an older plugin's struct ends before it.
  if (api->struct_size < PJRT_STRUCT_SIZE(PJRT_Api, PJRT_ExecuteContext_Destroy))
    throw std::runtime_error(
        "PJRT plugin is too old for communicating bundles (no execute-context "
        "support); upgrade the jax CUDA plugin");

  api_ = api;
  max_atoms_ = max_atoms;
  widths_ = widths;

  // Registration lives in the plugin's process-global registry: exactly once per process.
  static std::once_flag registered;
  std::call_once(registered, [&] {
    static PJRT_FFI_Type_Info type_info = {nullptr, nullptr, nullptr};
    PJRT_FFI_Type_Register_Args type_args{};
    type_args.struct_size = PJRT_FFI_Type_Register_Args_STRUCT_SIZE;
    type_args.type_name = kUserDataTypeName;
    type_args.type_name_size = sizeof(kUserDataTypeName) - 1;
    type_args.type_id = 0; // plugin assigns
    type_args.type_info = &type_info;
    if (PJRT_Error *error = extension->type_register(&type_args)) {
      PJRT_Error_Destroy_Args destroy{PJRT_Error_Destroy_Args_STRUCT_SIZE, nullptr, error};
      api->PJRT_Error_Destroy(&destroy);
      throw std::runtime_error("PJRT_FFI_Type_Register failed for the model comm user data type");
    }
    ModelCommUserData::id.type_id = type_args.type_id;

    const struct {
      const char *name;
      size_t name_size;
      void *handler;
    } handlers[] = {
        {kForwardTarget, sizeof(kForwardTarget) - 1,
         reinterpret_cast<void *>(kForwardCommHandler)},
        {kReverseTarget, sizeof(kReverseTarget) - 1,
         reinterpret_cast<void *>(kReverseCommHandler)},
    };
    for (const auto &entry : handlers) {
      PJRT_FFI_Register_Handler_Args handler_args{};
      handler_args.struct_size = PJRT_FFI_Register_Handler_Args_STRUCT_SIZE;
      handler_args.target_name = entry.name;
      handler_args.target_name_size = entry.name_size;
      handler_args.handler = entry.handler;
      handler_args.platform_name = "CUDA";
      handler_args.platform_name_size = 4;
      handler_args.traits = static_cast<PJRT_FFI_Handler_TraitsBits>(0);
      if (PJRT_Error *error = extension->register_handler(&handler_args)) {
        PJRT_Error_Destroy_Args destroy{PJRT_Error_Destroy_Args_STRUCT_SIZE, nullptr, error};
        api->PJRT_Error_Destroy(&destroy);
        throw std::runtime_error(std::string("PJRT_FFI_Register_Handler failed for ") +
                                 entry.name);
      }
    }
  });
  if (ModelCommUserData::id.type_id == 0)
    throw std::runtime_error("model comm user data type was not registered");

  user_data_ = new ModelCommUserData{};
  user_data_->exchange = this;

  PJRT_ExecuteContext_Create_Args context_args{};
  context_args.struct_size = PJRT_ExecuteContext_Create_Args_STRUCT_SIZE;
  if (PJRT_Error *error = api->PJRT_ExecuteContext_Create(&context_args)) {
    PJRT_Error_Destroy_Args destroy{PJRT_Error_Destroy_Args_STRUCT_SIZE, nullptr, error};
    api->PJRT_Error_Destroy(&destroy);
    throw std::runtime_error("PJRT_ExecuteContext_Create failed");
  }
  execute_context_ = context_args.context;

  PJRT_FFI_UserData_Add_Args add_args{};
  add_args.struct_size = PJRT_FFI_UserData_Add_Args_STRUCT_SIZE;
  add_args.context = execute_context_;
  add_args.user_data.type_id = ModelCommUserData::id.type_id;
  add_args.user_data.data = user_data_;
  if (PJRT_Error *error = extension->user_data_add(&add_args)) {
    PJRT_Error_Destroy_Args destroy{PJRT_Error_Destroy_Args_STRUCT_SIZE, nullptr, error};
    api->PJRT_Error_Destroy(&destroy);
    throw std::runtime_error("PJRT_FFI_UserData_Add failed");
  }

  const int max_width = *std::max_element(widths_.begin(), widths_.end());
  check_cuda(cuMemHostAlloc(reinterpret_cast<void **>(&pinned_),
                            static_cast<size_t>(max_atoms_) * max_width * sizeof(float), 0),
             "allocate pinned model comm staging");
  check_cuda(cuEventCreate(&staged_event_, CU_EVENT_DISABLE_TIMING),
             "create model comm staging event");
}

void ModelComm::close(const PJRT_Api *api)
{
  if (execute_context_ != nullptr && api != nullptr) {
    PJRT_ExecuteContext_Destroy_Args destroy_args{};
    destroy_args.struct_size = PJRT_ExecuteContext_Destroy_Args_STRUCT_SIZE;
    destroy_args.context = execute_context_;
    if (PJRT_Error *error = api->PJRT_ExecuteContext_Destroy(&destroy_args)) {
      PJRT_Error_Destroy_Args destroy{PJRT_Error_Destroy_Args_STRUCT_SIZE, nullptr, error};
      api->PJRT_Error_Destroy(&destroy);
    }
  }
  execute_context_ = nullptr;
  delete user_data_;
  user_data_ = nullptr;
  if (pinned_ != nullptr) {
    cuMemFreeHost(pinned_);
    pinned_ = nullptr;
  }
  if (staged_event_ != nullptr) {
    cuEventDestroy(staged_event_);
    staged_event_ = nullptr;
  }
  api_ = nullptr;
}

void ModelComm::set_service_callback(ServiceCallback callback)
{
  std::lock_guard<std::mutex> lock(mutex_);
  service_callback_ = std::move(callback);
}

void ModelComm::begin_step(int nlocal, int nghost)
{
  std::lock_guard<std::mutex> lock(mutex_);
  nlocal_ = nlocal;
  nghost_ = nghost;
}

void ModelComm::begin_service()
{
  std::lock_guard<std::mutex> lock(mutex_);
  servicing_ = true;
  done_ = false;
  request_pending_ = false;
  service_error_.clear();
  forward_site_ = 0;
  reverse_site_ = 0;
}

void ModelComm::mark_execution_done()
{
  std::lock_guard<std::mutex> lock(mutex_);
  done_ = true;
  condition_.notify_all();
}

void ModelComm::service_loop()
{
  std::unique_lock<std::mutex> lock(mutex_);
  for (;;) {
    condition_.wait(lock, [&] { return request_pending_ || done_; });
    if (request_pending_) {
      const ModelCommRequest request = request_;
      ServiceCallback callback = service_callback_;
      // The callback does MPI and touches staging; run without the lock for fair handler waits.
      lock.unlock();
      std::string error;
      if (!callback) {
        error = "no model comm service callback is registered";
      } else {
        try {
          callback(request);
        } catch (const std::exception &exception) {
          error = exception.what();
        } catch (...) {
          error = "model comm service callback failed";
        }
      }
      lock.lock();
      service_error_ = error;
      request_pending_ = false;
      condition_.notify_all();
    }
    if (done_ && !request_pending_) break;
  }
  servicing_ = false;
}

std::string ModelComm::validate_site(bool forward, int64_t rows, int64_t width)
{
  // Runs under mutex_. Validation is rank-identical, so failures abort before any MPI.
  if (rows != max_atoms_)
    return describe_buffer_error("row count", rows, max_atoms_);
  const int total_sites = static_cast<int>(widths_.size());
  if (forward) {
    if (forward_site_ >= total_sites)
      return "model comm saw more forward sites than the bundle declares";
    if (width != widths_[forward_site_])
      return describe_buffer_error("forward width", width, widths_[forward_site_]);
    ++forward_site_;
  } else {
    if (reverse_site_ >= total_sites)
      return "model comm saw more reverse sites than the bundle declares";
    // The backward sweep visits sites in mirror order.
    const int site = total_sites - 1 - reverse_site_;
    if (width != widths_[site])
      return describe_buffer_error("reverse width", width, widths_[site]);
    ++reverse_site_;
  }
  return "";
}

std::string ModelComm::comm_from_handler(bool forward, CUstream stream, const void *input,
                                                void *output, int64_t rows, int64_t width,
                                                const void *token_input, void *token_output)
{
  int nlocal = 0;
  int nghost = 0;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!servicing_)
      return "model comm invoked outside a serviced execution";
    const std::string error = validate_site(forward, rows, width);
    if (!error.empty()) return error;
    nlocal = nlocal_;
    nghost = nghost_;
  }

  const size_t row_bytes = static_cast<size_t>(width) * sizeof(float);
  const CUdeviceptr input_ptr = reinterpret_cast<CUdeviceptr>(input);
  const CUdeviceptr output_ptr = reinterpret_cast<CUdeviceptr>(output);

  // Aliasing normally returns the same buffer; the identity copy covers distinct ones.
  if (input != output)
    check_cuda(cuMemcpyDtoDAsync(output_ptr, input_ptr, static_cast<size_t>(rows) * row_bytes,
                                 stream),
               "model comm identity copy");
  // Forward sends owned rows; reverse sends owned and ghost adjoint rows.
  const int staged_rows = forward ? nlocal : nlocal + nghost;
  if (staged_rows > 0)
    check_cuda(cuMemcpyDtoHAsync(pinned_, input_ptr, static_cast<size_t>(staged_rows) * row_bytes,
                                 stream),
               "model comm stage to host");
  check_cuda(cuEventRecord(staged_event_, stream), "model comm staging event record");
  check_cuda(cuEventSynchronize(staged_event_), "model comm staging event wait");

  {
    std::unique_lock<std::mutex> lock(mutex_);
    request_ = ModelCommRequest{forward, pinned_, static_cast<int>(width), nlocal, nghost};
    request_pending_ = true;
    condition_.notify_all();
    condition_.wait(lock, [&] { return !request_pending_; });
    if (!service_error_.empty()) return service_error_;
  }

  if (forward) {
    if (nghost > 0)
      check_cuda(cuMemcpyHtoDAsync(output_ptr + static_cast<size_t>(nlocal) * row_bytes,
                                   pinned_ + static_cast<size_t>(nlocal) * width,
                                   static_cast<size_t>(nghost) * row_bytes, stream),
                 "model comm ghost rows to device");
  } else {
    if (nlocal > 0)
      check_cuda(cuMemcpyHtoDAsync(output_ptr, pinned_,
                                   static_cast<size_t>(nlocal) * row_bytes, stream),
                 "model comm owner adjoints to device");
    // Forwarded ghost cotangents vanish locally; padding keeps the identity copy, zero when masked.
    if (nghost > 0)
      check_cuda(cuMemsetD8Async(output_ptr + static_cast<size_t>(nlocal) * row_bytes, 0,
                                 static_cast<size_t>(nghost) * row_bytes, stream),
                 "model comm ghost adjoint zero");
  }
  // Enqueued last: the token releases the next site only after the copies above.
  if (token_output != token_input)
    check_cuda(cuMemcpyDtoDAsync(reinterpret_cast<CUdeviceptr>(token_output),
                                 reinterpret_cast<CUdeviceptr>(token_input), sizeof(float),
                                 stream),
               "model comm token copy");
  return "";
}

} // namespace pjrt
