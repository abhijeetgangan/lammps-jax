#include "pjrt/runtime.h"

#include <cstdlib>
#include <cstring>
#include <future>
#include <stdexcept>
#include <string>
#include <utility>
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
    case ElementType::F64:
      return PJRT_Buffer_Type_F64;
  }
  return PJRT_Buffer_Type_F32;
}

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

PJRT_Buffer_Type buffer_element_type(const PluginLibrary &library, PJRT_Buffer *buffer)
{
  PJRT_Buffer_ElementType_Args args{};
  args.struct_size = PJRT_Buffer_ElementType_Args_STRUCT_SIZE;
  args.buffer = buffer;
  library.check(library.api()->PJRT_Buffer_ElementType(&args), "PJRT_Buffer_ElementType");
  return args.type;
}

size_t executable_num_outputs(const PluginLibrary &library, PJRT_LoadedExecutable *executable)
{
  const PJRT_Api *api = library.api();
  PJRT_LoadedExecutable_GetExecutable_Args get_args{};
  get_args.struct_size = PJRT_LoadedExecutable_GetExecutable_Args_STRUCT_SIZE;
  get_args.loaded_executable = executable;
  library.check(api->PJRT_LoadedExecutable_GetExecutable(&get_args),
                "PJRT_LoadedExecutable_GetExecutable");

  PJRT_Executable_NumOutputs_Args num_args{};
  num_args.struct_size = PJRT_Executable_NumOutputs_Args_STRUCT_SIZE;
  num_args.executable = get_args.executable;
  PJRT_Error *num_error = api->PJRT_Executable_NumOutputs(&num_args);

  PJRT_Executable_Destroy_Args destroy_args{};
  destroy_args.struct_size = PJRT_Executable_Destroy_Args_STRUCT_SIZE;
  destroy_args.executable = get_args.executable;
  api->PJRT_Executable_Destroy(&destroy_args);

  library.check(num_error, "PJRT_Executable_NumOutputs");
  return num_args.num_outputs;
}

ExecutablePtr compile_program(const PluginLibrary &library, PJRT_Client *client,
                              const std::string &stablehlo,
                              const std::string &compile_options,
                              const std::string &label, size_t expected_outputs)
{
  const PJRT_Api *api = library.api();
  if (stablehlo.empty()) return ExecutablePtr(nullptr, ExecutableDeleter{api});
  if (compile_options.empty())
    throw std::runtime_error("Bundle is missing compile options for program '" + label +
                             "'; a proto-parse failure inside XLA would be the only "
                             "symptom otherwise");

  std::string format = "mlir";
  PJRT_Program program{};
  program.struct_size = PJRT_Program_STRUCT_SIZE;
  program.code = const_cast<char *>(stablehlo.data());
  program.code_size = stablehlo.size();
  program.format = const_cast<char *>(format.data());
  program.format_size = format.size();

  PJRT_Client_Compile_Args args{};
  args.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE;
  args.client = client;
  args.program = &program;
  args.compile_options = compile_options.data();
  args.compile_options_size = compile_options.size();
  library.check(api->PJRT_Client_Compile(&args), "PJRT_Client_Compile(" + label + ")");
  if (args.executable == nullptr)
    throw std::runtime_error("PJRT_Client_Compile returned a null " + label + " executable");
  ExecutablePtr executable(args.executable, ExecutableDeleter{api});

  // Bundles are external input: wrong output arity would write past the output list.
  const size_t num_outputs = executable_num_outputs(library, executable.get());
  if (num_outputs != expected_outputs)
    throw std::runtime_error("PJRT " + label + " program returns " +
                             std::to_string(num_outputs) + " outputs; expected " +
                             std::to_string(expected_outputs));
  return executable;
}

ExecutionResult execute_loaded(const PluginLibrary &library, PJRT_Client *client,
                               PJRT_Device *device, CUstream input_stream,
                               PJRT_LoadedExecutable *executable,
                               const ExecutionRequest &request, bool with_energy,
                               bool with_forces, OutputLifetime &output_lifetime,
                               PJRT_ExecuteContext *execute_context = nullptr)
{
  // Fused programs return energy and forces, single-purpose one; arity checked at compile.
  const size_t num_outputs = static_cast<size_t>(with_energy) + static_cast<size_t>(with_forces);
  const int energy_output_index = with_energy ? 0 : -1;
  const int force_output_index = with_forces ? (with_energy ? 1 : 0) : -1;
  output_lifetime.cleanup_before_execution();

  if (!executable) throw std::runtime_error("Selected PJRT executable is not initialized");

  const PJRT_Api *api = library.api();
  if (request.input_ready_event != nullptr)
    check_cuda(cuStreamWaitEvent(input_stream, request.input_ready_event, 0),
               "PJRT input stream wait");

  ExecutionResult result;

  std::vector<BufferPtr> input_guards;
  input_guards.reserve(request.inputs.size());
  for (const auto &input : request.inputs)
    input_guards.emplace_back(create_input_view(library, client, device, input, input_stream));

  std::vector<PJRT_Buffer *> input_buffers;
  input_buffers.reserve(input_guards.size());
  for (auto &input : input_guards) input_buffers.push_back(input.get());

  std::vector<PJRT_Buffer *> output_buffers(num_outputs, nullptr);
  PJRT_Buffer **argument_lists[1] = {input_buffers.data()};
  PJRT_Buffer **output_lists[1] = {output_buffers.data()};
  PJRT_Event *complete_event = nullptr;

  // Inputs are views of engine-owned device memory; never donate them.
  std::vector<int64_t> non_donatable(input_buffers.size());
  for (size_t i = 0; i < input_buffers.size(); ++i) non_donatable[i] = static_cast<int64_t>(i);

  PJRT_ExecuteOptions options{};
  options.struct_size = PJRT_ExecuteOptions_STRUCT_SIZE;
  options.non_donatable_input_indices = non_donatable.data();
  options.num_non_donatable_input_indices = non_donatable.size();
  // Carries the model-comm user data to communicating programs' FFI calls.
  options.context = execute_context;

  PJRT_LoadedExecutable_Execute_Args execute_args{};
  execute_args.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
  execute_args.executable = executable;
  execute_args.options = &options;
  execute_args.argument_lists = argument_lists;
  execute_args.num_devices = 1;
  execute_args.num_args = input_buffers.size();
  execute_args.output_lists = output_lists;
  execute_args.device_complete_events = &complete_event;
  execute_args.execute_device = device;
  library.check(api->PJRT_LoadedExecutable_Execute(&execute_args),
                "PJRT_LoadedExecutable_Execute");

  // Adopt outputs before the await: on the GPU client execution failures
  // arrive through the completion event after the buffers already exist.
  std::vector<BufferPtr> outputs;
  outputs.reserve(num_outputs);
  for (PJRT_Buffer *buffer : output_buffers) outputs.emplace_back(buffer, BufferDeleter{api});

  EventPtr complete(complete_event, EventDeleter{api});
  if (complete) {
    PJRT_Event_Await_Args await_args{};
    await_args.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
    await_args.event = complete.get();
    library.check(api->PJRT_Event_Await(&await_args), "PJRT_Event_Await");
  }

  for (const BufferPtr &output : outputs)
    if (!output) throw std::runtime_error("PJRT returned a null output buffer");

  // Outputs read back at contract precision; width mismatches fail loudly here.
  const bool f64 = request.scalar_type == ElementType::F64;
  const PJRT_Buffer_Type expected_type = f64 ? PJRT_Buffer_Type_F64 : PJRT_Buffer_Type_F32;
  for (const auto &output : outputs) {
    const PJRT_Buffer_Type actual_type = buffer_element_type(library, output.get());
    if (actual_type != expected_type)
      throw std::runtime_error(
          "PJRT program output element type " + std::to_string(actual_type) +
          " does not match the bundle precision (expected PJRT_Buffer_Type " +
          std::to_string(expected_type) + (f64 ? ", float64)" : ", float32)"));
  }

  if (energy_output_index >= 0) {
    const CUdeviceptr energy_pointer =
        output_pointer(library, outputs[energy_output_index].get());
    if (f64) {
      double energy = 0.0;
      check_cuda(cuMemcpyDtoHAsync(&energy, energy_pointer, sizeof(double), request.stream),
                 "copy energy");
      check_cuda(cuStreamSynchronize(request.stream), "synchronize energy copy");
      result.energy = energy;
    } else {
      float energy = 0.0f;
      check_cuda(cuMemcpyDtoHAsync(&energy, energy_pointer, sizeof(float), request.stream),
                 "copy energy");
      check_cuda(cuStreamSynchronize(request.stream), "synchronize energy copy");
      result.energy = static_cast<double>(energy);
    }
  }

  if (force_output_index >= 0) {
    const CUdeviceptr force_pointer = output_pointer(library, outputs[force_output_index].get());
    output_lifetime.retain_force_output(std::move(outputs[force_output_index]), force_pointer);
  }
  return result;
}

} // namespace

Runtime::~Runtime()
{
  try {
    close();
  } catch (...) {
  }
}

void Runtime::close()
{
  stop_worker();
  if (model_comm_) {
    model_comm_->close(library_.api());
    model_comm_.reset();
  }
  output_lifetime_.reset();
  force_executable_.reset();
  energy_executable_.reset();
  energy_force_executable_.reset();
  session_.close(library_);
  library_.close();
}

void Runtime::initialize(const std::string &plugin_path, const std::string &force_mlir,
                         const std::string &energy_mlir, const std::string &energy_and_forces_mlir,
                         const std::string &compile_options, const ClientOptions &client_options,
                         const CommConfig &comm_config,
                         const std::vector<std::string> &custom_call_targets)
{
  close();
  // Command buffers replay programs as CUDA graphs, worth 6%; prepended so user flags still win.
  {
    const std::string defaults =
        "--xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUDNN "
        "--xla_gpu_graph_min_graph_size=1";
    static bool defaults_applied = false;
    const char *existing = std::getenv("XLA_FLAGS");
    if (!defaults_applied) {
      defaults_applied = true;
      if (existing == nullptr || existing[0] == '\0')
        setenv("XLA_FLAGS", defaults.c_str(), 1);
      // A value not starting with '-' names an XLA flag file; leave it alone.
      else if (existing[std::strspn(existing, " \t\r\n")] == '-')
        setenv("XLA_FLAGS", (defaults + " " + existing).c_str(), 1);
    }
  }
  library_.open(plugin_path);
  session_.initialize(library_, client_options);
  // Handler registration must precede compiling programs that reference those targets.
  register_external_ffi_handlers(library_, custom_call_targets);
  if (!comm_config.widths.empty()) {
    model_comm_ = std::make_unique<ModelComm>();
    model_comm_->initialize(library_.api(), comm_config.max_atoms, comm_config.widths);
    if (comm_config.callback) model_comm_->set_service_callback(comm_config.callback);
  }
  force_executable_ =
      compile_program(library_, session_.client(), force_mlir, compile_options, "force", 1);
  energy_executable_ =
      compile_program(library_, session_.client(), energy_mlir, compile_options, "energy", 1);
  energy_force_executable_ = compile_program(library_, session_.client(),
                                             energy_and_forces_mlir, compile_options,
                                             "energy+forces", 2);
}

ExecutionResult Runtime::run_with_comm(const ExecutionRequest &request,
                                       const std::function<ExecutionResult()> &execute)
{
  if (!model_comm_) return execute();
  model_comm_->begin_step(request.nlocal, request.nghost);
  // Handlers block until this MPI thread services them; execution runs on a worker meanwhile.
  model_comm_->begin_service();
  ModelComm *model_comm = model_comm_.get();
  // The worker starts without a CUDA context; adopt the caller's for the driver-API calls.
  CUcontext cuda_context = nullptr;
  cuCtxGetCurrent(&cuda_context);
  std::packaged_task<ExecutionResult()> task([model_comm, &execute, cuda_context]() {
    // Guard first: a throw before mark_execution_done wedges service_loop.
    struct DoneGuard {
      ModelComm *owner;
      ~DoneGuard() { owner->mark_execution_done(); }
    } guard{model_comm};
    if (cuda_context != nullptr)
      check_cuda(cuCtxSetCurrent(cuda_context), "adopt CUDA context on model comm worker");
    return execute();
  });
  auto future = task.get_future();
  {
    std::lock_guard<std::mutex> lock(worker_mutex_);
    if (!worker_.joinable()) worker_ = std::thread([this] { worker_loop(); });
    worker_task_ = std::move(task);
    worker_has_task_ = true;
  }
  worker_cv_.notify_one();
  model_comm_->service_loop();
  return future.get();
}

void Runtime::worker_loop()
{
  for (;;) {
    std::packaged_task<ExecutionResult()> task;
    {
      std::unique_lock<std::mutex> lock(worker_mutex_);
      worker_cv_.wait(lock, [this] { return worker_has_task_ || worker_stop_; });
      if (worker_stop_) return;
      task = std::move(worker_task_);
      worker_has_task_ = false;
    }
    task();
  }
}

void Runtime::stop_worker()
{
  {
    std::lock_guard<std::mutex> lock(worker_mutex_);
    worker_stop_ = true;
  }
  worker_cv_.notify_one();
  if (worker_.joinable()) worker_.join();
  worker_stop_ = false;
}

ExecutionResult Runtime::execute_force(const ExecutionRequest &request)
{
  return run_with_comm(request, [&] {
    return execute_loaded(library_, session_.client(), session_.device(),
                          session_.input_stream_for(library_), force_executable_.get(),
                          request, false, true, output_lifetime_,
                          model_comm_ ? model_comm_->execute_context() : nullptr);
  });
}

ExecutionResult Runtime::execute_energy(const ExecutionRequest &request)
{
  return run_with_comm(request, [&] {
    return execute_loaded(library_, session_.client(), session_.device(),
                          session_.input_stream_for(library_), energy_executable_.get(),
                          request, true, false, output_lifetime_,
                          model_comm_ ? model_comm_->execute_context() : nullptr);
  });
}

ExecutionResult Runtime::execute_energy_force(const ExecutionRequest &request)
{
  return run_with_comm(request, [&] {
    return execute_loaded(library_, session_.client(), session_.device(),
                          session_.input_stream_for(library_), energy_force_executable_.get(),
                          request, true, true, output_lifetime_,
                          model_comm_ ? model_comm_->execute_context() : nullptr);
  });
}

void Runtime::consume_force_output(CUstream consumer_stream,
                                   const std::function<void(CUdeviceptr)> &consumer)
{
  output_lifetime_.consume_force_output(consumer_stream, consumer);
}

} // namespace pjrt
