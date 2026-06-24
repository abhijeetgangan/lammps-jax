#include "pjrt/loaded_executable.h"

#include "pjrt/buffer_interop.h"

#include <chrono>
#include <stdexcept>
#include <utility>

namespace pjrt {

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start, Clock::time_point stop)
{
  return std::chrono::duration<double, std::milli>(stop - start).count();
}

ExecutablePtr compile_mlir(const PluginLibrary &library, PJRT_Client *client,
                           const std::string &stablehlo, const std::string &compile_options,
                           const std::string &label)
{
  const PJRT_Api *api = library.api();
  if (stablehlo.empty()) return ExecutablePtr(nullptr, ExecutableDeleter{api});

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
  return ExecutablePtr(args.executable, ExecutableDeleter{api});
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

double copy_energy(CUstream stream, CUdeviceptr pointer)
{
  float energy = 0.0f;
  check_cuda(cuMemcpyDtoHAsync(&energy, pointer, sizeof(float), stream), "copy energy");
  check_cuda(cuStreamSynchronize(stream), "synchronize energy copy");
  return static_cast<double>(energy);
}

} // namespace

ExecutablePtr compile_program(const PluginLibrary &library, PJRT_Client *client,
                              const std::string &stablehlo,
                              const std::string &compile_options,
                              const std::string &label, size_t expected_outputs)
{
  ExecutablePtr executable = compile_mlir(library, client, stablehlo, compile_options, label);
  if (executable) {
    // Bundles are external input: a program whose output arity disagrees with
    // the caller-sized output list would make PJRT write past the end of it.
    const size_t num_outputs = executable_num_outputs(library, executable.get());
    if (num_outputs != expected_outputs)
      throw std::runtime_error("PJRT " + label + " program returns " +
                               std::to_string(num_outputs) + " outputs; expected " +
                               std::to_string(expected_outputs));
  }
  return executable;
}

ExecutionResult execute_loaded(const PluginLibrary &library, PJRT_Client *client,
                               PJRT_Device *device, CUstream input_stream,
                               PJRT_LoadedExecutable *executable,
                               const ExecutionRequest &request, bool with_energy,
                               bool with_forces, OutputLifetime &output_lifetime)
{
  // Fused programs return energy and forces; single-purpose programs return
  // one value. Arity was validated against the program at compile time.
  const size_t num_outputs = static_cast<size_t>(with_energy) + static_cast<size_t>(with_forces);
  const int energy_output_index = with_energy ? 0 : -1;
  const int force_output_index = with_forces ? (with_energy ? 1 : 0) : -1;
  const auto total_start = Clock::now();
  output_lifetime.cleanup_before_execution();

  if (!executable) throw std::runtime_error("Selected PJRT executable is not initialized");

  const PJRT_Api *api = library.api();
  if (request.input_ready_event != nullptr)
    check_cuda(cuStreamWaitEvent(input_stream, request.input_ready_event, 0),
               "PJRT input stream wait");

  ExecutionResult result;

  const auto view_start = Clock::now();
  std::vector<BufferPtr> input_guards;
  input_guards.reserve(request.inputs.size());
  for (const auto &input : request.inputs)
    input_guards.emplace_back(create_input_view(library, client, device, input, input_stream));
  const auto view_stop = Clock::now();

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
  const auto execute_start = Clock::now();
  library.check(api->PJRT_LoadedExecutable_Execute(&execute_args),
                "PJRT_LoadedExecutable_Execute");
  const auto execute_stop = Clock::now();

  EventPtr complete(complete_event, EventDeleter{api});
  const auto await_start = Clock::now();
  if (complete) {
    PJRT_Event_Await_Args await_args{};
    await_args.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
    await_args.event = complete.get();
    library.check(api->PJRT_Event_Await(&await_args), "PJRT_Event_Await");
  }
  const auto await_stop = Clock::now();

  const auto output_start = Clock::now();
  std::vector<BufferPtr> outputs;
  outputs.reserve(num_outputs);
  for (PJRT_Buffer *buffer : output_buffers) {
    if (!buffer) throw std::runtime_error("PJRT returned a null output buffer");
    outputs.emplace_back(buffer, BufferDeleter{api});
  }

  if (energy_output_index >= 0)
    result.energy = copy_energy(request.stream, output_pointer(library, outputs[energy_output_index].get()));

  if (force_output_index >= 0) {
    const CUdeviceptr force_pointer = output_pointer(library, outputs[force_output_index].get());
    output_lifetime.retain_force_output(std::move(outputs[force_output_index]), force_pointer);
  }
  const auto total_stop = Clock::now();
  result.pjrt_view_ms = elapsed_ms(view_start, view_stop);
  result.pjrt_execute_ms = elapsed_ms(execute_start, execute_stop);
  result.pjrt_await_ms = elapsed_ms(await_start, await_stop);
  result.pjrt_output_ms = elapsed_ms(output_start, total_stop);
  result.pjrt_total_ms = elapsed_ms(total_start, total_stop);
  return result;
}

} // namespace pjrt
