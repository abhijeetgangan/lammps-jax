#include "pjrt/runtime.h"

#include "pjrt/loaded_executable.h"

namespace pjrt {

Runtime::~Runtime()
{
  try {
    close();
  } catch (...) {
  }
}

void Runtime::close()
{
  output_lifetime_.reset();
  force_executable_.reset();
  energy_executable_.reset();
  energy_force_executable_.reset();
  session_.close(library_);
  library_.close();
}

void Runtime::initialize(const std::string &plugin_path, const ProgramSet &programs,
                         const std::string &compile_options, const ClientOptions &client_options)
{
  close();
  library_.open(plugin_path);
  session_.initialize(library_, client_options);
  force_executable_ =
      compile_program(library_, session_.client(), programs.force_mlir, compile_options, "force", 1);
  energy_executable_ =
      compile_program(library_, session_.client(), programs.energy_mlir, compile_options, "energy", 1);
  energy_force_executable_ = compile_program(library_, session_.client(),
                                             programs.energy_and_forces_mlir, compile_options,
                                             "energy+forces", 2);
}

ExecutionResult Runtime::execute_force(const ExecutionRequest &request)
{
  return execute_loaded(library_, session_.client(), session_.device(),
                        session_.input_stream_for(library_), force_executable_.get(),
                        request, false, true, output_lifetime_);
}

ExecutionResult Runtime::execute_energy(const ExecutionRequest &request)
{
  return execute_loaded(library_, session_.client(), session_.device(),
                        session_.input_stream_for(library_), energy_executable_.get(),
                        request, true, false, output_lifetime_);
}

ExecutionResult Runtime::execute_energy_force(const ExecutionRequest &request)
{
  return execute_loaded(library_, session_.client(), session_.device(),
                        session_.input_stream_for(library_), energy_force_executable_.get(),
                        request, true, true, output_lifetime_);
}

void Runtime::consume_force_output(CUstream consumer_stream,
                                   const std::function<void(CUdeviceptr)> &consumer)
{
  output_lifetime_.consume_force_output(consumer_stream, consumer);
}

} // namespace pjrt
