cmake_minimum_required(VERSION 3.18)

project(lammps_jax_plugin LANGUAGES CXX)

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "In-source builds are not supported")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(LAMMPS_HEADER_DIR "" CACHE PATH "Path to the LAMMPS src directory")
set(JAXLIB_INCLUDE_DIR "" CACHE PATH "Path to jaxlib/include containing xla/ffi/api/ffi.h")
set(PJRT_C_API_INCLUDE_DIR "" CACHE PATH "Optional path containing xla/pjrt/c/pjrt_c_api.h or pjrt_c_api.h")
set(KOKKOS_CONFIG_INCLUDE_DIR "" CACHE PATH "Directory containing generated KokkosCore_config.h")

if(NOT LAMMPS_HEADER_DIR)
  message(FATAL_ERROR "Set LAMMPS_HEADER_DIR to the LAMMPS src directory")
endif()
if(NOT JAXLIB_INCLUDE_DIR)
  message(FATAL_ERROR "Set JAXLIB_INCLUDE_DIR to jaxlib/include")
endif()
if(NOT EXISTS "${JAXLIB_INCLUDE_DIR}/xla/ffi/api/ffi.h")
  message(FATAL_ERROR
    "xla/ffi/api/ffi.h was not found under JAXLIB_INCLUDE_DIR; model_comm.cpp "
    "requires it and it has no vendored fallback.")
endif()

get_filename_component(LAMMPS_SOURCE_ROOT "${LAMMPS_HEADER_DIR}/.." ABSOLUTE)
set(KOKKOS_ROOT "${LAMMPS_SOURCE_ROOT}/lib/kokkos")

if(NOT EXISTS "${KOKKOS_CONFIG_INCLUDE_DIR}/KokkosCore_config.h")
  message(FATAL_ERROR
    "KokkosCore_config.h was not found. Configure/build LAMMPS with PKG_KOKKOS first "
    "and pass -DKOKKOS_CONFIG_INCLUDE_DIR=<dir-containing-KokkosCore_config.h>.")
endif()

find_path(CUDA_INCLUDE_DIR cuda.h
  HINTS "$ENV{CUDA_HOME}/include" "/usr/local/cuda/include")
find_library(CUDA_DRIVER_LIBRARY cuda
  HINTS "$ENV{CUDA_HOME}/lib64" "$ENV{CUDA_HOME}/targets/x86_64-linux/lib"
        "/usr/lib/x86_64-linux-gnu" "/usr/lib64")
find_package(MPI REQUIRED COMPONENTS CXX)

if(NOT CUDA_INCLUDE_DIR OR NOT CUDA_DRIVER_LIBRARY)
  message(FATAL_ERROR "CUDA driver headers/library are required for lammps-jax")
endif()

set(PJRT_C_API_HEADER_FOUND FALSE)
if(EXISTS "${JAXLIB_INCLUDE_DIR}/xla/pjrt/c/pjrt_c_api.h")
  set(PJRT_C_API_HEADER_FOUND TRUE)
endif()
if(PJRT_C_API_INCLUDE_DIR)
  if(EXISTS "${PJRT_C_API_INCLUDE_DIR}/xla/pjrt/c/pjrt_c_api.h"
     OR EXISTS "${PJRT_C_API_INCLUDE_DIR}/pjrt_c_api.h")
    set(PJRT_C_API_HEADER_FOUND TRUE)
  else()
    message(FATAL_ERROR
      "PJRT C API header was not found under PJRT_C_API_INCLUDE_DIR. Expected "
      "xla/pjrt/c/pjrt_c_api.h or pjrt_c_api.h.")
  endif()
endif()
if(NOT PJRT_C_API_HEADER_FOUND)
  message(STATUS
    "PJRT C API header not found under JAXLIB_INCLUDE_DIR or "
    "PJRT_C_API_INCLUDE_DIR; using the vendored copy in cpp/third_party.")
endif()

add_library(lammps_jax MODULE
  lammps_jax_model.cpp
  lammps_jax_plugin.cpp
  pair_jax_kokkos.cpp
  pjrt/client_session.cpp
  pjrt/model_comm.cpp
  pjrt/handles.cpp
  pjrt/plugin.cpp
  pjrt/output_lifetime.cpp
  pjrt/runtime.cpp
)

target_include_directories(lammps_jax PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${LAMMPS_HEADER_DIR}"
  "${LAMMPS_HEADER_DIR}/KOKKOS"
  "${KOKKOS_ROOT}/core/src"
  "${KOKKOS_ROOT}/containers/src"
  "${KOKKOS_ROOT}/algorithms/src"
  "${KOKKOS_ROOT}/tpls/desul/include"
  "${KOKKOS_ROOT}/tpls/mdspan/include"
  "${KOKKOS_CONFIG_INCLUDE_DIR}"
  "${KOKKOS_CONFIG_INCLUDE_DIR}/core/src"
  "${CUDA_INCLUDE_DIR}"
  "${JAXLIB_INCLUDE_DIR}"
)
if(PJRT_C_API_INCLUDE_DIR)
  target_include_directories(lammps_jax PRIVATE "${PJRT_C_API_INCLUDE_DIR}")
endif()

target_link_libraries(lammps_jax PRIVATE "${CUDA_DRIVER_LIBRARY}" MPI::MPI_CXX dl)

if(CMAKE_CXX_COMPILER MATCHES "nvcc_wrapper")
  target_compile_options(lammps_jax PRIVATE -fno-lto -extended-lambda -expt-relaxed-constexpr)
  target_link_options(lammps_jax PRIVATE -fno-lto)
endif()

set_target_properties(lammps_jax PROPERTIES
  PREFIX ""
  SUFFIX "plugin.so"
  OUTPUT_NAME "lammps_jax"
)

add_custom_target(plugins ALL DEPENDS lammps_jax)
