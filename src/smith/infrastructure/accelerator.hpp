// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file accelerator.hpp
 *
 * @brief This file contains the interface used for initializing/terminating any hardware accelerator-related
 * functionality
 */

#pragma once

#include "smith/smith_config.hpp"

#if defined(SMITH_USE_CUDA) || defined(SMITH_USE_HIP)
/**
 * @brief Macro that evaluates to `__host__ __device__` when compiling with nvcc or amdclang and does nothing on a host
 * compiler.
 */
#define SMITH_HOST_DEVICE __host__ __device__
/**
 * @brief Macro that evaluates to `__host__` when compiling with nvcc or amdclang and does nothing on a host compiler
 */
#define SMITH_HOST __host__
/**
 * @brief Macro that evaluates to `__host__` when compiling with nvcc or amdclang and does nothing on a host compiler
 */
#define SMITH_DEVICE __device__
#else
/**
 * @brief Macro that evaluates to `__host__ __device__` when compiling with nvcc or amdclang and does nothing on a host
 * compiler.
 */
#define SMITH_HOST_DEVICE
/**
 * @brief Macro that evaluates to `__host__` when compiling with nvcc or amdclang and does nothing on a host compiler
 */
#define SMITH_HOST
/**
 * @brief Macro that evaluates to `__device__` when compiling with nvcc or amdclang and does nothing on a host compiler
 */
#define SMITH_DEVICE
#endif

/**
 * Note: nvcc will sometimes emit a warning if a __host__ __device__ function calls a __host__-only or __device__-only
 * function. make_tensor is marked __host__ __device__ and is used frequently in the code base, so it was emitting a lot
 * of warnings. This pragma directive suppresses the warning for a specific function.
 */

#if defined(__CUDACC__)
#if __CUDAVER__ >= 75000
#define SMITH_SUPPRESS_NVCC_HOSTDEVICE_WARNING #pragma nv_exec_check_disable
#else
#define SMITH_SUPPRESS_NVCC_HOSTDEVICE_WARNING #pragma hd_warning_disable
#endif
#include <cuda_runtime.h>
#else
/**
 * @brief Macro to turn off specific nvcc warnings
 */
#define SMITH_SUPPRESS_NVCC_HOSTDEVICE_WARNING
#endif

#include <memory>
#include <cstring>
#include <tuple>

#include "axom/core.hpp"

#include "smith/infrastructure/logger.hpp"
#include "smith/infrastructure/memory.hpp"
#include "smith/infrastructure/profiling.hpp"
#include "smith/smith_config.hpp"

/**
 * @brief Accelerator functionality
 */
namespace smith {

/**
 * @brief enum used for signalling whether or not to perform certain calculations on the CPU or GPU
 */
enum class ExecutionSpace
{
  CPU,
  GPU,
  Dynamic  // Corresponds to execution that can "legally" happen on either the host or device
};

/**
 * @brief The default execution space for Smith builds
 */
constexpr ExecutionSpace default_execution_space = ExecutionSpace::CPU;

namespace detail {

/**
 * @brief Trait for "translating" between smith::ExecutionSpace and axom::MemorySpace
 */
template <ExecutionSpace space>
struct execution_to_memory {
  /// @brief The corresponding axom::MemorySpace
  static constexpr axom::MemorySpace value = axom::MemorySpace::Dynamic;
};

#ifdef SMITH_USE_UMPIRE
/// @overload
template <>
struct execution_to_memory<ExecutionSpace::CPU> {
  static constexpr axom::MemorySpace value = axom::MemorySpace::Host;
};

/// @overload
template <>
struct execution_to_memory<ExecutionSpace::GPU> {
  static constexpr axom::MemorySpace value = axom::MemorySpace::Device;
};

/// @overload
template <>
struct execution_to_memory<ExecutionSpace::Dynamic> {
  static constexpr axom::MemorySpace value = axom::MemorySpace::Unified;
};
#endif

/// @brief Helper template for @p execution_to_memory trait
template <ExecutionSpace space>
inline constexpr axom::MemorySpace execution_to_memory_v = execution_to_memory<space>::value;

/// @brief set the contents of an array to zero, byte-wise
template <typename T, int dim, axom::MemorySpace space>
void zero_out(axom::Array<T, dim, space>& arr)
{
  std::memset(arr.data(), 0, static_cast<std::size_t>(arr.size()) * sizeof(T));
}

/// @brief set the contents of an array to zero, byte-wise
template <typename T, int dim>
void zero_out(axom::ArrayView<T, dim, detail::host_memory_space>& arr)
{
  std::memset(arr.data(), 0, static_cast<std::size_t>(arr.size()) * sizeof(T));
}
#ifdef __CUDACC__
/// @overload
template <typename T, int dim>
void zero_out(axom::Array<T, dim, execution_to_memory_v<ExecutionSpace::GPU>>& arr)
{
  cudaMemset(arr.data(), 0, static_cast<std::size_t>(arr.size()) * sizeof(T));
}
#endif

}  // namespace detail

/// @brief Alias for an Array corresponding to a particular ExecutionSpace
template <typename T, int dim, ExecutionSpace space>
using ExecArray = axom::Array<T, dim, detail::execution_to_memory_v<space>>;

/// @brief Alias for an array on the CPU
template <typename T, int dim = 1>
using CPUArray = ExecArray<T, dim, ExecutionSpace::CPU>;

#ifdef SMITH_USE_CUDA

/// @brief Alias for an array on the GPU
template <typename T, int dim = 1>
using GPUArray = ExecArray<T, dim, ExecutionSpace::GPU>;

/// @brief Alias for an array in unified memory
template <typename T, int dim = 1>
using UnifiedArray = ExecArray<T, dim, ExecutionSpace::Dynamic>;

#else
// If not a CUDA build then force all arrays to be CPU

/// @brief Alias for an array on the GPU
template <typename T, int dim = 1>
using GPUArray = ExecArray<T, dim, ExecutionSpace::CPU>;

/// @brief Alias for an array in unified memory
template <typename T, int dim = 1>
using UnifiedArray = ExecArray<T, dim, ExecutionSpace::CPU>;

#endif

/// @brief Alias for an ArrayView corresponding to a particular ExecutionSpace
template <typename T, int dim, ExecutionSpace space>
using ExecArrayView = axom::ArrayView<T, dim, detail::execution_to_memory_v<space>>;

/// @brief Alias for an array view on the CPU
template <typename T, int dim = 1>
using CPUArrayView = ExecArrayView<T, dim, ExecutionSpace::CPU>;

#ifdef SMITH_USE_CUDA
/// @brief Alias for an array view on the GPU
template <typename T, int dim = 1>
using GPUArrayView = ExecArrayView<T, dim, ExecutionSpace::GPU>;
#endif

/// @brief convenience function for creating a view of an axom::Array type
template <typename T, int dim, axom::MemorySpace space>
auto view(axom::Array<T, dim, space>& arr)
{
  return axom::ArrayView<T, dim, space>(arr);
}

/**
 * @brief Namespace for methods involving accelerator-enabled builds
 */
namespace accelerator {

/**
 * @brief Initializes the device (GPU)
 *
 * @note This function should only be called once
 */
void initializeDevice(ExecutionSpace exec_space);

/**
 * @brief Cleans up the device, if applicable
 */
void terminateDevice();

#if defined(__CUDACC__)

/**
 * @brief Utility method to display last cuda error message
 *
 * @param[in] success_string A string to print if there are no CUDA error messages
 * @param[in] exit_on_error Exit on CUDA error
 */
inline void displayLastCUDAMessage(const char* success_string = "", bool exit_on_error = false)
{
  auto error = cudaGetLastError();
  if (error != cudaError::cudaSuccess) {
    if (exit_on_error) {
      SLIC_ERROR_ROOT(smith::profiling::concat("Last CUDA Error Message :", cudaGetErrorString(error)));
    } else {
      SLIC_WARNING_ROOT(smith::profiling::concat("Last CUDA Error Message :", cudaGetErrorString(error)));
    }
  } else if (strlen(success_string) > 0) {
    SLIC_INFO_ROOT(success_string);
  }
}

/**
 * @brief Utility method to query the amount of memory (bytes) that is free on the device at runtime
 *
 * Granularity appears to be about 2MB of device memory on volta.
 *
 * @return tuple of free and total memory at the moment on the device context
 */

inline std::tuple<std::size_t, std::size_t> getCUDAMemInfo()
{
  std::size_t free_memory, total_memory;
  cudaMemGetInfo(&free_memory, &total_memory);
  displayLastCUDAMessage();
  return std::make_tuple(free_memory, total_memory);
}

/**
 * @brief returns a string with GPU memory information
 */

inline std::string getCUDAMemInfoString()
{
  auto [free_memory, total_memory] = getCUDAMemInfo();
  return axom::fmt::format("Free memory: {} Total_memory: {}", free_memory, total_memory);
}

#endif

/**
 * @brief create shared_ptr to an array of `n` values of type `T`, either on the host or device
 * @tparam T the type of the value to be stored in the array
 * @tparam exec the memory space where the data lives
 * @param n how many entries to allocate in the array
 */
template <ExecutionSpace exec, typename T>
std::shared_ptr<T[]> make_shared_array(std::size_t n)
{
  if constexpr (exec == ExecutionSpace::CPU) {
    return std::shared_ptr<T[]>(new T[n]);
  }

#if defined(__CUDACC__)
  if constexpr (exec == ExecutionSpace::GPU) {
    T* data;
    cudaMalloc(&data, sizeof(T) * n);
    auto deleter = [](T* ptr) { cudaFree(ptr); };
    return std::shared_ptr<T[]>(data, deleter);
  }
#endif
}

/**
 * @brief create shared_ptr to an array of `n` values of type `T`, either on the host or device
 * @tparam T the type of the value to be stored in the array
 * @tparam exec the memory space where the data lives
 * @param n how many entries to allocate in the array
 */
template <ExecutionSpace exec, typename... T>
auto make_shared_arrays(std::size_t n)
{
  return std::tuple{make_shared_array<exec, T>(n)...};
}

}  // namespace accelerator

}  // namespace smith
