// Copyright (c) Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file profiling.hpp
 *
 * @brief Various helper functions and macros for profiling using Caliper
 */

#pragma once

#include <string>
#include <sstream>

#include "serac/serac_config.hpp" // IWYU pragma: keep

#ifdef SERAC_USE_ADIAK
#include "adiak.hpp"
#endif

#ifdef SERAC_USE_CALIPER
#include "caliper/cali-manager.h"
#include "caliper/cali.h"
#endif

#include "mpi.h"

/**
 * @def SERAC_SET_METADATA(name, data)
 * Sets metadata in adiak/caliper. Calls adiak::value
 */

#ifdef SERAC_USE_ADIAK
#define SERAC_SET_METADATA(name, data) adiak::value(name, data)
#else
#define SERAC_SET_METADATA(name, data)
#endif

/**
 * @def SERAC_MARK_FUNCTION
 * Marks a function for Caliper profiling. No-op macro when ENABLE_PROFILING is off.
 */

/**
 * @def SERAC_MARK_LOOP_BEGIN(id, name)
 * Marks the beginning of a loop block for Caliper profiling. No-op macro when ENABLE_PROFILING is off.
 */

/**
 * @def SERAC_MARK_LOOP_ITERATION(id, i)
 * Marks the beginning of a loop iteration for Caliper profiling. No-op macro when ENABLE_PROFILING is off.
 */

/**
 * @def SERAC_MARK_LOOP_END(id)
 * Marks the end of a loop block for Caliper profiling. No-op macro when ENABLE_PROFILING is off.
 */

/**
 * @def SERAC_MARK_BEGIN(id)
 * Marks the start of a region Caliper profiling. No-op macro when ENABLE_PROFILING is off.
 */

/**
 * @def SERAC_MARK_END(id)
 * Marks the end of a region Caliper profiling. No-op macro when ENABLE_PROFILING is off.
 */

/**
 * @def SERAC_MARK_SCOPE(name)
 * Marks a particular scope for Caliper profiling. No-op macro when ENABLE_PROFILING is off.
 */

// NOTE: The motivation behind wrapping Caliper macros to avoid conflicting macro definitions in the no-op case, and
// give downstream users the option to disable profiling Serac if it pollutes their timings.

#ifdef SERAC_USE_CALIPER
#define SERAC_MARK_FUNCTION CALI_CXX_MARK_FUNCTION
#define SERAC_MARK_LOOP_BEGIN(id, name) CALI_CXX_MARK_LOOP_BEGIN(id, name)
#define SERAC_MARK_LOOP_ITERATION(id, i) CALI_CXX_MARK_LOOP_ITERATION(id, i)
#define SERAC_MARK_LOOP_END(id) CALI_CXX_MARK_LOOP_END(id)
#define SERAC_MARK_BEGIN(name) CALI_MARK_BEGIN(name)
#define SERAC_MARK_END(name) CALI_MARK_END(name)
#define SERAC_MARK_SCOPE(name) CALI_CXX_MARK_SCOPE(name)
#else
// Define no-op macros in case Serac has not been configured with Caliper
#define SERAC_MARK_FUNCTION
#define SERAC_MARK_LOOP_BEGIN(id, name)
#define SERAC_MARK_LOOP_ITERATION(id, i)
#define SERAC_MARK_LOOP_END(id)
#define SERAC_MARK_BEGIN(name)
#define SERAC_MARK_END(name)
#define SERAC_MARK_SCOPE(name)
#endif

/// profiling namespace
namespace serac::profiling {

/**
 * @brief Initializes performance monitoring using the Caliper and Adiak libraries
 * @param comm The MPI communicator (used by Adiak), optional
 * @param options The Caliper ConfigManager config string, optional
 * @see https://software.llnl.gov/Caliper/ConfigManagerAPI.html#configmanager-configuration-string-syntax
 */
void initialize([[maybe_unused]] MPI_Comm comm = MPI_COMM_WORLD, [[maybe_unused]] std::string options = "");

/**
 * @brief Concludes performance monitoring and writes collected data to a file
 */
void finalize();

/// Produces a string by applying << to all arguments
template <typename... T>
std::string concat(T... args)
{
  std::stringstream ss;
  // this fold expression is a more elegant way to implement the concatenation,
  // but nvcc incorrectly generates warning "warning: expression has no effect"
  // when using the fold expression version
  // (ss << ... << args);
  ((ss << args), ...);
  return ss.str();
}

}  // namespace serac::profiling
