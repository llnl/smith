// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file memory.hpp
 *
 * @brief This file defines the host memory space
 */

#pragma once

#include "axom/core.hpp"

#include "smith/smith_config.hpp"

namespace smith {

namespace detail {

/**
 * @brief Sets the axom memory space based on whether or not Umpire is being used
 */
#ifdef SMITH_USE_UMPIRE
constexpr axom::MemorySpace host_memory_space = axom::MemorySpace::Host;
#else
constexpr axom::MemorySpace host_memory_space = axom::MemorySpace::Dynamic;
#endif

}  // namespace detail

}  // namespace smith
