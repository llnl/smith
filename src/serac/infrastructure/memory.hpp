// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
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

#include "serac/serac_config.hpp"

namespace serac {

namespace detail {

/**
 * @brief Sets the axom memory space based on whether or not Umpire is being used
 */
#ifdef SERAC_USE_UMPIRE
constexpr axom::MemorySpace host_memory_space = axom::MemorySpace::Host;
#else
constexpr axom::MemorySpace host_memory_space = axom::MemorySpace::Dynamic;
#endif

}  // namespace detail

}  // namespace serac
