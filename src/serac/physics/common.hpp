// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file common.hpp
 *
 * @brief A file defining some enums and structs that are used by the different physics modules
 */
#pragma once

#include "serac/numerics/odes.hpp"

namespace serac {

/**
 * @brief a struct that is used in the physics modules to clarify which template arguments are
 * user-controlled parameters (e.g. for design optimization)
 */
template <typename... T>
struct Parameters {
  static constexpr int n = sizeof...(T);  ///< how many parameters were specified
};

}  // namespace serac
