// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
#pragma once

#include "mfem.hpp"

namespace smith {

static constexpr uint32_t NO_DIFFERENTIATION = uint32_t(1) << 31;

template <uint32_t i>
struct DifferentiateWRT {};

/**
 * @brief this type exists solely as a way to signal to `smith::Functional` that the function
 * smith::Functional::operator()` should differentiate w.r.t. a specific argument
 */
struct differentiate_wrt_this {
  const mfem::Vector& ref;  ///< the actual data wrapped by this type

  /// @brief implicitly convert back to `mfem::Vector` to extract the actual data
  operator const mfem::Vector&() const { return ref; }
};

/**
 * @brief this function is intended to only be used in combination with
 *   `smith::Functional::operator()`, as a way for the user to express that
 *   it should both evaluate and differentiate w.r.t. a specific argument (only 1 argument at a time)
 *
 * For example:
 * @code{.cpp}
 *     mfem::Vector arg0 = ...;
 *     mfem::Vector arg1 = ...;
 *     mfem::Vector just_the_value = my_functional(arg0, arg1);
 *     auto [value, gradient_wrt_arg1] = my_functional(arg0, differentiate_wrt(arg1));
 * @endcode
 */
inline auto differentiate_wrt(const mfem::Vector& v) { return differentiate_wrt_this{v}; }

}  // namespace smith
