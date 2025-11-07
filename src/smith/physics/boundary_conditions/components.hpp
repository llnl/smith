// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file solid_mechanics.hpp
 *
 * @brief Tools for tagging a set of components of a vector field for boundary condition enforcement
 */

#pragma once

#include <bitset>

namespace smith {

/// Type giving vector components meaningful names and restricting inputs to Components class to meaningful values
enum class Component : size_t
{
  X = 0b001,
  Y = 0b010,
  Z = 0b100,
  ALL = 0b111
};

/// A set to flag components of a vector field
class Components {
 public:
  /// @brief Constructor
  Components(Component i) : flags_{size_t(i)} {};

  /// @brief Indexing operator to check if a component is flagged
  bool operator[](size_t i) const { return flags_[i]; };

  /// See docstring on function declaration
  friend Components operator+(Component i, Component j);

  /// @brief Flag an additional component using the plus operator
  Components operator+(Component i)
  {
    flags_ |= size_t(i);
    return *this;
  };

  /// See docstring on function definition
  friend Components operator+(Component i, Components j);

 private:
  /// @brief Stores a bitmask indicating which of the X, Y, and Z vector components are currently flagged
  std::bitset<3> flags_;
};

/// @brief Construct a Components object from the sum of individual vector components
inline Components operator+(Component i, Component j) { return Components(i) + j; };

/// @brief Add an additional component to the set of flagged Components
inline Components operator+(Component i, Components c) { return c + i; }

}  // namespace smith
