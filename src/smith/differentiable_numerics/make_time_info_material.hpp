// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <utility>

#include "smith/physics/common.hpp"
#include "smith/infrastructure/accelerator.hpp"

namespace smith {

/// @brief Adapter that lets a material without time information satisfy the TimeInfo material interface.
template <typename Material>
struct TimeInfoMaterial {
  /// State type forwarded from the wrapped material.
  using State = typename Material::State;

  Material material;  ///< Wrapped material.
  double density;     ///< Density forwarded for solid mechanics inertial terms.

  /// @brief Evaluate the wrapped material, ignoring TimeInfo and velocity gradient.
  template <typename StateType, typename GradUType, typename GradVType, typename... Args>
  SMITH_HOST_DEVICE auto operator()(const TimeInfo& /*t_info*/, StateType&& state, GradUType&& grad_u,
                                    const GradVType& /*grad_v*/, Args&&... args) const
  {
    return material(std::forward<StateType>(state), std::forward<GradUType>(grad_u), std::forward<Args>(args)...);
  }
};

/// @brief Create a TimeInfoMaterial adapter for a material with signature material(state, grad_u, args...).
template <typename Material>
TimeInfoMaterial<Material> makeTimeInfoMaterial(Material mat)
{
  const double density = mat.density;
  return TimeInfoMaterial<Material>{std::move(mat), density};
}

}  // namespace smith
