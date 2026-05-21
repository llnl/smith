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

template <typename Material>
struct TimeInfoMaterial : Material {
  using State = typename Material::State;

  template <typename StateType, typename GradUType, typename GradVType, typename... Args>
  SMITH_HOST_DEVICE auto operator()(const TimeInfo& /*t_info*/, StateType&& state, GradUType&& grad_u,
                                    const GradVType& /*grad_v*/, Args&&... args) const
  {
    const Material& material = *this;
    return material(std::forward<StateType>(state), std::forward<GradUType>(grad_u), std::forward<Args>(args)...);
  }
};

template <typename Material>
TimeInfoMaterial<Material> makeTimeInfoMaterial(Material mat)
{
  return TimeInfoMaterial<Material>{std::move(mat)};
}

}  // namespace smith
