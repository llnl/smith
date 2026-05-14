// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include "smith/physics/common.hpp"
#include "smith/physics/materials/parameterized_solid_material.hpp"
#include "smith/physics/materials/solid_material.hpp"

namespace smith::solid_mechanics {

/// @brief TimeInfo-aware wrapper for `NeoHookean`.
struct TimeInfoNeoHookean {
  /// State type reused from wrapped material.
  using State = NeoHookean::State;

  double density;  ///< Mass density.
  double K;        ///< Bulk modulus.
  double G;        ///< Shear modulus.

  template <typename T, int dim, typename GradVType>
  /// @brief Evaluate wrapped material, ignoring velocity gradient.
  SMITH_HOST_DEVICE auto operator()(const TimeInfo& /*t_info*/, State& state, const tensor<T, dim, dim>& grad_u,
                                    const GradVType& /*grad_v*/) const
  {
    return NeoHookean{.density = density, .K = K, .G = G}(state, grad_u);
  }
};

/// @brief TimeInfo-aware wrapper for `ParameterizedNeoHookeanSolid`.
struct TimeInfoParameterizedNeoHookeanSolid {
  /// State type reused from wrapped material.
  using State = ParameterizedNeoHookeanSolid::State;

  double density;  ///< Mass density.
  double K0;       ///< Base bulk modulus.
  double G0;       ///< Base shear modulus.

  template <int dim, typename DispGradType, typename GradVType, typename BulkType, typename ShearType>
  /// @brief Evaluate wrapped material, ignoring velocity gradient.
  SMITH_HOST_DEVICE auto operator()(const TimeInfo& /*t_info*/, State& state,
                                    const smith::tensor<DispGradType, dim, dim>& grad_u, const GradVType& /*grad_v*/,
                                    const BulkType& delta_k, const ShearType& delta_g) const
  {
    return ParameterizedNeoHookeanSolid{.density = density, .K0 = K0, .G0 = G0}(state, grad_u, delta_k, delta_g);
  }
};

}  // namespace smith::solid_mechanics
