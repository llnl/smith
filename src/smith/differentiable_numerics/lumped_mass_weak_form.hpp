// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file lumped_mass_weak_form.hpp
 *
 * @brief smith::functional implementation for evaluating nodal lumped masses, give an input density field
 */

#pragma once

#include "smith/physics/functional_weak_form.hpp"
#include "smith/physics/mesh.hpp"

namespace smith {

/// @brief creates a lumped mass weak form
template <int spatial_dim, typename LumpedFieldSpace, typename DensitySpace, typename... parameter_space>
auto createSolidMassWeakForm(const std::string& physics_name, std::shared_ptr<smith::Mesh>& mesh,
                             const FiniteElementState& lumped_field, const FiniteElementState& density)
{
  static constexpr int lumped_dim = LumpedFieldSpace::components;

  using WeakFormT = smith::FunctionalWeakForm<spatial_dim, LumpedFieldSpace, Parameters<DensitySpace>>;
  auto weak_form = std::make_shared<WeakFormT>(physics_name, mesh, lumped_field.space(),
                                               typename WeakFormT::SpacesT{&density.space()});
  weak_form->addBodyIntegral(smith::DependsOn<0>{}, mesh->entireBodyName(), [](double /*time*/, auto /*X*/, auto Rho) {
    if constexpr (lumped_dim == 1) {
      return smith::tuple{get<VALUE>(Rho), tensor<double, spatial_dim>{}};
    } else {
      auto ones = make_tensor<lumped_dim>([](int) { return 1.0; });
      return smith::tuple{get<VALUE>(Rho) * ones, tensor<double, lumped_dim, spatial_dim>{}};
    }
  });

  return weak_form;
}

}  // namespace smith
