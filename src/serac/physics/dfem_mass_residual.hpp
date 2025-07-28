// Copyright (c) Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file dfem_mass_residual.hpp
 */

#pragma once

#include "mfem.hpp"

#include "serac/physics/dfem_residual.hpp"
#include "serac/physics/mesh.hpp"
#include "serac/physics/state/finite_element_state.hpp"

namespace serac {

template <int MassDim, int SpatialDim>
auto create_solid_mass_residual(const std::string& physics_name, std::shared_ptr<serac::Mesh>& mesh,
                                const FiniteElementState& lumped_field, const FiniteElementState& density,
                                const mfem::IntegrationRule& ir)
{
  enum FieldIDs
  {
    COORD,
    DENSITY,
    TEST
  };

  auto residual = std::make_shared<DfemResidual>(
      physics_name, mesh, lumped_field.space(),
      DfemResidual::SpacesT{static_cast<mfem::ParFiniteElementSpace*>(mesh->mfemParMesh().GetNodes()->FESpace()),
                            &density.space()});

  mfem::future::tuple<mfem::future::Gradient<COORD>, mfem::future::Weight, mfem::future::Value<DENSITY>>
      mass_integral_inputs{};
  mfem::future::tuple<mfem::future::Value<TEST>> mass_integral_outputs{};

  residual->addBodyIntegral(
      mesh->mfemParMesh().attributes,
      [](mfem::future::tensor<mfem::real_t, SpatialDim, SpatialDim> dX_dxi, mfem::real_t weight, double rho) {
        auto ones = mfem::future::make_tensor<MassDim>([](int) { return 1.0; });
        auto J = mfem::future::det(dX_dxi) * weight;
        return mfem::future::tuple{rho * ones * J};
      },
      mass_integral_inputs, mass_integral_outputs, ir, std::index_sequence<>{});
  return residual;
}

}  // namespace serac
