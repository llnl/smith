// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file dfem_mass_weak_form.hpp
 */

#pragma once

#include "smith/smith_config.hpp"

#ifdef SMITH_USE_DFEM

#include "mfem.hpp"

#include "smith/physics/dfem_weak_form.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/state/finite_element_state.hpp"
#include "smith/physics/boundary_conditions/boundary_condition_manager.hpp"

namespace smith {

/**
 * @brief
 *
 */
class LumpedMassExplicitNewmark {
 public:
  LumpedMassExplicitNewmark(const std::shared_ptr<WeakForm>& weak_form, const std::shared_ptr<WeakForm>& mass_weak_form,
                            std::shared_ptr<BoundaryConditionManager> bc_manager)
      : weak_form_(weak_form), mass_weak_form_(mass_weak_form), bc_manager_(bc_manager)
  {
  }

  std::tuple<std::vector<FiniteElementState>, double> advanceState(const std::vector<ConstFieldPtr>& states,
                                                                   const std::vector<ConstFieldPtr>& params,
                                                                   double time, double dt)
  {
    SLIC_ERROR_ROOT_IF(states.size() != 4, "Expected 4 states: displacement, velocity, acceleration, and coordinates");

    enum States
    {
      DISPLACEMENT,
      VELOCITY,
      ACCELERATION,
      COORDINATES
    };

    enum Params
    {
      DENSITY
    };

    const auto& u = *states[DISPLACEMENT];
    const auto& v = *states[VELOCITY];
    const auto& a = *states[ACCELERATION];

    auto v_pred = v;
    v_pred.Add(0.5 * dt, a);
    auto u_pred = u;
    u_pred.Add(dt, v_pred);

    if (bc_manager_) {
      u_pred.SetSubVector(bc_manager_->allEssentialTrueDofs(), 0.0);
    }

    TimeInfo time_info(time, dt);

    auto m_inv = mass_weak_form_->residual(time_info, &u, {states[COORDINATES], params[DENSITY]});
    m_inv.Reciprocal();

    std::vector<ConstFieldPtr> pred_states = {&u_pred, &v_pred, &a, states[COORDINATES], params[DENSITY]};

    auto force_resid = weak_form_->residual(time_info, &u_pred, pred_states);

    FiniteElementState a_pred(a.space(), "acceleration_pred");
    auto a_pred_ptr = a_pred.Write();
    auto m_inv_ptr = m_inv.Read();
    auto force_resid_ptr = force_resid.Read();
    mfem::forall_switch(a_pred.UseDevice(), a_pred.Size(),
                        [=] MFEM_HOST_DEVICE(int i) { a_pred_ptr[i] = m_inv_ptr[i] * force_resid_ptr[i]; });

    v_pred.Add(0.5 * dt, a_pred);

    return {{u_pred, v_pred, a_pred, *states[COORDINATES]}, time + dt};
  }

 private:
  std::shared_ptr<WeakForm> weak_form_;
  std::shared_ptr<WeakForm> mass_weak_form_;
  std::shared_ptr<BoundaryConditionManager> bc_manager_;
};

template <int MassDim, int SpatialDim>
auto create_solid_mass_weak_form(const std::string& physics_name, std::shared_ptr<smith::Mesh>& mesh,
                                 const FiniteElementState& lumped_field, const FiniteElementState& density,
                                 const mfem::IntegrationRule& ir)
{
  enum FieldIDs
  {
    COORDINATES,
    DENSITY,
    TEST
  };

  auto residual = std::make_shared<DfemWeakForm>(
      physics_name, mesh, lumped_field.space(),
      DfemWeakForm::SpacesT{static_cast<mfem::ParFiniteElementSpace*>(mesh->mfemParMesh().GetNodes()->FESpace()),
                            &density.space()});

  mfem::future::tuple<mfem::future::Gradient<COORDINATES>, mfem::future::Weight, mfem::future::Value<DENSITY>>
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

}  // namespace smith

#endif
