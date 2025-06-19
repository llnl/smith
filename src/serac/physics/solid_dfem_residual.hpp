// Copyright (c) Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file solid_dfem_residual.hpp
 *
 * @brief Implements the residual interface for solid mechanics physics.
 * Derives from dfem_residual.
 */

#pragma once

#include "serac/physics/dfem_residual.hpp"

namespace serac {

template <typename Material, int dim>
struct MomentumQFunction {
  SERAC_HOST_DEVICE inline auto operator()(
      mfem::real_t dt, const mfem::future::tensor<mfem::real_t, Material::state_size>& internal_state,
      const mfem::future::tensor<mfem::real_t, dim, dim>& du_dxi,
      const mfem::future::tensor<mfem::real_t, dim, dim>& dv_dxi, const mfem::future::tensor<mfem::real_t, dim, dim>&,
      const mfem::future::tensor<mfem::real_t, dim, dim>& dX_dxi, mfem::real_t weight) const
  {
    auto dxi_dX = mfem::future::inv(dX_dxi);
    auto du_dX = mfem::future::dot(du_dxi, dxi_dX);
    auto dv_dX = mfem::future::dot(dv_dxi, dxi_dX);
    auto P = mfem::future::get<0>(material.pkStress(dt, internal_state, du_dX, dv_dxi));
    auto JxW = mfem::future::det(dX_dxi) * weight * mfem::future::transpose(dxi_dX);
    return mfem::future::tuple{P * JxW};
  }

  Material material;  ///< the material model to use for computing the stress
};

/**
 * @brief The nonlinear residual class
 *
 * This uses Functional to compute the solid mechanics residuals and tangent
 * stiffness matrices.
 *
 * @tparam order The order of the discretization of the displacement and velocity fields
 * @tparam dim The spatial dimension of the mesh
 */
template <int dim>
class SolidDfemResidual : public DfemResidual {
 public:
  /// @brief disp, velo, accel
  static constexpr int NUM_STATE_VARS = 3;

  /**
   * @brief Construct a new SolidResidual object
   *
   * @param physics_name A name for the physics module instance
   * @param mesh The serac Mesh
   * @param diff_op A differentiable operator that computes the residual and jacobian
   */
  SolidDfemResidual(std::string physics_name, std::shared_ptr<Mesh> mesh,
                    mfem::future::DifferentiableOperator&& diff_op)
      : DfemResidual(physics_name, mesh, std::move(diff_op))
  {
  }

  /**
   * @brief Set the material stress response and mass properties for the physics module
   *
   * @tparam MaterialType The solid material type
   * @tparam StateType the type that contains the internal variables for MaterialType
   * @param body_name string name for a registered body Domain on the mesh
   * @param material A material that provides a function to evaluate stress
   * @pre material must be a object that can be called with the following arguments:
   *    1. `MaterialType::State & state` an mutable reference to the internal variables for this quadrature point
   *    2. `tensor<T,dim,dim> du_dx` the displacement gradient at this quadrature point
   *    3. `tuple{value, derivative}`, a tuple of values and derivatives for each parameter field
   *            specified in the `DependsOn<...>` argument.
   *
   * @note The actual types of these arguments passed will be `double`, `tensor<double, ... >` or tuples thereof
   *    when doing direct evaluation. When differentiating with respect to one of the inputs, its stored
   *    values will change to `dual` numbers rather than `double`. (e.g. `tensor<double,3>` becomes `tensor<dual<...>,
   * 3>`)
   *
   * @param qdata the buffer of material internal variables at each quadrature point
   *
   * @pre MaterialType must have a public method `density` which can take FiniteElementState parameter inputs
   * @pre MaterialType must have a public method 'pkStress' which returns the first Piola-Kirchhoff stress
   *
   */
  template <typename MaterialType>
  void setMaterial(std::string, const MaterialType& material, const mfem::IntegrationRule& displacement_ir)
  {
    mfem::future::tuple inputs{mfem::future::Gradient<0>{}, mfem::future::Gradient<1>{}, mfem::future::Gradient<2>{},
                               mfem::future::Gradient<3>{}, mfem::future::Weight{}};
    mfem::future::tuple outputs{mfem::future::Gradient<4>{}};
    mfem::Array<int> solid_domain_attributes(DfemResidual::mesh_->mfemParMesh().attributes.Max());
    auto momentum_qf = MomentumQFunction<MaterialType, dim>{.material = material};
    DfemResidual::diff_op_.AddDomainIntegrator(momentum_qf, inputs, outputs, displacement_ir, solid_domain_attributes);
  }
};

}  // namespace serac
