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

#include "serac/physics/dfem_residual2.hpp"

namespace serac {

template <typename Material, typename... Parameters>
struct StressDivQFunction {
  SERAC_HOST_DEVICE inline auto operator()(
      // mfem::real_t dt, // TODO: figure out how to pass this in
      const mfem::future::tensor<mfem::real_t, Material::dim, Material::dim>& du_dxi,
      const mfem::future::tensor<mfem::real_t, Material::dim, Material::dim>& dv_dxi,
      const mfem::future::tensor<mfem::real_t, Material::dim, Material::dim>&,
      const mfem::future::tensor<mfem::real_t, Material::dim, Material::dim>& dX_dxi, mfem::real_t weight,
      Parameters... params) const
  {
    auto dxi_dX = mfem::future::inv(dX_dxi);
    auto du_dX = mfem::future::dot(du_dxi, dxi_dX);
    auto dv_dX = mfem::future::dot(dv_dxi, dxi_dX);
    double dt = 1.0;  // TODO: figure out how to pass this in to the qfunction
    auto P = mfem::future::get<0>(material.pkStress(dt, du_dX, dv_dxi, params...));
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
class SolidDfemResidual2 : public DfemResidual2 {
 public:
  /// @brief disp, velo, accel
  static constexpr int NUM_STATE_VARS = 4;

  enum FieldIDs
  {
    DISP,   ///< displacement
    VELO,   ///< velocity
    ACCEL,  ///< acceleration
    COORD   ///< coordinates
  };

  /**
   * @brief Construct a new SolidResidual object
   *
   * @param physics_name A name for the physics module instance
   * @param mesh The serac Mesh
   */
  SolidDfemResidual2(std::string physics_name, std::shared_ptr<Mesh> mesh,
                     const mfem::ParFiniteElementSpace& test_space,
                     const std::vector<const mfem::ParFiniteElementSpace*>& parameter_fe_spaces = {})
      : DfemResidual2(physics_name, mesh), test_space_(test_space)
  {
    parameter_fields_.reserve(parameter_fe_spaces.size());
    for (auto space : parameter_fe_spaces) {
      parameter_fields_.emplace_back(NUM_STATE_VARS + parameter_fields_.size(), space);
    }
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
  template <int... active_parameters, typename MaterialType>
  void setMaterial(DependsOn<active_parameters...>, std::string /*body_name*/, const MaterialType& material,
                   const mfem::IntegrationRule& displacement_ir)
  {
    mfem::future::tuple qfunction_inputs{
        // TODO: figure out how to pass in dt
        mfem::future::Gradient<DISP>{},  mfem::future::Gradient<VELO>{},
        mfem::future::Gradient<ACCEL>{}, mfem::future::Gradient<COORD>{},
        mfem::future::Weight{},          mfem::future::Value<active_parameters + NUM_STATE_VARS>{}...};
    mfem::future::tuple qfunction_outputs{mfem::future::Gradient<DISP>{}};
    // TODO: find out the right attributes from the body name
    mfem::Array<int> solid_domain_attributes(DfemResidual2::mesh_->mfemParMesh().attributes.Max());
    auto stress_div_qf = StressDivQFunction<MaterialType, typename MaterialType::param_type>{.material = material};
    DfemResidual2::addBodyIntegral(stress_div_qf, solutionFieldDescriptors(), parameter_fields_, qfunction_inputs,
                                   qfunction_outputs, displacement_ir, solid_domain_attributes,
                                   std::integer_sequence<size_t, 0>{});
  }

 protected:
  /**
   * @brief Construct the field descriptors for the differentiable operator
   *
   * @param parameter_fe_spaces The parameter finite element spaces
   * @return A vector of field descriptors for the differentiable operator
   */
  static std::vector<mfem::future::FieldDescriptor> parameterFieldDescriptors(
      std::vector<const mfem::ParFiniteElementSpace*> parameter_fe_spaces)
  {
    std::vector<mfem::future::FieldDescriptor> field_descriptors;
    field_descriptors.reserve(parameter_fe_spaces.size());
    for (const auto& space : parameter_fe_spaces) {
      field_descriptors.emplace_back(NUM_STATE_VARS + field_descriptors.size(), space);
    }
    return field_descriptors;
  }
  /**
   * @brief Construct the field descriptors for the solution fields
   *
   */
  std::vector<mfem::future::FieldDescriptor> solutionFieldDescriptors()
  {
    return {
        {DISP, &test_space_},
        {VELO, &test_space_},
        {ACCEL, &test_space_},
        {COORD, static_cast<const mfem::ParFiniteElementSpace*>(DfemResidual2::mesh_->mfemParMesh().GetNodalFESpace())},
    };
  }
  const mfem::ParFiniteElementSpace& test_space_;
  std::vector<mfem::future::FieldDescriptor> parameter_fields_;
};

}  // namespace serac
