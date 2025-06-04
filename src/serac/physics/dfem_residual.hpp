// Copyright (c) Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file dfem_residual.hpp
 *
 * @brief Implements the residual interface using serac::ShapeAwareDfem.
 * Allows for generic specification of body and boundary integrals
 */

#pragma once

#include "mfem.hpp"

#include "serac/physics/residual.hpp"
#include "serac/physics/mesh.hpp"
#include "serac/physics/state/finite_element_state.hpp"
#include "serac/physics/state/finite_element_dual.hpp"

namespace serac {

/**
 * @brief The nonlinear residual class
 *
 * This uses Dfem to compute fairly arbitrary residuals and tangent
 * stiffness matrices based on body and boundary integrals.
 *
 */
class DfemResidual : public Residual {
 public:

  /**
   * @brief Construct a new DfemResidual object
   *
   * @param physics_name A name for the physics module instance
   * @param diff_op MFEM differentiable operator r-value
   */
  DfemResidual(std::string physics_name, std::shared_ptr<Mesh> mesh, mfem::future::DifferentiableOperator&& diff_op)
      : Residual(physics_name), mesh_( mesh ), 
      mesh_nodes_( static_cast<mfem::ParGridFunction*>( mesh->mfemParMesh().GetNodes()) ), diff_op_( std::move(diff_op) )
  {
  }

  /**
   * @brief Add a body integral contribution to the residual
   *
   * @tparam BodyIntegralType The type of the body integral
   * @param body_integral A function describing the body force applied
   * used.
   * @pre body_integral must be a object that can be called with the following arguments:
   *    1. `double t` the time
   *    2. `tensor<T,dim> x` the spatial coordinates for the quadrature point
   *    3. `tuple{value, derivative}`, a variadic list of tuples (each with a values and derivative),
   *            one tuple for each of the trial spaces specified in the `DependsOn<...>` argument.
   * @note The actual types of these arguments passed will be `double`, `tensor<double, ... >` or tuples thereof
   *    when doing direct evaluation. When differentiating with respect to one of the inputs, its stored
   *    values will change to `dual` numbers rather than `double`. (e.g. `tensor<double,3>` becomes `tensor<dual<...>,
   * 3>`)
   *
   */
   template <typename BodyIntegralType, typename InputType, typename OutputType>
  void addBodyIntegral(BodyIntegralType body_integral, InputType inputs, OutputType outputs, 
    const mfem::IntegrationRule& integration_rule, mfem::Array<int> domain_attributes, std::integer_sequence<size_t, 0> derivative_ids)
  {
    diff_op_.AddDomainIntegrator(body_integral, inputs, outputs, integration_rule, domain_attributes, derivative_ids);
  }

  /// @overload
  mfem::Vector residual(double, double, const std::vector<FieldPtr>& fields, int block_row = 0) const override
  {
    SLIC_ERROR_IF(block_row != 0, "Invalid block row and column requested in fieldJacobian for FunctionalResidual");
    diff_op_.SetParameters({mesh_nodes_});
    mfem::Vector ret(mesh_nodes_->Size());
    diff_op_.Mult( *fields[0], ret );
    return ret;
  }

  std::unique_ptr<mfem::HypreParMatrix> jacobian(double, double, const std::vector<FieldPtr>&,
    const std::vector<double>&,
    int) const override {
      return std::make_unique<mfem::HypreParMatrix>();
    }

  /// @overload
  void jvp(double, double, const std::vector<FieldPtr>& fields, const std::vector<FieldPtr>& v_fields,
           const std::vector<DualFieldPtr>& jvp_reactions) const override
  {
    SLIC_ERROR_IF(v_fields.size() != fields.size(),
                  "Invalid number of field sensitivities relative to the number of fields");
    SLIC_ERROR_IF(jvp_reactions.size() != 1, "DfemResidual nonlinear systems only supports 1 output residual");

    auto grad_op = diff_op_.GetDerivative(0, {fields[0]}, {mesh_nodes_} );
    grad_op->AddMult(*v_fields[0], *jvp_reactions[0]);
  }

  /// @overload
  void vjp(double, double, const std::vector<FieldPtr>& fields, const std::vector<FieldPtr>& v_fields,
           const std::vector<DualFieldPtr>& vjp_sensitivities) const override
  {
    SLIC_ERROR_IF(vjp_sensitivities.size() != fields.size(),
                  "Invalid number of field sensitivities relative to the number of fields");
    SLIC_ERROR_IF(v_fields.size() != 1, "DfemResidual nonlinear systems only supports 1 output residual");

    auto grad_op = diff_op_.GetDerivative(0, {fields[0]}, {mesh_nodes_} );
    grad_op->AddMultTranspose(*v_fields[0], *vjp_sensitivities[0]);
  }

 protected:

  /// @brief primary mesh
  std::shared_ptr<Mesh> mesh_;

  /// @brief reference nodal coordinates
  mfem::ParGridFunction* mesh_nodes_;

  /// @brief dfem residual evaluator
  mutable mfem::future::DifferentiableOperator diff_op_;
};

}  // namespace serac
