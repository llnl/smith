// Copyright (c) Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file dfem_residual2.hpp
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
 * This uses dFEM to compute fairly arbitrary residuals and tangent
 * stiffness matrices based on body and boundary integrals.
 *
 */
class DfemResidual2 : public Residual {
 public:
  using SpacesT = std::vector<const mfem::ParFiniteElementSpace*>;  ///< typedef

  /**
   * @brief Construct a new DfemResidual object
   *
   * @param physics_name A name for the physics module instance
   * @param mesh The serac mesh
   * @param diff_op A differentiable operator that computes the residual and jacobian
   */
  DfemResidual2(std::string physics_name, std::shared_ptr<Mesh> mesh) : Residual(physics_name), mesh_(mesh) {}

  /**
   * @brief Add a body integral contribution to the residual
   *
   * @tparam BodyIntegralType The type of the body integral
   * // DependsOn<active_parameters...> can be indices into fields which the body integral may depend on
   * @param body_name The name of the registered domain over which the body force is applied. If nothing is supplied
   * the entire domain is
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
  template <typename QFunctionType, typename InputType, typename OutputType>
  void addBodyIntegral(QFunctionType qfunction, const std::vector<mfem::future::FieldDescriptor>& solution_fields,
                       const std::vector<mfem::future::FieldDescriptor>& param_fields, InputType qfunction_inputs,
                       OutputType qfunction_outputs, const mfem::IntegrationRule& integration_rule,
                       mfem::Array<int> domain_attributes, std::integer_sequence<size_t, 0> derivative_ids)
  {
    // get field IDs
    std::vector<size_t> field_ids;
    field_ids.reserve(solution_fields.size() + param_fields.size());
    for (const auto& field : solution_fields) {
      field_ids.push_back(field.id);
    }
    for (const auto& field : param_fields) {
      field_ids.push_back(field.id);
    }
    // just keep the first input field as a solution field, make the rest parameter fields to simplify the Mult() call
    std::vector<mfem::future::FieldDescriptor> other_fields;
    other_fields.reserve(param_fields.size() + solution_fields.size() - 1);
    for (size_t i = 1; i < solution_fields.size(); ++i) {
      other_fields.push_back(solution_fields[i]);
    }
    for (const auto& field : param_fields) {
      other_fields.push_back(field);
    }
    diff_ops_with_field_ids_.emplace_back(
        std::make_tuple(mfem::future::DifferentiableOperator({solution_fields[0]}, other_fields, mesh_->mfemParMesh()),
                        std::move(field_ids)));
    std::get<0>(diff_ops_with_field_ids_.back())
        .AddDomainIntegrator(qfunction, qfunction_inputs, qfunction_outputs, integration_rule, domain_attributes,
                             derivative_ids);
  }

  /// @overload
  mfem::Vector residual(double, double, const std::vector<FieldPtr>& fields, int block_row = 0) const override
  {
    SLIC_ERROR_IF(block_row != 0, "Invalid block row and column requested in fieldJacobian for DfemResidual2");
    mfem::Vector resid(fields[0]->space().GetVSize());
    resid = 0.0;
    for (auto& diff_op_with_field_ids : diff_ops_with_field_ids_) {
      std::vector<mfem::Vector*> other_fields;
      auto& field_ids = std::get<1>(diff_op_with_field_ids);
      other_fields.reserve(field_ids.size() - 1);
      for (size_t i = 1; i < field_ids.size(); ++i) {
        other_fields.push_back(&fields[field_ids[i]]->gridFunction());
      }
      std::get<0>(diff_op_with_field_ids).SetParameters(other_fields);
      mfem::Vector term_resid(fields[0]->space().GetTrueVSize());
      std::get<0>(diff_op_with_field_ids).Mult(*fields[0], term_resid);
      resid += term_resid;
    }
    return resid;
  }

  /// @overload
  std::unique_ptr<mfem::HypreParMatrix> jacobian(double, double, const std::vector<FieldPtr>&,
                                                 const std::vector<double>&, int) const override
  {
    return std::make_unique<mfem::HypreParMatrix>();
  }

  /// @overload
  void jvp(double, double, const std::vector<FieldPtr>& fields, const std::vector<FieldPtr>& v_fields,
           const std::vector<DualFieldPtr>& jvp_reactions) const override
  {
    SLIC_ERROR_IF(v_fields.size() != fields.size(),
                  "Invalid number of field sensitivities relative to the number of fields");
    SLIC_ERROR_IF(jvp_reactions.size() != 1, "DfemResidual nonlinear systems only supports 1 output residual");

    auto grad_op = diff_op_.GetDerivative(0, {fields[0]}, {mesh_nodes_});
    grad_op->AddMult(*v_fields[0], *jvp_reactions[0]);
  }

  /// @overload
  void vjp(double, double, const std::vector<FieldPtr>& fields, const std::vector<FieldPtr>& v_fields,
           const std::vector<DualFieldPtr>& vjp_sensitivities) const override
  {
    SLIC_ERROR_IF(vjp_sensitivities.size() != fields.size(),
                  "Invalid number of field sensitivities relative to the number of fields");
    SLIC_ERROR_IF(v_fields.size() != 1, "FunctionalResidual nonlinear systems only supports 1 output residual");

    auto grad_op = diff_op_.GetDerivative(0, {fields[0]}, {mesh_nodes_});
    grad_op->AddMultTranspose(*v_fields[0], *vjp_sensitivities[0]);
  }

 protected:
  /// @brief primary mesh
  std::shared_ptr<Mesh> mesh_;

  mutable std::vector<std::tuple<mfem::future::DifferentiableOperator, std::vector<size_t>>> diff_ops_with_field_ids_;
};

}  // namespace serac
