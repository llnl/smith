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

// NOTE: Args needs to be on the functor struct instead of the operator() so that operator() isn't overloaded and dfem
// can deduce the type
template <typename Primal, typename OrigQFn, typename... Args>
struct InnerQFunction {
  InnerQFunction(OrigQFn orig_qfn) : orig_qfn_(orig_qfn) {}

  SERAC_HOST_DEVICE inline auto operator()(Primal V, Args... args) const
  {
    auto orig_residual = mfem::future::get<0>(orig_qfn_(std::forward<Args>(args)...));
    return mfem::future::tuple{mfem::future::inner(V, orig_residual)};
  }

  OrigQFn orig_qfn_;
};

// Step 2: deduce the type of the parameters and the first tuple element of the return type of the operator()
// Step 3: create the InnerQFunction with the deduced types
template <typename OrigQFn, typename R, typename... Args>
auto makeInnerQFunction(OrigQFn orig_qfn, R (OrigQFn::*)(Args...) const)
{
  // TODO: is there a better way to get the type of the first tuple element?
  return InnerQFunction<decltype(mfem::future::type<0>(R{})), OrigQFn, Args...>{orig_qfn};
}

// Step 1: get function pointer to operator()
template <typename OrigQFn>
auto makeInnerQFunction(OrigQFn orig_qfn)
{
  return makeInnerQFunction(orig_qfn, &OrigQFn::operator());
}

/**
 * @brief The nonlinear residual class
 *
 * This uses dFEM to compute fairly arbitrary residuals and tangent
 * stiffness matrices based on body and boundary integrals.
 *
 */
class DfemResidual : public Residual {
 public:
  using SpacesT = std::vector<const mfem::ParFiniteElementSpace*>;  ///< typedef

  /**
   * @brief Construct a new DfemResidual object
   *
   * @param physics_name A name for the physics module instance
   * @param mesh The serac mesh
   * @param diff_op A differentiable operator that computes the residual and jacobian
   */
  DfemResidual(std::string physics_name, std::shared_ptr<Mesh> mesh,
               const mfem::ParFiniteElementSpace& output_mfem_space,
               const std::vector<const mfem::ParFiniteElementSpace*>& input_mfem_spaces)
      : Residual(physics_name),
        mesh_(mesh),
        output_mfem_space_(output_mfem_space),
        input_mfem_spaces_(input_mfem_spaces),
        residual_(makeFieldDescriptors({&output_mfem_space}, input_mfem_spaces.size()),
                  makeFieldDescriptors(input_mfem_spaces), mesh->mfemParMesh()),
        // TODO: add space for residual.  do we need a space for the output (virtual work)?
        virtual_work_(makeFieldDescriptors({&output_mfem_space}, input_mfem_spaces.size()),
                      makeFieldDescriptors(input_mfem_spaces), mesh->mfemParMesh())
  {
  }

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
  template <typename BodyIntegralType, typename InputType, typename DerivIdsType, typename OutputType>
  void addBodyIntegral(mfem::Array<int> domain_attributes, BodyIntegralType body_integral, InputType integral_inputs,
                       OutputType integral_outputs, const mfem::IntegrationRule& integration_rule,
                       DerivIdsType derivative_ids)
  {
    residual_.AddDomainIntegrator(body_integral, integral_inputs, integral_outputs, integration_rule, domain_attributes,
                                  derivative_ids);
    auto scalar_body_integral = makeInnerQFunction(body_integral);
    // TODO: updated integral_inputs and integral_outputs for scalar_body_integral
    virtual_work_.AddDomainIntegrator(scalar_body_integral, integral_inputs,
                                      mfem::future::tuple<mfem::future::Sum<0>>{}, integration_rule, domain_attributes,
                                      derivative_ids);
  }

  /// @overload
  mfem::Vector residual(double, double, const std::vector<ConstFieldPtr>& fields, int block_row = 0) const override
  {
    SLIC_ERROR_ROOT_IF(block_row != 0, "Invalid block row and column requested in fieldJacobian for DfemResidual");
    mfem::Vector resid(output_mfem_space_.GetTrueVSize());
    resid = 0.0;
    residual_.SetParameters(getLVectors(fields));
    residual_.Mult(resid, resid);
    return resid;
  }

  /// @overload
  std::unique_ptr<mfem::HypreParMatrix> jacobian(double, double, const std::vector<ConstFieldPtr>&,
                                                 const std::vector<double>&, int) const override
  {
    SLIC_ERROR_ROOT("DfemResidual does not support matrix assembly");
    return std::make_unique<mfem::HypreParMatrix>();
  }

  /// @overload
  void jvp(double, double, const std::vector<ConstFieldPtr>& fields, const std::vector<ConstFieldPtr>& v_fields,
           const std::vector<DualFieldPtr>& jvp_reactions) const override
  {
    SLIC_ERROR_ROOT_IF(v_fields.size() != fields.size(),
                       "Invalid number of field sensitivities relative to the number of fields");
    SLIC_ERROR_ROOT_IF(jvp_reactions.size() != 1, "DfemResidual nonlinear systems only supports 1 output residual");

    *jvp_reactions[0] = 0.0;

    for (size_t input_col = 0; input_col < fields.size(); ++input_col) {
      if (v_fields[input_col] != nullptr) {
        auto deriv_op = residual_.GetDerivative(input_col, {&fields[0]->gridFunction()}, getLVectors(fields));
        deriv_op->AddMult(*v_fields[input_col], *jvp_reactions[0]);
      }
    }
  }

  /// @overload
  void vjp(double, double, const std::vector<ConstFieldPtr>& fields, const std::vector<ConstFieldPtr>& v_fields,
           const std::vector<DualFieldPtr>& vjp_sensitivities) const override
  {
    SLIC_ERROR_ROOT_IF(vjp_sensitivities.size() != fields.size(),
                       "Invalid number of field sensitivities relative to the number of fields");
    SLIC_ERROR_ROOT_IF(v_fields.size() != 1, "FunctionalResidual nonlinear systems only supports 1 output residual");

    for (size_t input_col = 0; input_col < fields.size(); ++input_col) {
      if (vjp_sensitivities[input_col] != nullptr) {
        // TODO: update to use the virtual_work_ operator once it's working right
        // auto grad_op = residual_.GetDerivative(input_col, {&fields[0]->gridFunction()}, getLVectors(fields));
        // grad_op->AddMultTranspose(*v_fields[0], *vjp_sensitivities[input_col]);
      }
    }
  }

 protected:
  static std::vector<mfem::future::FieldDescriptor> makeFieldDescriptors(
      const std::vector<const mfem::ParFiniteElementSpace*>& spaces, size_t offset = 0)
  {
    std::vector<mfem::future::FieldDescriptor> field_descriptors;
    field_descriptors.reserve(spaces.size());
    for (size_t i = 0; i < spaces.size(); ++i) {
      field_descriptors.emplace_back(i + offset, spaces[i]);
    }
    return field_descriptors;
  }

  std::vector<mfem::Vector*> getLVectors(const std::vector<ConstFieldPtr>& fields) const
  {
    std::vector<mfem::Vector*> fields_l;
    fields_l.reserve(fields.size());
    for (size_t i = 0; i < fields.size(); ++i) {
      fields_l.push_back(&fields[i]->gridFunction());
    }
    return fields_l;
  }

  /// @brief primary mesh
  std::shared_ptr<Mesh> mesh_;
  const mfem::ParFiniteElementSpace& output_mfem_space_;
  std::vector<const mfem::ParFiniteElementSpace*> input_mfem_spaces_;
  mutable mfem::future::DifferentiableOperator residual_;
  mutable mfem::future::DifferentiableOperator virtual_work_;
};

}  // namespace serac
