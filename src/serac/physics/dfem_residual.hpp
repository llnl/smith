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
template <typename InputType, typename OutputType>
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
        differentiable_operators_(input_mfem_spaces.size() + 1),
        output_mfem_space_(output_mfem_space),
        input_mfem_spaces_(input_mfem_spaces)
  {
    std::vector<const mfem::ParFiniteElementSpace*> parameter_spaces;
    parameter_spaces.reserve(input_mfem_spaces.size());
    for (size_t i = 1; i < input_mfem_spaces.size(); ++i) {
      parameter_spaces.push_back(input_mfem_spaces[i]);
    }
    // TODO: look into different test and trial FE spaces
    // parameter_spaces.push_back(&output_mfem_space);
    differentiable_operators_[0] = std::make_unique<mfem::future::DifferentiableOperator>(
        makeFieldDescriptors({input_mfem_spaces[0]}), makeFieldDescriptors(parameter_spaces, 1), mesh->mfemParMesh());
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
  template <int... active_inputs, typename BodyIntegralType>
  void addBodyIntegral(DependsOn<active_inputs...>, mfem::Array<int> domain_attributes, BodyIntegralType body_integral,
                       const mfem::IntegrationRule& integration_rule, std::integer_sequence<size_t, 0> derivative_ids)
  {
    InputType integral_inputs;
    OutputType integral_outputs;
    // ParameterReducer<BodyIntegralType, active_inputs...> reduced_integral(body_integral);
    differentiable_operators_[0]->AddDomainIntegrator(body_integral, integral_inputs, integral_outputs,
                                                      integration_rule, domain_attributes, derivative_ids);
  }

  /// @overload
  mfem::Vector residual(double, double, const std::vector<ConstFieldPtr>& fields, int block_row = 0) const override
  {
    SLIC_ERROR_ROOT_IF(block_row != 0, "Invalid block row and column requested in fieldJacobian for DfemResidual");
    mfem::Vector resid(fields[0]->space().GetTrueVSize());
    resid = 0.0;
    std::vector<mfem::Vector*> other_fields;
    other_fields.reserve(fields.size() - 1);
    for (size_t i = 1; i < fields.size(); ++i) {
      other_fields.push_back(&fields[i]->gridFunction());
    }
    differentiable_operators_[0]->SetParameters(other_fields);
    differentiable_operators_[0]->Mult(*fields[0], resid);
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

    // *jvp_reactions[0] = 0.0;

    // for (size_t input_col = 0; input_col < fields.size(); ++input_col) {
    //   if (v_fields[input_col] == nullptr) {
    //     continue;
    //   }
    //   for (auto& residual_term_with_field_ids : residual_terms_with_field_ids_) {
    //     const auto& field_ids = std::get<1>(residual_term_with_field_ids);
    //     auto field_id_it = std::find(field_ids.begin(), field_ids.end(), input_col);
    //     // this residual term does not depend on this input field
    //     if (field_id_it == field_ids.end()) {
    //       continue;
    //     }
    //   if (field_ids[0] != input_col) {
    //     continue;
    //   }
    //   auto& residual_term_ = std::get<0>(residual_term_with_field_ids);
    //   std::vector<mfem::Vector*> other_fields;
    //   other_fields.reserve(field_ids.size() - 1);
    //   for (size_t i = 1; i < field_ids.size(); ++i) {
    //     other_fields.push_back(&fields[field_ids[i]]->gridFunction());
    //   }
    //   residual_term_.SetParameters(other_fields);
    //   mfem::Vector K(fields[0]->space().GetTrueVSize());
    //   residual_term_.AddMult(*v_fields[input_col], K);
    //   (*jvp_reactions[0]) += K;
    // }
    // // find the residual term that corresponds to this input field
    // auto it = std::find_if(residual_terms_with_field_ids_.begin(), residual_terms_with_field_ids_.end(),
    //                        [&](const auto& term_and_ids) {
    //                          const auto& field_ids = std::get<1>(term_and_ids);
    //                          return field_ids[0] == input_col;
    //                        });
    // if (it == residual_terms_with_field_ids_.end()) {
    //   continue;
    // }
    // const auto& residual_term_ = std::get<0>(*it);
    // std::vector<mfem::Vector*> other_fields;
    // const auto& field_ids = std::get<1>(*it);
    // other_fields.reserve(field_ids.size() - 1);
    // for (size_t i = 1; i < field_ids.size(); ++i) {
    //   other_fields.push_back(&fields[field_ids[i]]->gridFunction());
    // }
    // residual_term_.SetParameters(other_fields);
    //   }
    // }
  }

  /// @overload
  void vjp(double, double, const std::vector<ConstFieldPtr>& fields, const std::vector<ConstFieldPtr>& v_fields,
           const std::vector<DualFieldPtr>& vjp_sensitivities) const override
  {
    SLIC_ERROR_ROOT_IF(vjp_sensitivities.size() != fields.size(),
                       "Invalid number of field sensitivities relative to the number of fields");
    SLIC_ERROR_ROOT_IF(v_fields.size() != 1, "FunctionalResidual nonlinear systems only supports 1 output residual");

    // auto grad_op = residual_term_.GetDerivative(0, {fields[0]}, {mesh_nodes_});
    // grad_op->AddMultTranspose(*v_fields[0], *vjp_sensitivities[0]);
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

  /// @brief primary mesh
  std::shared_ptr<Mesh> mesh_;
  std::vector<std::unique_ptr<mfem::future::DifferentiableOperator>> differentiable_operators_;
  const mfem::ParFiniteElementSpace& output_mfem_space_;
  std::vector<const mfem::ParFiniteElementSpace*> input_mfem_spaces_;

 private:
  template <typename Integrand, int dim, int... active_inputs>
  struct ParameterReducer {
    ParameterReducer(Integrand integrand) : integrand_(integrand) {}

    template <typename... Args>
    SERAC_HOST_DEVICE inline auto operator()(Args&&... args) const
    {
      auto arg_tuple = mfem::future::make_tuple(std::forward<Args>(args)...);
      return integrand_(std::get<active_inputs>(arg_tuple)...);
    }

    SERAC_HOST_DEVICE inline auto operator()(const mfem::future::tensor<mfem::real_t, dim, dim>& du_dxi,
                                             const mfem::future::tensor<mfem::real_t, dim, dim>& dv_dxi,
                                             const mfem::future::tensor<mfem::real_t, dim, dim>& da_dxi,
                                             const mfem::future::tensor<mfem::real_t, dim, dim>& dX_dxi,
                                             mfem::real_t weight, double K) const
    {
      auto arg_tuple = mfem::future::make_tuple(du_dxi, dv_dxi, da_dxi, dX_dxi, weight, K);
      return integrand_(std::get<active_inputs>(arg_tuple)...);
    }

   private:
    Integrand integrand_;
  };
};

}  // namespace serac
