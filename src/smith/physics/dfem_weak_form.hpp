// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file dfem_weak_form.hpp
 *
 * @brief Implements the WeakForm interface using dfem. Allows for generic specification of body and boundary integrals
 */

#pragma once

#include "smith/smith_config.hpp"

#ifdef SMITH_USE_DFEM

#include "smith/physics/weak_form.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/state/finite_element_state.hpp"

// NOTE (EBC): these should be upstreamed to MFEM, so let's put them in the mfem::future namespace
namespace mfem {
namespace future {

// Inner product of 1D tensors
template <typename S, typename T, int m>
MFEM_HOST_DEVICE auto inner(const tensor<S, m>& A, const tensor<T, m>& B) -> decltype(S{} * T{})
{
  decltype(S{} * T{}) sum{};
  for (int i = 0; i < m; i++) {
    sum += A[i] * B[i];
  }
  return sum;
}

/**
 * @brief computes det(A + I) - 1, where precision is not lost when the entries A_{ij} << 1
 *
 * detApIm1(A) = det(A + I) - 1
 * When the entries of A are small compared to unity, computing
 * det(A + I) - 1 directly will suffer from catastrophic cancellation.
 *
 * @param A Input matrix
 * @return det(A + I) - 1, where I is the identity matrix
 */
template <typename T>
MFEM_HOST_DEVICE constexpr auto detApIm1(const mfem::future::tensor<T, 2, 2>& A)
{
  // From the Cayley-Hamilton theorem, we get that for any N by N matrix A,
  // det(A - I) - 1 = I1(A) + I2(A) + ... + IN(A),
  // where the In are the principal invariants of A.
  // We inline the definitions of the principal invariants to increase computational speed.

  // equivalent to tr(A) + det(A)
  return A(0, 0) - A(0, 1) * A(1, 0) + A(1, 1) + A(0, 0) * A(1, 1);
}

/// @overload
template <typename T>
MFEM_HOST_DEVICE constexpr auto detApIm1(const mfem::future::tensor<T, 3, 3>& A)
{
  // For notes on the implementation, see the 2x2 version.

  // clang-format off
  // equivalent to tr(A) + I2(A) + det(A)
  return A(0, 0) + A(1, 1) + A(2, 2) 
       - A(0, 1) * A(1, 0) * (1 + A(2, 2))
       + A(0, 0) * A(1, 1) * (1 + A(2, 2))
       - A(0, 2) * A(2, 0) * (1 + A(1, 1))
       - A(1, 2) * A(2, 1) * (1 + A(0, 0))
       + A(0, 0) * A(2, 2)
       + A(1, 1) * A(2, 2)
       + A(0, 1) * A(1, 2) * A(2, 0)
       + A(0, 2) * A(1, 0) * A(2, 1);
  // clang-format on
}

}  // namespace future
}  // namespace mfem

namespace smith {

// NOTE: Args needs to be on the functor struct instead of the operator() so that operator() isn't overloaded and dfem
// can deduce the type
/**
 * @brief Helper struct to build a (typically scalar) q-function as the inner product of an existing q-function output
 * with a given tensor of the same type
 */
template <typename Primal, typename OrigQFn, typename... Args>
struct InnerQFunction {
  InnerQFunction(OrigQFn orig_qfn) : orig_qfn_(orig_qfn) {}

  SMITH_HOST_DEVICE inline auto operator()(Primal V, Args... args) const
  {
    Primal orig_residual = mfem::future::get<0>(orig_qfn_(std::forward<Args>(args)...));
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
 * @brief A nonlinear WeakForm class implemented using dfem
 *
 * This uses dfem to compute fairly general residuals and tangent stiffness matrices based on body and boundary weak
 * form integrals.
 *
 */
class DfemWeakForm : public WeakForm {
 public:
  using SpacesT = std::vector<const mfem::ParFiniteElementSpace*>;  ///< typedef

  /**
   * @brief Construct a new DfemWeakForm object
   *
   * @param physics_name A name for the physics module instance
   * @param mesh The smith mesh
   * @param output_mfem_space Test space
   * @param input_mfem_spaces Vector of finite element spaces which are arguments to the residual
   */
  DfemWeakForm(std::string physics_name, std::shared_ptr<Mesh> mesh,
               const mfem::ParFiniteElementSpace& output_mfem_space, const SpacesT& input_mfem_spaces)
      : WeakForm(physics_name),
        mesh_(mesh),
        output_mfem_space_(output_mfem_space),
        input_mfem_spaces_(input_mfem_spaces),
        weak_form_(makeFieldDescriptors({&output_mfem_space}, input_mfem_spaces.size()),
                   makeFieldDescriptors(input_mfem_spaces), mesh->mfemParMesh()),
        v_dot_weak_form_residual_(makeFieldDescriptors({&output_mfem_space}, input_mfem_spaces.size()),
                                  makeFieldDescriptors(input_mfem_spaces), mesh->mfemParMesh()),
        residual_vector_(output_mfem_space.GetTrueVSize())
  {
    // sum field operator doesn't work with sum factorization
    v_dot_weak_form_residual_.DisableTensorProductStructure();
    residual_vector_.UseDevice(true);
  }

  /**
   * @brief Add a body integral contribution to the residual
   *
   * @tparam BodyIntegralType The type of the body integral
   * @tparam InputType mfem::future::tuple holding mfem::future::FieldOperator of the body integral inputs
   * @tparam OutputType mfem::future::tuple holding the single mfem::future::FieldOperator of the body integral output
   * @tparam DerivIdsType std::index_sequence of field IDs where derivatives are needed
   * @param domain_attributes Array of MFEM element attributes over which to compute the integral
   * @param body_integral A function describing the body force applied.  Our convention for the sign of the residual
   * vector is that it is expected to be a 'negative force', so the mass terms show up with a positive sign in the
   * residual.  This also ensures that the Jacobian of the residual is positive definite for most physics.  A body
   * integrand involving 'right hand side' contributions like a body load, should be supplied by the user with a
   * negative sign.
   * @param integral_inputs Empty InputType
   * @param integral_outputs Empty OutputType
   * @param integration_rule Integration rule to use on each element
   * @param derivative_ids Empty DerivIdsType
   *
   * @pre body_integral must take all fields as either an InputType or an OutputType
   *
   */
  template <typename BodyIntegralType, typename InputType, typename OutputType, typename DerivIdsType>
  void addBodyIntegral(mfem::Array<int> domain_attributes, BodyIntegralType body_integral, InputType integral_inputs,
                       OutputType integral_outputs, const mfem::IntegrationRule& integration_rule,
                       DerivIdsType derivative_ids)
  {
    weak_form_.AddDomainIntegrator(body_integral, integral_inputs, integral_outputs, integration_rule,
                                   domain_attributes, derivative_ids);
    auto scalar_body_integral = makeInnerQFunction(body_integral);
    v_dot_weak_form_residual_.AddDomainIntegrator(
        scalar_body_integral, addToTupleType(integral_inputs, mfem::future::get<0>(integral_outputs)),
        mfem::future::tuple<mfem::future::Sum<mfem::future::get<0>(integral_outputs).GetFieldId()>>{}, integration_rule,
        domain_attributes, derivative_ids);
  }

  /// @overload
  mfem::Vector residual(TimeInfo time_info, ConstFieldPtr /*shape_disp*/, const std::vector<ConstFieldPtr>& fields,
                        const std::vector<ConstQuadratureFieldPtr>& /*quad_fields*/ = {}) const override
  {
    dt_ = time_info.dt();
    cycle_ = time_info.cycle();

    weak_form_.SetParameters(getLVectors(fields));
    weak_form_.Mult(residual_vector_, residual_vector_);
    return residual_vector_;
  }

  /// @overload
  std::unique_ptr<mfem::HypreParMatrix> jacobian(
      TimeInfo time_info, ConstFieldPtr /*shape_disp*/, const std::vector<ConstFieldPtr>& /*fields*/,
      const std::vector<double>& /*jacobian_weights*/,
      const std::vector<ConstQuadratureFieldPtr>& /*quad_fields*/ = {}) const override
  {
    SLIC_ERROR_ROOT("DfemWeakForm does not support matrix assembly");
    dt_ = time_info.dt();
    cycle_ = time_info.cycle();

    return std::make_unique<mfem::HypreParMatrix>();
  }

  /// @overload
  void jvp(TimeInfo time_info, ConstFieldPtr /*shape_disp*/, const std::vector<ConstFieldPtr>& /*fields*/,
           const std::vector<ConstQuadratureFieldPtr>& /*quad_fields*/, ConstFieldPtr /*v_shape_disp*/,
           const std::vector<ConstFieldPtr>& /*v_fields*/,
           const std::vector<ConstQuadratureFieldPtr>& /*v_quad_fields*/, DualFieldPtr /*jvp_reaction*/) const override
  {
    SLIC_ERROR_ROOT("DfemWeakForm does not support jvp calculations");

    // SLIC_ERROR_IF(v_fields.size() != fields.size(),
    //               "Invalid number of field sensitivities relative to the number of fields");
    // SLIC_ERROR_IF(jvp_reactions.size() != 1, "FunctionalResidual nonlinear systems only supports 1 output residual");
    dt_ = time_info.dt();
    cycle_ = time_info.cycle();

    // TODO (EBC): add in a future PR...
    // std::vector<mfem::Vector*> test_par_gf({&fields[0]->gridFunction()});
    // std::vector<mfem::Vector*> field_par_gf = getLVectors(fields);

    // *jvp_reactions[0] = 0.0;

    // for (size_t input_col = 0; input_col < fields.size(); ++input_col) {
    //   if (v_fields[input_col] != nullptr) {
    //     auto deriv_op = weak_form_.GetDerivative(input_col, test_par_gf, field_par_gf);
    //     deriv_op->AddMult(*v_fields[input_col], *jvp_reactions[0]);
    //   }
    // }
  }

  /// @overload
  void vjp(TimeInfo time_info, ConstFieldPtr /*shape_disp*/, const std::vector<ConstFieldPtr>& /*fields*/,
           const std::vector<ConstQuadratureFieldPtr>& /*quad_fields*/, ConstFieldPtr /*v_fields*/,
           DualFieldPtr /*vjp_shape_disp_sensitivity*/, const std::vector<DualFieldPtr>& /*vjp_sensitivities*/,
           const std::vector<QuadratureFieldPtr>& /*vjp_quad_field_sensitivities*/) const override
  {
    SLIC_ERROR_ROOT("DfemWeakForm does not support vjp calculations");

    // SLIC_ERROR_IF(vjp_sensitivities.size() != fields.size(),
    //               "Invalid number of field sensitivities relative to the number of fields");
    // SLIC_ERROR_IF(v_fields.size() != 1, "FunctionalResidual nonlinear systems only supports 1 output residual");
    dt_ = time_info.dt();
    cycle_ = time_info.cycle();

    // TODO (EBC): add in a future PR...
    // std::vector<mfem::Vector*> test_par_gf({&v_fields[0]->gridFunction()});
    // std::vector<mfem::Vector*> field_par_gf = getLVectors(fields);
    // // field_par_gf.push_back(&v_fields[0]->gridFunction());

    // for (size_t input_col = 0; input_col < fields.size(); ++input_col) {
    //   if (vjp_sensitivities[input_col] != nullptr) {
    //     auto deriv_op = v_dot_weak_form_residual_.GetDerivative(input_col, test_par_gf, field_par_gf);
    //     // do this entry by entry until assembly is supported
    //     mfem::Vector direction(vjp_sensitivities[input_col]->Size());
    //     direction = 0.0;
    //     for (int i = 0; i < vjp_sensitivities[input_col]->Size(); ++i) {
    //       direction[i] = 1.0;
    //       mfem::Vector value(1);
    //       deriv_op->Mult(direction, value);
    //       (*vjp_sensitivities[input_col])[i] += value[0];
    //       direction[i] = 0.0;
    //     }
    //   }
    // }
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

  template <typename Tnew, typename... Ttuple>
  static auto addToTupleType(const mfem::future::tuple<Ttuple...>&, const Tnew&)
  {
    return mfem::future::tuple<Tnew, Ttuple...>{};
  }

  // The field ID doesn't matter, since the test function is one
  template <int Id, template <int> class FieldOp>
  static auto makeVirtualWorkOutputs(mfem::future::tuple<FieldOp<Id>>)
  {
    return mfem::future::tuple<mfem::future::Sum<Id>>{};
  }

  /// @brief timestep, this needs to be held here and modified for rate dependent applications
  mutable double dt_ = std::numeric_limits<double>::max();

  /// @brief cycle or step or iteration.  This counter is useful for certain time integrators.
  mutable size_t cycle_ = 0;

  /// @brief primary mesh
  std::shared_ptr<Mesh> mesh_;

  /// @brief Output field (test) space
  const mfem::ParFiniteElementSpace& output_mfem_space_;

  /// @brief Input field (trial) spaces
  std::vector<const mfem::ParFiniteElementSpace*> input_mfem_spaces_;

  /// @brief dfem residual evaluator
  mutable mfem::future::DifferentiableOperator weak_form_;

  /// @brief dfem residual times an arbitrary vector v (same space as residual) evaluator
  mutable mfem::future::DifferentiableOperator v_dot_weak_form_residual_;

  /// @brief residual vector
  mutable mfem::Vector residual_vector_;
};

}  // namespace smith

#endif
