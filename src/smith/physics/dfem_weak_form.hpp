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

#ifdef SMITH_USE_ENZYME

#include "smith/physics/weak_form.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/state/finite_element_state.hpp"
#include "mfem/fem/pbilinearform.hpp"
#include "smith/differentiable_numerics/dfem_reverse_derivative.hpp"
#include "mfem/fem/dfem/fielddescriptor.hpp"
#include "mfem/fem/dfem/fieldoperator.hpp"
#include "mfem/fem/dfem/doperator.hpp"
#include "mfem/fem/dfem/tuple.hpp"
#include "mfem/fem/dfem/backends/local_qf/prelude.hpp"

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
        weak_form_(makeFieldDescriptors(input_mfem_spaces),
                   makeFieldDescriptors({&output_mfem_space}, input_mfem_spaces.size()), mesh->mfemParMesh()),
        dummy_sensitivities_(input_mfem_spaces.size()),
        residual_vector_(output_mfem_space.GetTrueVSize())
  {
    residual_vector_.UseDevice(true);
    for (auto& v : dummy_sensitivities_) {
      v.UseDevice(true);
    }
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
    weak_form_.AddDomainIntegrator<mfem::future::LocalQFBackend>(body_integral, integral_inputs, integral_outputs,
                                                                 integration_rule, domain_attributes, derivative_ids);

    if constexpr (derivative_ids.size() > 0) {
      reverse_integrator_adders_.push_back([=, domain_attributes = std::move(domain_attributes)](
                                               mfem::future::DifferentiableOperator& rev_dop) {
        mfem::future::reverse::AddReverseGradientIntegrator<V_ID>(rev_dop, body_integral, integral_inputs,
                                                                  integral_outputs, integration_rule, domain_attributes,
                                                                  derivative_ids);
      });
    }
  }

  /**
   * @brief Add a boundary integral contribution to the residual
   *
   * Reverse-mode support is currently only wired for domain integrals.
   */
  template <typename BoundaryIntegralType, typename InputType, typename OutputType, typename DerivIdsType>
  void addBoundaryIntegral(mfem::Array<int> boundary_attributes, BoundaryIntegralType boundary_integral,
                           InputType integral_inputs, OutputType integral_outputs,
                           const mfem::IntegrationRule& integration_rule, DerivIdsType derivative_ids)
  {
    weak_form_.AddBoundaryIntegrator<mfem::future::LocalQFBackend>(boundary_integral, integral_inputs,
                                                                   integral_outputs, integration_rule,
                                                                   boundary_attributes, derivative_ids);

    if constexpr (derivative_ids.size() > 0) {
      reverse_integrator_adders_.push_back([=, boundary_attributes = std::move(boundary_attributes)](
                                               mfem::future::DifferentiableOperator& rev_dop) {
        mfem::future::reverse::AddReverseBoundaryGradientIntegrator<V_ID>(
            rev_dop, boundary_integral, integral_inputs,
            integral_outputs, integration_rule, boundary_attributes,
            derivative_ids);
      });
    }
  }

  /// @overload
  mfem::Vector residual(TimeInfo time_info, ConstFieldPtr /*shape_disp*/, const std::vector<ConstFieldPtr>& fields,
                        const std::vector<ConstQuadratureFieldPtr>& /*quad_fields*/ = {}) const override
  {
    dt_ = time_info.dt();
    cycle_ = time_info.cycle();
    current_time_ = time_info.time();

    auto fields_t = getTMultiVector(fields);
    mfem::MultiVector residual_t;
    residual_t.MakeRef(residual_vector_);
    residual_vector_ = 0.0;
    weak_form_.Mult(fields_t, residual_t);
    return residual_vector_;
  }

  /// @overload
  std::unique_ptr<mfem::HypreParMatrix> jacobian(
      TimeInfo time_info, ConstFieldPtr /*shape_disp*/, const std::vector<ConstFieldPtr>& fields,
      const std::vector<double>& jacobian_weights,
      const std::vector<ConstQuadratureFieldPtr>& /*quad_fields*/ = {}) const override
  {
    dt_ = time_info.dt();
    cycle_ = time_info.cycle();
    current_time_ = time_info.time();

    std::unique_ptr<mfem::HypreParMatrix> J;

    SLIC_ERROR_IF(jacobian_weights.size() != fields.size(),
                  "Invalid number of jacobian weights relative to the number of fields");

    for (size_t input_col = 0; input_col < jacobian_weights.size(); ++input_col) {
      if (jacobian_weights[input_col] != 0.0) {
        auto fields_t = getTMultiVector(fields);
        auto deriv_op = weak_form_.GetDerivative(input_col, fields_t);
        mfem::SparseMatrix* jac_local_ptr = nullptr;
        deriv_op->Assemble(jac_local_ptr);
        if (jac_local_ptr == nullptr) {
          jac_local_ptr = new mfem::SparseMatrix(deriv_op->Height(), deriv_op->Width());
          jac_local_ptr->Finalize();
        }
        std::unique_ptr<mfem::SparseMatrix> jac_local(jac_local_ptr);
        int num_ranks = 1;
        MPI_Comm_size(output_mfem_space_.GetComm(), &num_ranks);
        SLIC_ERROR_IF(num_ranks != 1, "Prototype DfemWeakForm jacobian assembly is currently serial only.");
        const int nnz = jac_local->NumNonZeroElems();
        std::vector<HYPRE_BigInt> global_j(static_cast<size_t>(nnz));
        const int* local_j = jac_local->GetJ();
        const auto* col_starts = input_mfem_spaces_[input_col]->GetTrueDofOffsets();
        for (int k = 0; k < nnz; ++k) {
          global_j[static_cast<size_t>(k)] = col_starts[0] + local_j[k];
        }
        auto* jac_hypre =
            new mfem::HypreParMatrix(output_mfem_space_.GetComm(), jac_local->Height(),
                                     output_mfem_space_.GlobalTrueVSize(), input_mfem_spaces_[input_col]->GlobalTrueVSize(),
                                     jac_local->GetI(), global_j.data(), jac_local->GetData(),
                                     output_mfem_space_.GetTrueDofOffsets(), col_starts);
        if (J) {
          SLIC_ERROR_IF(J->N() != jac_hypre->N(),
                        "Multiple nonzero jacobian weights are being used on inconsistently sized input arguments.");
          SLIC_ERROR_IF(J->M() != jac_hypre->M(),
                        "Multiple nonzero jacobian weights are being used on inconsistently sized input arguments.");
          J->Add(jacobian_weights[input_col], *jac_hypre);
          delete jac_hypre;
        } else {
          J.reset(jac_hypre);
          if (jacobian_weights[input_col] != 1.0) {
            (*J) *= jacobian_weights[input_col];
          }
        }
      }
    }

    return J;
  }

  /// @overload
  void jvp(TimeInfo time_info, ConstFieldPtr /*shape_disp*/, const std::vector<ConstFieldPtr>& fields,
           const std::vector<ConstQuadratureFieldPtr>& /*quad_fields*/, ConstFieldPtr /*v_shape_disp*/,
           const std::vector<ConstFieldPtr>& v_fields,
           const std::vector<ConstQuadratureFieldPtr>& /*v_quad_fields*/, DualFieldPtr jvp_reaction) const override
  {
    dt_ = time_info.dt();
    cycle_ = time_info.cycle();

    SLIC_ERROR_IF(v_fields.size() != fields.size(),
                  "Invalid number of field sensitivities relative to the number of fields");

    auto* reaction = static_cast<mfem::Vector*>(jvp_reaction);
    *reaction = 0.0;
    mfem::Vector contribution(reaction->Size());
    contribution.UseDevice(true);

    for (size_t input_col = 0; input_col < fields.size(); ++input_col) {
      if (v_fields[input_col] != nullptr) {
        auto fields_t = getTMultiVector(fields);
        auto deriv_op = weak_form_.GetDerivative(input_col, fields_t);
        contribution = 0.0;
        deriv_op->Mult(*v_fields[input_col], contribution);
        reaction->Add(1.0, contribution);
      }
    }
  }

  /// @overload
  void vjp(TimeInfo time_info, ConstFieldPtr /*shape_disp*/, const std::vector<ConstFieldPtr>& fields,
           const std::vector<ConstQuadratureFieldPtr>& /*quad_fields*/, ConstFieldPtr v_field,
           DualFieldPtr vjp_shape_disp_sensitivity, const std::vector<DualFieldPtr>& vjp_sensitivities,
           const std::vector<QuadratureFieldPtr>& /*vjp_quad_field_sensitivities*/) const override
  {
    dt_ = time_info.dt();
    cycle_ = time_info.cycle();

    SLIC_ERROR_IF(vjp_sensitivities.size() != fields.size(),
                  "Invalid number of field sensitivities relative to the number of fields");

    if (vjp_shape_disp_sensitivity) {
      *vjp_shape_disp_sensitivity = 0.0;
    }

    if (!reverse_weak_form_initialized_) {
      auto rev_infds = makeFieldDescriptors(input_mfem_spaces_);
      rev_infds.emplace_back(V_ID, &output_mfem_space_);

      std::vector<mfem::future::FieldDescriptor> rev_outfds;
      for (size_t i = 0; i < input_mfem_spaces_.size(); ++i) {
        rev_outfds.emplace_back(i, input_mfem_spaces_[i]);
      }

      reverse_weak_form_ = std::make_unique<mfem::future::DifferentiableOperator>(rev_infds, rev_outfds, mesh_->mfemParMesh());
      for (const auto& adder : reverse_integrator_adders_) {
        adder(*reverse_weak_form_);
      }
      reverse_weak_form_initialized_ = true;
    }

    mfem::MultiVector X(static_cast<int>(fields.size()) + 1);
    for (size_t i = 0; i < fields.size(); ++i) {
      X.MakeRef(static_cast<int>(i), const_cast<mfem::Vector&>(static_cast<const mfem::Vector&>(*fields[i])));
    }
    X.MakeRef(static_cast<int>(fields.size()), const_cast<mfem::Vector&>(static_cast<const mfem::Vector&>(*v_field)));

    mfem::MultiVector Y(static_cast<int>(fields.size()));
    for (size_t i = 0; i < fields.size(); ++i) {
      if (vjp_sensitivities[i]) {
        *vjp_sensitivities[i] = 0.0;
        Y.MakeRef(static_cast<int>(i), *vjp_sensitivities[i]);
      } else {
        if (dummy_sensitivities_[i].Size() != input_mfem_spaces_[i]->GetTrueVSize()) {
          dummy_sensitivities_[i].SetSize(input_mfem_spaces_[i]->GetTrueVSize());
          dummy_sensitivities_[i].UseDevice(true);
        }
        dummy_sensitivities_[i] = 0.0;
        Y.MakeRef(static_cast<int>(i), dummy_sensitivities_[i]);
      }
    }

    reverse_weak_form_->Mult(X, Y);
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

  mfem::MultiVector getTMultiVector(const std::vector<ConstFieldPtr>& fields) const
  {
    mfem::MultiVector fields_t(static_cast<int>(fields.size()));
    for (size_t i = 0; i < fields.size(); ++i) {
      fields_t.MakeRef(static_cast<int>(i), const_cast<mfem::Vector&>(static_cast<const mfem::Vector&>(*fields[i])));
    }
    return fields_t;
  }

  static std::vector<double> oneHotWeights(size_t num_fields, size_t active_field)
  {
    std::vector<double> weights(num_fields, 0.0);
    weights[active_field] = 1.0;
    return weights;
  }

  std::unique_ptr<mfem::SparseMatrix> assembleJacobianFromAction(const mfem::future::DerivativeOperator& deriv_op) const
  {
    const int num_rows = deriv_op.Height();
    const int num_cols = deriv_op.Width();
    auto jac = std::make_unique<mfem::SparseMatrix>(num_rows, num_cols);

    mfem::Vector ej(num_cols);
    mfem::Vector col(num_rows);
    ej.UseDevice(false);
    col.UseDevice(false);

    constexpr double zero_tol = 1.0e-14;
    for (int j = 0; j < num_cols; ++j) {
      ej = 0.0;
      ej(j) = 1.0;
      col = 0.0;
      mfem::MultiVector col_mv{col};
      deriv_op.Mult(ej, col_mv);
      for (int i = 0; i < num_rows; ++i) {
        if (std::abs(col(i)) > zero_tol) {
          jac->Add(i, j, col(i));
        }
      }
    }
    jac->Finalize();
    return jac;
  }

  /// @brief Field ID for the seed field V in reverse mode
  static constexpr int V_ID = 999;

  /// @brief timestep, this needs to be held here and modified for rate dependent applications
  mutable double dt_ = std::numeric_limits<double>::max();

  /// @brief cycle or step or iteration.  This counter is useful for certain time integrators.
  mutable size_t cycle_ = 0;

  /// @brief current time (end of step), updated at the top of residual()/jacobian().
  /// Available via currentTimePtr() so derived classes can wire time-dependent
  /// boundary integrators (e.g. tractions) without re-registering a qfunction each call.
  mutable double current_time_ = 0.0;

 public:
  /// @brief Pointer to the current-time scalar, valid for the lifetime of *this.
  /// Intended for use by derived classes / qfunctions that need time inside the
  /// quadrature kernel (e.g. ramped tractions).
  const double* currentTimePtr() const { return &current_time_; }

 protected:

  /// @brief primary mesh
  std::shared_ptr<Mesh> mesh_;

  /// @brief Output field (test) space
  const mfem::ParFiniteElementSpace& output_mfem_space_;

  /// @brief Input field (trial) spaces
  std::vector<const mfem::ParFiniteElementSpace*> input_mfem_spaces_;

  /// @brief dfem residual evaluator
  mutable mfem::future::DifferentiableOperator weak_form_;

  /// @brief reverse-mode dfem evaluator for VJP
  mutable std::unique_ptr<mfem::future::DifferentiableOperator> reverse_weak_form_;

  /// @brief flag to indicate if reverse_weak_form_ has been initialized
  mutable bool reverse_weak_form_initialized_ = false;

  /// @brief list of reverse integrator adders
  std::vector<std::function<void(mfem::future::DifferentiableOperator&)>> reverse_integrator_adders_;

  /// @brief dummy sensitivities for fields where no sensitivity is requested
  mutable std::vector<mfem::Vector> dummy_sensitivities_;

  /// @brief residual vector
  mutable mfem::Vector residual_vector_;
};

}  // namespace smith

#endif
