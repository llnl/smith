// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file weak_form_block_operator.hpp
 *
 * @brief Helpers for using Smith weak forms as block preconditioner operators.
 */

#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "mfem.hpp"

#include "smith/differentiable_numerics/field_state.hpp"
#include "smith/numerics/block_preconditioner.hpp"
#include "smith/physics/common.hpp"

namespace smith {

class BoundaryConditionManager;
class WeakForm;

/**
 * @brief Mapping from a nonlinear solve block to an input field of a weak form.
 */
struct StateBlockBinding {
  int block_index;  ///< Nonlinear solve block index in the monolithic state vector.
  int field_index;  ///< Weak-form input field index to update from the block.
};

/**
 * @brief Callable that rebuilds a weak-form operator from a nonlinear state.
 */
using StateDependentWeakFormOperator =
    std::function<std::unique_ptr<mfem::HypreParMatrix>(const mfem::Vector&, const mfem::Array<int>&)>;

/**
 * @brief Assemble a weak-form Jacobian operator for use in a block preconditioner.
 */
std::unique_ptr<mfem::HypreParMatrix> buildWeakFormOperator(const WeakForm& weak_form, FieldState shape_disp,
                                                            std::vector<FieldState> fields,
                                                            std::vector<double> jacobian_weights, TimeInfo time_info,
                                                            mfem::Array<int> ess_tdofs = mfem::Array<int>());

/**
 * @brief Assemble a weak-form Jacobian operator, eliminating essential dofs from a boundary-condition manager.
 */
std::unique_ptr<mfem::HypreParMatrix> buildWeakFormOperator(const WeakForm& weak_form, FieldState shape_disp,
                                                            std::vector<FieldState> fields,
                                                            std::vector<double> jacobian_weights, TimeInfo time_info,
                                                            const BoundaryConditionManager* bc_manager);

/**
 * @brief Build a callable that updates bound weak-form fields from state and assembles the weak-form operator.
 *
 * @note @a weak_form must outlive the returned callable.
 */
StateDependentWeakFormOperator makeStateDependentWeakFormOperator(const WeakForm& weak_form, FieldState shape_disp,
                                                                  std::vector<FieldState> fields,
                                                                  std::vector<double> jacobian_weights,
                                                                  TimeInfo time_info, mfem::Array<int> ess_tdofs,
                                                                  std::vector<StateBlockBinding> state_block_bindings);

/**
 * @brief Build a callable using essential dofs copied from a boundary-condition manager.
 *
 * @note @a weak_form must outlive the returned callable.
 */
StateDependentWeakFormOperator makeStateDependentWeakFormOperator(const WeakForm& weak_form, FieldState shape_disp,
                                                                  std::vector<FieldState> fields,
                                                                  std::vector<double> jacobian_weights,
                                                                  TimeInfo time_info,
                                                                  const BoundaryConditionManager* bc_manager,
                                                                  std::vector<StateBlockBinding> state_block_bindings);

/**
 * @brief Build a fixed block override from a weak-form Jacobian operator.
 */
BlockProviderOverride makeWeakFormBlockProviderOverride(int block_index, const WeakForm& weak_form,
                                                        FieldState shape_disp, std::vector<FieldState> fields,
                                                        std::vector<double> jacobian_weights, TimeInfo time_info,
                                                        mfem::Array<int> ess_tdofs = mfem::Array<int>());

/**
 * @brief Build a fixed block override using essential dofs copied from a boundary-condition manager.
 */
BlockProviderOverride makeWeakFormBlockProviderOverride(int block_index, const WeakForm& weak_form,
                                                        FieldState shape_disp, std::vector<FieldState> fields,
                                                        std::vector<double> jacobian_weights, TimeInfo time_info,
                                                        const BoundaryConditionManager* bc_manager);

/**
 * @brief Build a state-dependent block override from a weak-form Jacobian operator.
 *
 * @note @a weak_form must outlive the returned provider and every preconditioner
 *       to which that provider is supplied.
 */
BlockProviderOverride makeStateDependentWeakFormBlockProviderOverride(
    int block_index, const WeakForm& weak_form, FieldState shape_disp, std::vector<FieldState> fields,
    std::vector<double> jacobian_weights, TimeInfo time_info, mfem::Array<int> ess_tdofs,
    std::vector<StateBlockBinding> state_block_bindings);

/**
 * @brief Build a state-dependent block override using essential dofs copied from a boundary-condition manager.
 *
 * @note @a weak_form must outlive the returned provider and every preconditioner
 *       to which that provider is supplied.
 */
BlockProviderOverride makeStateDependentWeakFormBlockProviderOverride(
    int block_index, const WeakForm& weak_form, FieldState shape_disp, std::vector<FieldState> fields,
    std::vector<double> jacobian_weights, TimeInfo time_info, const BoundaryConditionManager* bc_manager,
    std::vector<StateBlockBinding> state_block_bindings);

/**
 * @brief Build a fixed assembled Schur operator from a weak form.
 *
 * The weak form is assembled once using the supplied fields and Jacobian weights.
 * Essential dofs are copied from @a bc_manager when it is non-null.
 *
 * @param weak_form Weak form whose Jacobian approximates the Schur complement.
 * @param shape_disp Mesh shape displacement used to assemble the weak form.
 * @param fields Fields at which the weak form is evaluated.
 * @param jacobian_weights Weights selecting the weak-form Jacobian contributions.
 * @param time_info Time information used to assemble the weak form.
 * @param bc_manager Optional boundary conditions for the Schur operator.
 */
SchurOperator weakFormSchur(const WeakForm& weak_form, FieldState shape_disp, std::vector<FieldState> fields,
                            std::vector<double> jacobian_weights, TimeInfo time_info,
                            const BoundaryConditionManager* bc_manager = nullptr);

/**
 * @brief Build a state-dependent Schur operator from a single-field weak form.
 *
 * The weak form is differentiated with respect to @a schur_field and refreshed
 * from block 1 of the current nonlinear state. Essential dofs are copied from
 * @a bc_manager when it is non-null.
 *
 * @note @a weak_form must outlive the returned operator and every preconditioner
 *       to which that operator is supplied.
 *
 * @param weak_form Weak form whose Jacobian approximates the Schur complement.
 * @param shape_disp Mesh shape displacement used to assemble the weak form.
 * @param schur_field Field updated from the Schur block at each nonlinear iterate.
 * @param time_info Time information used to assemble the weak form.
 * @param bc_manager Optional boundary conditions for the Schur field.
 */
SchurOperator weakFormSchur(const WeakForm& weak_form, FieldState shape_disp, FieldState schur_field,
                            TimeInfo time_info, const BoundaryConditionManager* bc_manager = nullptr);

}  // namespace smith
