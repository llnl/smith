// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "smith/differentiable_numerics/weak_form_block_operator.hpp"

#include <stdexcept>
#include <utility>

#include "smith/physics/boundary_conditions/boundary_condition_manager.hpp"
#include "smith/physics/weak_form.hpp"

namespace smith {

namespace {

mfem::Array<int> copyEssentialTrueDofs(const BoundaryConditionManager* bc_manager)
{
  if (!bc_manager) {
    return mfem::Array<int>();
  }
  return bc_manager->allEssentialTrueDofs();
}

class WeakFormBlockOperatorBuilder {
 public:
  WeakFormBlockOperatorBuilder(const WeakForm& weak_form, FieldState shape_disp, std::vector<FieldState> fields,
                               std::vector<double> jacobian_weights, TimeInfo time_info, mfem::Array<int> ess_tdofs,
                               std::vector<StateBlockBinding> state_block_bindings)
      : weak_form_(weak_form),
        shape_disp_(std::move(shape_disp)),
        fields_(std::move(fields)),
        jacobian_weights_(std::move(jacobian_weights)),
        time_info_(time_info),
        ess_tdofs_(std::move(ess_tdofs)),
        state_block_bindings_(std::move(state_block_bindings))
  {
    validate();
  }

  std::unique_ptr<mfem::HypreParMatrix> build() const
  {
    auto op =
        weak_form_.jacobian(time_info_, shape_disp_.get().get(), getConstFieldPointers(fields_), jacobian_weights_);
    if (!op) {
      throw std::invalid_argument("Weak-form operator builder received a null weak-form Jacobian");
    }
    eliminateEssentialDofs(*op);
    return op;
  }

  std::unique_ptr<mfem::HypreParMatrix> updateAndBuild(const mfem::Vector& state, const mfem::Array<int>& block_offsets)
  {
    updateFieldsFromState(state, block_offsets);
    return build();
  }

 private:
  void validate() const
  {
    if (jacobian_weights_.size() != fields_.size()) {
      throw std::invalid_argument("Weak-form operator jacobian_weights size must match fields size");
    }
    for (const auto& binding : state_block_bindings_) {
      if (binding.block_index < 0) {
        throw std::invalid_argument("Weak-form operator state block index must be non-negative");
      }
      if (binding.field_index < 0 || binding.field_index >= static_cast<int>(fields_.size())) {
        throw std::invalid_argument("Weak-form operator field index is out of range");
      }
    }
  }

  void updateFieldsFromState(const mfem::Vector& state, const mfem::Array<int>& block_offsets)
  {
    for (const auto& binding : state_block_bindings_) {
      MFEM_VERIFY(binding.block_index + 1 < block_offsets.Size(), "Weak-form operator state block is out of range");

      const int block_begin = block_offsets[binding.block_index];
      const int block_size = block_offsets[binding.block_index + 1] - block_begin;
      MFEM_VERIFY(block_begin >= 0 && block_size >= 0 && block_begin + block_size <= state.Size(),
                  "Weak-form operator block offsets are inconsistent with the state size");

      FieldState& field = fields_[static_cast<size_t>(binding.field_index)];
      MFEM_VERIFY(field.get()->Size() == block_size,
                  "Weak-form operator cannot update a field from a block with incompatible size");

      mfem::Vector block_view;
      block_view.MakeRef(const_cast<mfem::Vector&>(state), block_begin, block_size);
      *field.get() = block_view;
    }
  }

  void eliminateEssentialDofs(mfem::HypreParMatrix& op) const
  {
    if (ess_tdofs_.Size() == 0) {
      return;
    }
    mfem::HypreParMatrix* eliminated_entries = op.EliminateRowsCols(ess_tdofs_);
    delete eliminated_entries;
  }

  const WeakForm& weak_form_;
  FieldState shape_disp_;
  std::vector<FieldState> fields_;
  std::vector<double> jacobian_weights_;
  TimeInfo time_info_;
  mfem::Array<int> ess_tdofs_;
  std::vector<StateBlockBinding> state_block_bindings_;
};

}  // namespace

std::unique_ptr<mfem::HypreParMatrix> buildWeakFormOperator(const WeakForm& weak_form, FieldState shape_disp,
                                                            std::vector<FieldState> fields,
                                                            std::vector<double> jacobian_weights, TimeInfo time_info,
                                                            mfem::Array<int> ess_tdofs)
{
  WeakFormBlockOperatorBuilder builder(weak_form, std::move(shape_disp), std::move(fields), std::move(jacobian_weights),
                                       time_info, std::move(ess_tdofs), {});
  return builder.build();
}

std::unique_ptr<mfem::HypreParMatrix> buildWeakFormOperator(const WeakForm& weak_form, FieldState shape_disp,
                                                            std::vector<FieldState> fields,
                                                            std::vector<double> jacobian_weights, TimeInfo time_info,
                                                            const BoundaryConditionManager* bc_manager)
{
  return buildWeakFormOperator(weak_form, std::move(shape_disp), std::move(fields), std::move(jacobian_weights),
                               time_info, copyEssentialTrueDofs(bc_manager));
}

StateDependentWeakFormOperator makeStateDependentWeakFormOperator(const WeakForm& weak_form, FieldState shape_disp,
                                                                  std::vector<FieldState> fields,
                                                                  std::vector<double> jacobian_weights,
                                                                  TimeInfo time_info, mfem::Array<int> ess_tdofs,
                                                                  std::vector<StateBlockBinding> state_block_bindings)
{
  WeakFormBlockOperatorBuilder builder(weak_form, std::move(shape_disp), std::move(fields), std::move(jacobian_weights),
                                       time_info, std::move(ess_tdofs), std::move(state_block_bindings));
  return [builder = std::move(builder)](const mfem::Vector& state, const mfem::Array<int>& block_offsets) mutable {
    return builder.updateAndBuild(state, block_offsets);
  };
}

StateDependentWeakFormOperator makeStateDependentWeakFormOperator(const WeakForm& weak_form, FieldState shape_disp,
                                                                  std::vector<FieldState> fields,
                                                                  std::vector<double> jacobian_weights,
                                                                  TimeInfo time_info,
                                                                  const BoundaryConditionManager* bc_manager,
                                                                  std::vector<StateBlockBinding> state_block_bindings)
{
  return makeStateDependentWeakFormOperator(weak_form, std::move(shape_disp), std::move(fields),
                                            std::move(jacobian_weights), time_info, copyEssentialTrueDofs(bc_manager),
                                            std::move(state_block_bindings));
}

BlockProviderOverride makeWeakFormBlockProviderOverride(int block_index, const WeakForm& weak_form,
                                                        FieldState shape_disp, std::vector<FieldState> fields,
                                                        std::vector<double> jacobian_weights, TimeInfo time_info,
                                                        mfem::Array<int> ess_tdofs)
{
  return makeFixedBlockProviderOverride(
      block_index, buildWeakFormOperator(weak_form, std::move(shape_disp), std::move(fields),
                                         std::move(jacobian_weights), time_info, std::move(ess_tdofs)));
}

BlockProviderOverride makeWeakFormBlockProviderOverride(int block_index, const WeakForm& weak_form,
                                                        FieldState shape_disp, std::vector<FieldState> fields,
                                                        std::vector<double> jacobian_weights, TimeInfo time_info,
                                                        const BoundaryConditionManager* bc_manager)
{
  return makeWeakFormBlockProviderOverride(block_index, weak_form, std::move(shape_disp), std::move(fields),
                                           std::move(jacobian_weights), time_info, copyEssentialTrueDofs(bc_manager));
}

BlockProviderOverride makeStateDependentWeakFormBlockProviderOverride(
    int block_index, const WeakForm& weak_form, FieldState shape_disp, std::vector<FieldState> fields,
    std::vector<double> jacobian_weights, TimeInfo time_info, mfem::Array<int> ess_tdofs,
    std::vector<StateBlockBinding> state_block_bindings)
{
  auto initial_operator = buildWeakFormOperator(weak_form, shape_disp, fields, jacobian_weights, time_info, ess_tdofs);
  auto weak_form_operator_update = makeStateDependentWeakFormOperator(
      weak_form, std::move(shape_disp), std::move(fields), std::move(jacobian_weights), time_info, std::move(ess_tdofs),
      std::move(state_block_bindings));
  auto block_builder = [weak_form_operator_update = std::move(weak_form_operator_update)](
                           const mfem::Vector& state,
                           const mfem::Array<int>& block_offsets) mutable -> std::unique_ptr<mfem::Operator> {
    return weak_form_operator_update(state, block_offsets);
  };
  return makeStateDependentBlockProviderOverride(block_index, std::move(block_builder), std::move(initial_operator));
}

BlockProviderOverride makeStateDependentWeakFormBlockProviderOverride(
    int block_index, const WeakForm& weak_form, FieldState shape_disp, std::vector<FieldState> fields,
    std::vector<double> jacobian_weights, TimeInfo time_info, const BoundaryConditionManager* bc_manager,
    std::vector<StateBlockBinding> state_block_bindings)
{
  return makeStateDependentWeakFormBlockProviderOverride(
      block_index, weak_form, std::move(shape_disp), std::move(fields), std::move(jacobian_weights), time_info,
      copyEssentialTrueDofs(bc_manager), std::move(state_block_bindings));
}

}  // namespace smith
