#include <cmath>
#include "smith/physics/weak_form.hpp"
#include "smith/physics/field_types.hpp"
#include "smith/physics/boundary_conditions/boundary_condition_manager.hpp"
#include "smith/differentiable_numerics/nonlinear_solve.hpp"
#include "smith/differentiable_numerics/nonlinear_block_solver.hpp"

namespace smith {

namespace {

void applyBoundaryConditions(double time, const BoundaryConditionManager* bc_manager, FEFieldPtr& primal_field,
                             const FEFieldPtr& bc_field_ptr)
{
  if (!bc_manager) return;
  if (bc_field_ptr) {
    auto constrained_dofs = bc_manager->allEssentialTrueDofs();
    for (int i = 0; i < constrained_dofs.Size(); i++) {
      int j = constrained_dofs[i];
      (*primal_field)[j] = (*bc_field_ptr)(j);
    }
  } else {
    for (auto& bc : bc_manager->essentials()) {
      bc.setDofs(*primal_field, time);
    }
  }
}

bool hasNonFinite(const mfem::Vector& vector)
{
  for (int i = 0; i < vector.Size(); ++i)
    if (!std::isfinite(vector[i])) return true;
  return false;
}

bool hasNonFinite(const std::vector<FEFieldPtr>& fields)
{
  for (const auto& field : fields)
    if (hasNonFinite(*field)) return true;
  return false;
}

std::vector<FEFieldPtr> deepCopyFields(const std::vector<FEFieldPtr>& src)
{
  std::vector<FEFieldPtr> out;
  out.reserve(src.size());
  for (const auto& f : src) out.push_back(std::make_shared<FiniteElementState>(*f));
  return out;
}

// Field whose constrained tdofs are alpha-interpolated between prev_bc and target_bc.
// Unconstrained values irrelevant — applyBoundaryConditions only reads tdofs.
FEFieldPtr lerpBcField(const FEFieldPtr& prev_bc, const FEFieldPtr& target_bc, const BoundaryConditionManager* bc_mgr,
                       double alpha)
{
  auto out = std::make_shared<FiniteElementState>(*target_bc);
  if (!bc_mgr) return out;
  const auto& tdofs = bc_mgr->allEssentialTrueDofs();
  for (int i = 0; i < tdofs.Size(); ++i) {
    int j = tdofs[i];
    (*out)[j] = (1.0 - alpha) * (*prev_bc)[j] + alpha * (*target_bc)[j];
  }
  return out;
}

// Set BC overrides at fraction alpha along the prev → target segment. alpha>=1 clears.
void setBcOverrides(std::vector<FEFieldPtr>& current_bc_overrides, const std::vector<FEFieldPtr>& prev_bc,
                    const std::vector<FEFieldPtr>& target_bc,
                    const std::vector<const BoundaryConditionManager*>& bc_managers, double alpha)
{
  if (alpha >= 1.0) {
    for (auto& o : current_bc_overrides) o = nullptr;
  } else {
    for (size_t r = 0; r < current_bc_overrides.size(); ++r)
      current_bc_overrides[r] = lerpBcField(prev_bc[r], target_bc[r], bc_managers[r], alpha);
  }
}

template <typename RawJacFn>
FEFieldPtr predictBcStep(const NonlinearBlockSolverBase* solver, const FEFieldPtr& base_state, double base_alpha,
                         double attempt_alpha, double base_time, const RawJacFn& raw_jac_fn,
                         const std::vector<FEFieldPtr>& base_bc, const std::vector<FEFieldPtr>& requested_bc,
                         const std::vector<const BoundaryConditionManager*>& bc_managers,
                         std::vector<FEFieldPtr>& current_bc_overrides)
{
  if (!solver->warmStartEnabled() || !bc_managers[0]) return nullptr;

  auto saved_bc_overrides = current_bc_overrides;
  setBcOverrides(current_bc_overrides, base_bc, requested_bc, bc_managers, base_alpha);
  auto raw_jacobian = raw_jac_fn(std::vector<FEFieldPtr>{base_state}, base_time);
  current_bc_overrides = saved_bc_overrides;
  if (!raw_jacobian) return nullptr;

  auto attempt_bc = lerpBcField(base_bc[0], requested_bc[0], bc_managers[0], attempt_alpha);
  return solver->predictBcStep(base_state, attempt_bc, *raw_jacobian, bc_managers[0]->allEssentialTrueDofs());
}

struct BcAttempt {
  std::vector<FEFieldPtr> fields;
  bool converged = false;
};

template <typename ResFn, typename JacFn, typename RawJacFn>
BcAttempt attemptBcStep(const NonlinearBlockSolverBase* solver, const std::vector<FEFieldPtr>& anchor,
                        double anchor_alpha, double attempt_alpha, double anchor_time, const ResFn& res_fn,
                        const JacFn& jac_fn, const RawJacFn& raw_jac_fn, const std::vector<FEFieldPtr>& base_bc,
                        const std::vector<FEFieldPtr>& requested_bc,
                        const std::vector<const BoundaryConditionManager*>& bc_managers,
                        std::vector<FEFieldPtr>& current_bc_overrides, double time)
{
  auto initial_guess = deepCopyFields(anchor);
  if (auto prediction = predictBcStep(solver, anchor[0], anchor_alpha, attempt_alpha, anchor_time, raw_jac_fn, base_bc,
                                      requested_bc, bc_managers, current_bc_overrides)) {
    initial_guess[0] = std::move(prediction);
  }

  setBcOverrides(current_bc_overrides, base_bc, requested_bc, bc_managers, attempt_alpha);
  for (size_t row = 0; row < initial_guess.size(); ++row) {
    applyBoundaryConditions(time, bc_managers[row], initial_guess[row], current_bc_overrides[row]);
  }

  auto fields = solver->solve(initial_guess, res_fn, jac_fn);
  bool converged = solver->lastSolveConverged() && !hasNonFinite(fields);
  return {std::move(fields), converged};
}

template <typename ResFn, typename JacFn, typename RawJacFn>
std::vector<FEFieldPtr> solveWithBcContinuation(const NonlinearBlockSolverBase* solver,
                                                const std::vector<FEFieldPtr>& previous_fields, const ResFn& res_fn,
                                                const JacFn& jac_fn, const RawJacFn& raw_jac_fn,
                                                const std::vector<const BoundaryConditionManager*>& bc_managers,
                                                std::vector<FEFieldPtr>& current_bc_overrides,
                                                const TimeInfo& time_info)
{
  const auto& options = solver->bcRampOptions();
  SLIC_ERROR_IF(options.max_cutbacks < 0, "BcRampOptions.max_cutbacks must be nonnegative");
  SLIC_ERROR_IF(options.max_cutbacks > 0 && (options.shrink_factor <= 0.0 || options.shrink_factor >= 1.0),
                "BcRampOptions.shrink_factor must be in (0, 1)");

  std::vector<FEFieldPtr> base_bc = previous_fields;
  auto requested_bc = deepCopyFields(previous_fields);
  for (size_t row = 0; row < requested_bc.size(); ++row) {
    applyBoundaryConditions(time_info.time(), bc_managers[row], requested_bc[row], nullptr);
  }

  auto result = attemptBcStep(solver, previous_fields, 0.0, 1.0, time_info.time() - time_info.dt(), res_fn, jac_fn,
                              raw_jac_fn, base_bc, requested_bc, bc_managers, current_bc_overrides, time_info.time());
  if (result.converged || options.max_cutbacks == 0) return result.fields;

  auto anchor = deepCopyFields(previous_fields);
  double anchor_alpha = 0.0;
  double failed_alpha = 1.0;
  double anchor_time = time_info.time() - time_info.dt();

  for (int cutbacks_used = 0; cutbacks_used < options.max_cutbacks; ++cutbacks_used) {
    double attempt_alpha = anchor_alpha + options.shrink_factor * (failed_alpha - anchor_alpha);
    result = attemptBcStep(solver, anchor, anchor_alpha, attempt_alpha, anchor_time, res_fn, jac_fn, raw_jac_fn,
                           base_bc, requested_bc, bc_managers, current_bc_overrides, time_info.time());
    if (!result.converged) {
      failed_alpha = attempt_alpha;
      continue;
    }

    anchor = deepCopyFields(result.fields);
    anchor_alpha = attempt_alpha;
    anchor_time = time_info.time();
    result = attemptBcStep(solver, anchor, anchor_alpha, 1.0, anchor_time, res_fn, jac_fn, raw_jac_fn, base_bc,
                           requested_bc, bc_managers, current_bc_overrides, time_info.time());
    if (result.converged) return result.fields;
    failed_alpha = 1.0;
  }

  for (auto& bc_override : current_bc_overrides) bc_override = nullptr;
  SLIC_ERROR_ROOT(axom::fmt::format("BC ramp exhausted max_cutbacks={}, last_good_alpha={}, failed_alpha={}",
                                    options.max_cutbacks, anchor_alpha, failed_alpha));
  return result.fields;
}

}  // namespace

std::vector<FieldState> block_solve(const std::vector<WeakForm*>& residual_evals,
                                    const std::vector<std::vector<size_t>> block_indices, const FieldState& shape_disp,
                                    const std::vector<std::vector<FieldState>>& states,
                                    const std::vector<std::vector<FieldState>>& params, const TimeInfo& time_info,
                                    const NonlinearBlockSolverBase* solver,
                                    const std::vector<const BoundaryConditionManager*>& bc_managers)
{
  SMITH_MARK_FUNCTION;
  size_t num_rows_ = residual_evals.size();

  SLIC_ERROR_IF(num_rows_ != block_indices.size(), "Block indices size not consistent with number of residual rows");
  SLIC_ERROR_IF(num_rows_ != states.size(),
                "Number of state input vectors not consistent with number of residual rows");
  SLIC_ERROR_IF(num_rows_ != params.size(),
                "Number of parameter input vectors not consistent with number of residual rows");
  SLIC_ERROR_IF(num_rows_ != bc_managers.size(),
                "Number of boundary condition manager not consistent with number of residual rows");

  for (size_t r = 0; r < num_rows_; ++r) {
    SLIC_ERROR_IF(num_rows_ != block_indices[r].size(), "All block index rows must have the same number of columns");
  }

  // Validate block_indices bounds against states sizes
  for (size_t row = 0; row < num_rows_; ++row) {
    for (size_t col = 0; col < num_rows_; ++col) {
      size_t idx = block_indices[row][col];
      if (idx != invalid_block_index) {
        SLIC_ERROR_IF(idx >= states[row].size(),
                      axom::fmt::format("block_indices[{}][{}] = {} is out of bounds (states[{}].size() = {})", row,
                                        col, idx, row, states[row].size()));
      }
    }
    SLIC_ERROR_IF(
        block_indices[row][row] == invalid_block_index,
        axom::fmt::format("block_indices[{}][{}] (diagonal entry) must not be invalid_block_index", row, row));
  }

  std::vector<size_t> num_state_inputs;
  std::vector<gretl::StateBase> allFields;
  for (auto& ss : states) {
    num_state_inputs.push_back(ss.size());
    for (auto& s : ss) allFields.push_back(s);
  }
  std::vector<size_t> num_param_inputs;
  for (auto& ps : params) {
    num_param_inputs.push_back(ps.size());
    for (auto& p : ps) allFields.push_back(p);
  }
  allFields.push_back(shape_disp);
  struct ZeroDualVectors {
    std::vector<FEDualPtr> operator()(const std::vector<FEFieldPtr>& fs)
    {
      std::vector<FEDualPtr> ds(fs.size());
      for (size_t i = 0; i < fs.size(); ++i) {
        ds[i] = std::make_shared<FiniteElementDual>(fs[i]->space(), fs[i]->name() + "_dual");
      }
      return ds;
    }
  };

  FieldVecState sol =
      shape_disp.create_state<std::vector<FEFieldPtr>, std::vector<FEDualPtr>>(allFields, ZeroDualVectors());
  sol.set_eval([=](const gretl::UpstreamStates& upstreams, gretl::DownstreamState& downstream) {
    SMITH_MARK_BEGIN("solve forward");
    const size_t num_rows = num_state_inputs.size();
    std::vector<std::vector<FEFieldPtr>> input_fields(num_rows);
    SLIC_ERROR_IF(num_rows != num_param_inputs.size(), "row count for params and states are inconsistent");

    // BC overrides written by the BC ramp cutback loop; nullptr means evaluate BC coefficient at time.
    std::vector<FEFieldPtr> current_bc_overrides(num_rows, nullptr);

    // The order of inputs in upstreams is:
    // states of residual 0, states of residual 1, ... , params of residual 0, params of residual 1, ...
    size_t field_count = 0;
    for (size_t row_i = 0; row_i < num_rows; ++row_i) {
      for (size_t state_i = 0; state_i < num_state_inputs[row_i]; ++state_i) {
        input_fields[row_i].push_back(upstreams[field_count++].get<FEFieldPtr>());
      }
    }
    for (size_t row_i = 0; row_i < num_rows; ++row_i) {
      for (size_t param_i = 0; param_i < num_param_inputs[row_i]; ++param_i) {
        input_fields[row_i].push_back(upstreams[field_count++].get<FEFieldPtr>());
      }
    }

    std::vector<FEFieldPtr> diagonal_fields(num_rows);
    for (size_t row_i = 0; row_i < num_rows; ++row_i) {
      size_t prime_unknown_row_i = block_indices[row_i][row_i];
      SLIC_ERROR_IF(prime_unknown_row_i == invalid_block_index,
                    "The primary unknown field (field index for block_index[n][n], must not be invalid)");
      diagonal_fields[row_i] = std::make_shared<FiniteElementState>(*input_fields[row_i][prime_unknown_row_i]);
    }

    // Anchor for warm start and BC ramp: snapshot pre-BC.
    auto u_prev_snapshot = deepCopyFields(diagonal_fields);

    for (size_t row_i = 0; row_i < num_rows; ++row_i) {
      FEFieldPtr primal_field_row_i = diagonal_fields[row_i];
      applyBoundaryConditions(time_info.time(), bc_managers[row_i], primal_field_row_i, current_bc_overrides[row_i]);
    }

    for (size_t row_i = 0; row_i < num_rows; ++row_i) {
      for (size_t col_j = 0; col_j < num_rows; ++col_j) {
        size_t prime_unknown_ij = block_indices[row_i][col_j];
        if (prime_unknown_ij != invalid_block_index) {
          input_fields[row_i][block_indices[row_i][col_j]] = diagonal_fields[col_j];
        }
      }
    }

    const FEFieldPtr shape_disp_ptr = upstreams[field_count].get<FEFieldPtr>();

    auto eval_residuals_at = [=, &current_bc_overrides](const std::vector<FEFieldPtr>& unknowns, double eval_time) {
      SLIC_ERROR_IF(unknowns.size() != num_rows,
                    "block solver unknowns size must match the number or residuals in block_solve");
      std::vector<mfem::Vector> residuals(num_rows);

      for (size_t row_i = 0; row_i < num_rows; ++row_i) {
        FEFieldPtr primal_field_row_i = diagonal_fields[row_i];
        *primal_field_row_i = *unknowns[row_i];
        applyBoundaryConditions(eval_time, bc_managers[row_i], primal_field_row_i, current_bc_overrides[row_i]);
      }
      for (size_t row_i = 0; row_i < num_rows; ++row_i) {
        residuals[row_i] = residual_evals[row_i]->residual(time_info, shape_disp_ptr.get(),
                                                           getConstFieldPointers(input_fields[row_i]));
        if (bc_managers[row_i]) {
          residuals[row_i].SetSubVector(bc_managers[row_i]->allEssentialTrueDofs(), 0.0);
        }
      }
      return residuals;
    };
    auto eval_residuals = [&, eval_residuals_at](const std::vector<FEFieldPtr>& unknowns) {
      return eval_residuals_at(unknowns, time_info.time());
    };

    auto assemble_jacobians_raw_at = [=, &current_bc_overrides](const std::vector<FEFieldPtr>& unknowns,
                                                                double eval_time) {
      SLIC_ERROR_IF(unknowns.size() != num_rows,
                    "block solver unknown size must match the number or residuals in block_solve");
      std::vector<std::vector<std::unique_ptr<mfem::HypreParMatrix>>> jacobians(num_rows);

      for (size_t row_i = 0; row_i < num_rows; ++row_i) {
        FEFieldPtr primal_field_row_i = diagonal_fields[row_i];
        *primal_field_row_i = *unknowns[row_i];
        applyBoundaryConditions(eval_time, bc_managers[row_i], primal_field_row_i, current_bc_overrides[row_i]);
      }

      for (size_t row_i = 0; row_i < num_rows; ++row_i) {
        std::vector<FEFieldPtr> row_field_inputs = input_fields[row_i];
        std::vector<double> tangent_weights(row_field_inputs.size(), 0.0);
        TimeInfo eval_time_info(eval_time - time_info.dt(), time_info.dt(), time_info.cycle());
        for (size_t col_j = 0; col_j < num_rows; ++col_j) {
          size_t field_index_to_diff = block_indices[row_i][col_j];
          if (field_index_to_diff != invalid_block_index) {
            tangent_weights[field_index_to_diff] = 1.0;
            auto jac_ij = residual_evals[row_i]->jacobian(eval_time_info, shape_disp_ptr.get(),
                                                          getConstFieldPointers(row_field_inputs), tangent_weights);
            jacobians[row_i].emplace_back(std::move(jac_ij));
            tangent_weights[field_index_to_diff] = 0.0;
          } else {
            jacobians[row_i].emplace_back(nullptr);
          }
        }
      }
      return jacobians;
    };
    auto assemble_jacobians_raw = [&, assemble_jacobians_raw_at](const std::vector<FEFieldPtr>& unknowns) {
      return assemble_jacobians_raw_at(unknowns, time_info.time());
    };

    auto eval_jacobians = [=, &assemble_jacobians_raw](const std::vector<FEFieldPtr>& unknowns) {
      auto jacobians = assemble_jacobians_raw(unknowns);
      for (size_t row_i = 0; row_i < num_rows; ++row_i) {
        if (!bc_managers[row_i]) continue;
        if (jacobians[row_i][row_i]) {
          jacobians[row_i][row_i]->EliminateBC(bc_managers[row_i]->allEssentialTrueDofs(),
                                               mfem::Operator::DiagonalPolicy::DIAG_ONE);
        }
        for (size_t col_j = 0; col_j < num_rows; ++col_j) {
          if (col_j == row_i) continue;
          if (jacobians[row_i][col_j])
            jacobians[row_i][col_j]->EliminateRows(bc_managers[row_i]->allEssentialTrueDofs());
          if (jacobians[col_j][row_i])
            delete jacobians[col_j][row_i]->EliminateCols(bc_managers[row_i]->allEssentialTrueDofs());
        }
      }
      return jacobians;
    };

    auto assemble_single_raw = [&](const std::vector<FEFieldPtr>& u,
                                   double eval_time) -> std::unique_ptr<mfem::HypreParMatrix> {
      auto jacs = assemble_jacobians_raw_at(u, eval_time);
      return (jacs.empty() || jacs[0].empty()) ? nullptr : std::move(jacs[0][0]);
    };

    const bool use_bc_continuation = solver->warmStartEnabled() || solver->bcRampOptions().max_cutbacks != 0;
    SLIC_ERROR_IF(use_bc_continuation && num_rows != 1, "BC prediction and cutback only support single-block solves");

    if (use_bc_continuation) {
      diagonal_fields = solveWithBcContinuation(solver, u_prev_snapshot, eval_residuals, eval_jacobians,
                                                assemble_single_raw, bc_managers, current_bc_overrides, time_info);
    } else {
      diagonal_fields = solver->solve(diagonal_fields, eval_residuals, eval_jacobians);
    }

    downstream.set<std::vector<FEFieldPtr>, std::vector<FEDualPtr>>(diagonal_fields);

    SMITH_MARK_END("solve forward");
  });

  sol.set_vjp([=](gretl::UpstreamStates& upstreams, const gretl::DownstreamState& downstream) {
    SMITH_MARK_BEGIN("solve reverse");
    const std::vector<FEFieldPtr> s = downstream.get<std::vector<FEFieldPtr>>();  // get the final solution
    const std::vector<FEDualPtr> s_dual =
        downstream.get_dual<std::vector<FEDualPtr>, std::vector<FEFieldPtr>>();  // get the dual load

    const size_t num_rows = num_state_inputs.size();
    SLIC_ERROR_IF(s_dual.size() != num_rows,
                  "block solver vjp downstream size must match the number or residuals in block_solve");

    std::vector<std::vector<FEFieldPtr>> input_fields(num_rows);
    size_t field_count = 0;
    for (size_t row_i = 0; row_i < num_rows; ++row_i) {
      for (size_t state_i = 0; state_i < num_state_inputs[row_i]; ++state_i) {
        input_fields[row_i].push_back(upstreams[field_count++].get<FEFieldPtr>());
      }
    }
    for (size_t row_i = 0; row_i < num_rows; ++row_i) {
      for (size_t param_i = 0; param_i < num_param_inputs[row_i]; ++param_i) {
        input_fields[row_i].push_back(upstreams[field_count++].get<FEFieldPtr>());
      }
    }

    // if the field is a primal variable we solved before,
    // make a copy so we don't accidentally override the original copy
    std::vector<FEFieldPtr> diagonal_fields(num_rows);
    for (size_t row_i = 0; row_i < num_rows; ++row_i) {
      diagonal_fields[row_i] = std::make_shared<FiniteElementState>(*input_fields[row_i][block_indices[row_i][row_i]]);
      *diagonal_fields[row_i] = *s[row_i];
    }

    for (size_t row_i = 0; row_i < num_rows; ++row_i) {
      for (size_t col_j = 0; col_j < num_rows; ++col_j) {
        if (block_indices[row_i][col_j] != invalid_block_index) {
          input_fields[row_i][block_indices[row_i][col_j]] = diagonal_fields[col_j];
        }
      }
    }

    const FEFieldPtr shape_disp_ptr = upstreams[field_count].get<FEFieldPtr>();

    // I'm not sure this will be the right timestamp to apply boundary condition during backward propagation
    // Need to double check for time-dependent boundary conditions
    for (size_t row_i = 0; row_i < num_rows; ++row_i) {
      FEFieldPtr primal_field_row_i = diagonal_fields[row_i];
      applyBoundaryConditions(time_info.time(), bc_managers[row_i], primal_field_row_i, nullptr);
    }

    solver->clearMemory();

    std::vector<std::vector<std::unique_ptr<mfem::HypreParMatrix>>> jacobians(num_rows);
    for (size_t row_i = 0; row_i < num_rows; ++row_i) {
      std::vector<FEFieldPtr> row_field_inputs = input_fields[row_i];
      std::vector<double> tangent_weights(row_field_inputs.size(), 0.0);
      for (size_t col_j = 0; col_j < num_rows; ++col_j) {
        size_t field_index_to_diff = block_indices[row_i][col_j];
        if (field_index_to_diff != invalid_block_index) {
          tangent_weights[field_index_to_diff] = 1.0;
          auto jac_ij = residual_evals[row_i]->jacobian(time_info, shape_disp_ptr.get(),
                                                        getConstFieldPointers(row_field_inputs), tangent_weights);
          jacobians[row_i].emplace_back(std::move(jac_ij));
          tangent_weights[field_index_to_diff] = 0.0;
        } else {
          jacobians[row_i].emplace_back(nullptr);
        }
      }
    }

    // Apply BCs to the block system
    for (size_t row_i = 0; row_i < num_rows; ++row_i) {
      if (!bc_managers[row_i]) {
        continue;
      }
      s_dual[row_i]->SetSubVector(bc_managers[row_i]->allEssentialTrueDofs(), 0.0);

      mfem::HypreParMatrix* Jii =
          jacobians[row_i][row_i]->EliminateRowsCols(bc_managers[row_i]->allEssentialTrueDofs());
      delete Jii;
      for (size_t col_j = 0; col_j < num_rows; ++col_j) {
        if (col_j != row_i) {
          if (jacobians[row_i][col_j]) {
            jacobians[row_i][col_j]->EliminateRows(bc_managers[row_i]->allEssentialTrueDofs());
          }
          if (jacobians[col_j][row_i]) {
            mfem::HypreParMatrix* Jji =
                jacobians[col_j][row_i]->EliminateCols(bc_managers[row_i]->allEssentialTrueDofs());
            delete Jji;
          }
        }
      }
    }

    // Take the transpose of the block system
    std::vector<std::vector<std::unique_ptr<mfem::HypreParMatrix>>> jacobians_T(num_rows);
    for (size_t col_j = 0; col_j < num_rows; ++col_j) {
      for (size_t row_i = 0; row_i < num_rows; ++row_i) {
        if (jacobians[row_i][col_j]) {
          jacobians_T[col_j].emplace_back(std::unique_ptr<mfem::HypreParMatrix>(jacobians[row_i][col_j]->Transpose()));
        } else {
          jacobians_T[col_j].emplace_back(nullptr);
        }
      }
    }
    for (size_t row_i = 0; row_i < num_rows; ++row_i) {
      for (size_t col_j = 0; col_j < num_rows; ++col_j) {
        jacobians[row_i][col_j].reset();
      }
    }

    std::vector<FEFieldPtr> adjoint_fields(num_rows);
    adjoint_fields = solver->solveAdjoint(s_dual, jacobians_T);
    for (size_t row_i = 0; row_i < num_rows; ++row_i) {
      *adjoint_fields[row_i] *= -1.0;
    }

    // Update sensitivities
    std::vector<std::vector<FEDualPtr>> field_sensitivities(num_rows);
    FEDualPtr shape_disp_sensitivity = upstreams[field_count].get_dual<FEDualPtr, FEFieldPtr>();
    size_t dual_index = 0;
    for (size_t row_i = 0; row_i < num_rows; ++row_i) {
      for (size_t state_i = 0; state_i < num_state_inputs[row_i]; ++state_i) {
        field_sensitivities[row_i].push_back(upstreams[dual_index++].get_dual<FEDualPtr, FEFieldPtr>());
      }
    }
    for (size_t row_i = 0; row_i < num_rows; ++row_i) {
      for (size_t param_i = 0; param_i < num_param_inputs[row_i]; ++param_i) {
        field_sensitivities[row_i].push_back(upstreams[dual_index++].get_dual<FEDualPtr, FEFieldPtr>());
      }
    }
    SLIC_ERROR_IF(field_count != dual_index, "Number of sensitivities must equal to number of upstreams");

    // No sensitivity needed for primal fields
    for (size_t row_i = 0; row_i < num_rows; ++row_i) {
      for (size_t col_j = 0; col_j < num_rows; ++col_j) {
        if (block_indices[row_i][col_j] != invalid_block_index) {
          field_sensitivities[row_i][block_indices[row_i][col_j]] = nullptr;
        }
      }
    }

    for (size_t row_i = 0; row_i < num_rows; ++row_i) {
      residual_evals[row_i]->vjp(time_info, shape_disp_ptr.get(), getConstFieldPointers(input_fields[row_i]), {},
                                 adjoint_fields[row_i].get(), shape_disp_sensitivity.get(),
                                 getFieldPointers(field_sensitivities[row_i]), {});
    }

    SMITH_MARK_END("solve reverse");
  });

  sol.finalize();

  std::vector<FieldState> results;
  for (size_t i = 0; i < num_rows_; ++i) {
    FieldState s = gretl::create_state<FEFieldPtr, FEDualPtr>(
        zero_dual_from_state(),
        [i](const std::vector<FEFieldPtr>& sols) {
          auto state_copy = std::make_shared<FiniteElementState>(sols[i]->space(), sols[i]->name());
          *state_copy = *sols[i];
          return state_copy;
        },
        [i](const std::vector<FEFieldPtr>&, const FEFieldPtr&, std::vector<FEDualPtr>& sols_,
            const FEDualPtr& output_) { *sols_[i] += *output_; },
        sol);

    results.emplace_back(s);
  }

  return results;
}

}  // namespace smith
