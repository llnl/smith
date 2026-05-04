#include <cmath>
#include "smith/physics/weak_form.hpp"
#include "smith/physics/field_types.hpp"
#include "smith/physics/boundary_conditions/boundary_condition_manager.hpp"
#include "smith/differentiable_numerics/nonlinear_solve.hpp"
#include "smith/differentiable_numerics/nonlinear_block_solver.hpp"
#include "smith/numerics/equation_solver.hpp"

namespace smith {

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

bool anyNonFinite(const std::vector<FEFieldPtr>& fields)
{
  for (const auto& f : fields) {
    const auto& v = *f;
    for (int i = 0; i < v.Size(); ++i) {
      if (!std::isfinite(v[i])) return true;
    }
  }
  return false;
}

std::vector<FEFieldPtr> deepCopyFields(const std::vector<FEFieldPtr>& src)
{
  std::vector<FEFieldPtr> out;
  out.reserve(src.size());
  for (const auto& f : src) {
    out.push_back(std::make_shared<FiniteElementState>(*f));
  }
  return out;
}

namespace {

static FEFieldPtr lerpBcField(const FEFieldPtr& prev_bc, const FEFieldPtr& target_bc,
                              const BoundaryConditionManager* bc_mgr, double alpha)
{
  auto out = std::make_shared<FiniteElementState>(*target_bc);
  if (!bc_mgr) return out;
  const auto& tdofs = bc_mgr->allEssentialTrueDofs();
  for (int i = 0; i < tdofs.Size(); ++i) {
    int j = tdofs[i];
    (*out)[j] = (1.0 - alpha) * (*prev_bc)[j] + alpha * (*target_bc)[j];
  }
  return out;
  return out;
}

template <typename JacFn>
static void warmStartBlockSolve(
    const NonlinearBlockSolverBase* solver_base, std::vector<FEFieldPtr>& diagonal_fields,
    const std::function<std::vector<mfem::Vector>(const std::vector<FEFieldPtr>&)>& eval_residuals,
    const JacFn& eval_jacobians, TimeInfo& current_time_info, const TimeInfo& time_info, size_t num_rows)
{
  auto* solver = dynamic_cast<const NonlinearBlockSolver*>(solver_base);
  if (!solver || !solver->retained_nonlinear_options_ || !solver->retained_nonlinear_options_->warm_start) {
    return;
  }

  // 1. Evaluate residual at t_old
  current_time_info =
      TimeInfo(time_info.time() - time_info.dt(), time_info.dt(), time_info.cycle() > 0 ? time_info.cycle() - 1 : 0);
  auto r_old = eval_residuals(diagonal_fields);

  // 2. Evaluate residual at t_new
  current_time_info = time_info;
  auto r_new = eval_residuals(diagonal_fields);

  // 3. Form RHS = -(r_new - r_old)
  mfem::Array<int> block_offsets(static_cast<int>(num_rows) + 1);
  block_offsets[0] = 0;
  for (size_t r = 0; r < num_rows; ++r) {
    block_offsets[static_cast<int>(r + 1)] = diagonal_fields[r]->space().TrueVSize();
  }
  block_offsets.PartialSum();

  mfem::BlockVector block_rhs(block_offsets);
  for (size_t r = 0; r < num_rows; ++r) {
    auto& rhs_block = block_rhs.GetBlock(static_cast<int>(r));
    for (int i = 0; i < rhs_block.Size(); ++i) {
      rhs_block[i] = -(r_new[r][i] - r_old[r][i]);
    }
  }

  // 4. Evaluate Jacobians
  auto jacobians = eval_jacobians(diagonal_fields);

  // 5. Form BlockOperator
  mfem::BlockOperator block_jac(block_offsets);
  for (size_t i = 0; i < num_rows; ++i) {
    for (size_t j = 0; j < num_rows; ++j) {
      auto& J = jacobians[i][j];
      if (J) {
        block_jac.SetBlock(static_cast<int>(i), static_cast<int>(j), J.get());
      }
    }
  }

  // 6. Solve
  mfem::BlockVector block_du(block_offsets);
  block_du = 0.0;

  if (solver->nonlinear_solver_) {
    auto& lin_solver = solver->nonlinear_solver_->linearSolver();
    if (num_rows == 1) {
      lin_solver.SetOperator(*jacobians[0][0]);
      lin_solver.Mult(block_rhs.GetBlock(0), block_du.GetBlock(0));
    } else {
      lin_solver.SetOperator(block_jac);
      lin_solver.Mult(block_rhs, block_du);
    }
  }

  // 7. Update interior fields
  for (size_t r = 0; r < num_rows; ++r) {
    *diagonal_fields[r] += block_du.GetBlock(static_cast<int>(r));
  }
}

template <typename ResFn, typename JacFn>
std::vector<FEFieldPtr> runBcRampCutbackLoop(const NonlinearBlockSolverBase* solver, std::vector<FEFieldPtr> last_good,
                                             const ResFn& res_fn, const JacFn& jac_fn,
                                             const std::vector<FEFieldPtr>& prev_bc,
                                             const std::vector<FEFieldPtr>& target_bc,
                                             const std::vector<const BoundaryConditionManager*>& bc_managers,
                                             std::shared_ptr<std::vector<FEFieldPtr>> current_bc_overrides,
                                             size_t num_rows)
{
  const auto& opts = solver->bcRampOptions();
  SLIC_ERROR_IF(num_rows != 1,
                axom::fmt::format("BC ramp cutback only supports single-block solves (got {} rows)", num_rows));
  SLIC_ERROR_IF(opts.shrink_factor <= 0.0 || opts.shrink_factor >= 1.0,
                "BcRampOptions.shrink_factor must be in (0, 1)");
  SLIC_ERROR_IF(opts.max_cutbacks <= 0, "BcRampOptions.max_cutbacks must be > 0");
  SLIC_ERROR_IF(opts.intermediate_max_iterations <= 0, "BcRampOptions.intermediate_max_iterations must be > 0");

  double last_good_alpha = 0.0;
  // Fast path at alpha=1 already failed; start cutback at the first shrunk fraction.
  double alpha = last_good_alpha + (1.0 - last_good_alpha) * opts.shrink_factor;
  int cutbacks = 1;  // count the failed alpha=1 fast path
  const bool print_ramp = solver->printLevel() >= 1;

  while (true) {
    for (size_t r = 0; r < num_rows; ++r) {
      (*current_bc_overrides)[r] = lerpBcField(prev_bc[r], target_bc[r], bc_managers[r], alpha);
    }
    const bool intermediate = (alpha < 1.0);
    solver->setIntermediateTolerancePolicy(intermediate, opts.intermediate_absolute_tol_fac,
                                           opts.intermediate_relative_tol, opts.intermediate_max_iterations);

    auto sols = solver->solve(last_good, res_fn, jac_fn);
    const bool converged = solver->lastSolveConverged() && !anyNonFinite(sols);
    if (print_ramp && !(converged && alpha >= 1.0)) {
      mfem::out << "[BcRamp] attempted alpha=" << alpha << " cutbacks=" << cutbacks << " converged=" << converged
                << "\n";
    }

    if (converged) {
      last_good = deepCopyFields(sols);
      last_good_alpha = alpha;
      if (alpha >= 1.0) {
        for (size_t r = 0; r < num_rows; ++r) (*current_bc_overrides)[r] = nullptr;
        return last_good;
      }
      alpha = 1.0;
    } else {
      SLIC_ERROR_IF(cutbacks >= opts.max_cutbacks,
                    axom::fmt::format("BC ramp exhausted max_cutbacks={} without reaching target.", opts.max_cutbacks));
      cutbacks++;
      alpha = last_good_alpha + (alpha - last_good_alpha) * opts.shrink_factor;
    }
  }
}

template <typename ResFn, typename JacFn>
std::vector<FEFieldPtr> solveWithOptionalBcRamp(const NonlinearBlockSolverBase* solver,
                                                const std::vector<FEFieldPtr>& diagonal_fields, const ResFn& res_fn,
                                                const JacFn& jac_fn,
                                                const std::vector<const BoundaryConditionManager*>& bc_managers,
                                                std::shared_ptr<std::vector<FEFieldPtr>> current_bc_overrides,
                                                double time, double dt, size_t num_rows)
{
  // Fast path: alpha=1 single-time BC eval. Bit-identical to ramp-disabled behavior.
  for (size_t r = 0; r < num_rows; ++r) (*current_bc_overrides)[r] = nullptr;
  solver->setIntermediateTolerancePolicy(false, 1.0, 0.0, 0);
  auto guess_snapshot = deepCopyFields(diagonal_fields);
  auto sols = solver->solve(diagonal_fields, res_fn, jac_fn);

  if (!solver->bcRampOptions().enabled) return sols;
  if (solver->lastSolveConverged() && !anyNonFinite(sols)) return sols;

  if (solver->printLevel() >= 1) {
    mfem::out << "[BcRamp] alpha=1 fast path failed (converged=" << solver->lastSolveConverged()
              << "), entering cutback\n";
  }

  // Precompute prev_bc and target_bc for cutbacks
  std::vector<FEFieldPtr> prev_bc(num_rows);
  std::vector<FEFieldPtr> target_bc(num_rows);
  for (size_t r = 0; r < num_rows; ++r) {
    prev_bc[r] = std::make_shared<FiniteElementState>(*guess_snapshot[r]);
    applyBoundaryConditions(time - dt, bc_managers[r], prev_bc[r], nullptr);

    target_bc[r] = std::make_shared<FiniteElementState>(*guess_snapshot[r]);
    applyBoundaryConditions(time, bc_managers[r], target_bc[r], nullptr);
  }

  // Cutback path: re-anchor at the last non-NaN guess and ramp alpha back from 1.
  return runBcRampCutbackLoop(solver, std::move(guess_snapshot), res_fn, jac_fn, prev_bc, target_bc, bc_managers,
                              current_bc_overrides, num_rows);
}

}  // namespace

std::vector<FieldState> block_solve(const std::vector<WeakForm*>& residual_evals,
                                    const std::vector<std::vector<size_t>> block_indices, const FieldState& shape_disp,
                                    const std::vector<std::vector<FieldState>>& states,
                                    const std::vector<std::vector<FieldState>>& params, const TimeInfo& time_info,
                                    const NonlinearBlockSolverBase* solver,
                                    const std::vector<const BoundaryConditionManager*>& bc_managers,
                                    const std::vector<FEFieldPtr>& bc_field_overrides)
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
    // Validate that diagonal entry is not invalid
    SLIC_ERROR_IF(
        block_indices[row][row] == invalid_block_index,
        axom::fmt::format("block_indices[{}][{}] (diagonal entry) must not be invalid_block_index", row, row));
  }

  std::vector<size_t> num_state_inputs;
  std::vector<gretl::StateBase> allFields;
  for (auto& ss : states) {
    num_state_inputs.push_back(ss.size());
    for (auto& s : ss) {
      allFields.push_back(s);
    }
  }
  std::vector<size_t> num_param_inputs;
  for (auto& ps : params) {
    num_param_inputs.push_back(ps.size());
    for (auto& p : ps) {
      allFields.push_back(p);
    }
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

    // The override pointer state changes over the cutback loop.
    auto current_bc_overrides = std::make_shared<std::vector<FEFieldPtr>>(num_rows, nullptr);
    if (!bc_field_overrides.empty()) {
      for (size_t row_i = 0; row_i < num_rows; ++row_i) {
        (*current_bc_overrides)[row_i] = bc_field_overrides[row_i];
      }
    }

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

    for (size_t row_i = 0; row_i < num_rows; ++row_i) {
      FEFieldPtr primal_field_row_i = diagonal_fields[row_i];
      applyBoundaryConditions(time_info.time(), bc_managers[row_i], primal_field_row_i, (*current_bc_overrides)[row_i]);
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

    TimeInfo current_time_info = time_info;

    auto eval_residuals = [=, &current_time_info](const std::vector<FEFieldPtr>& unknowns) {
      SLIC_ERROR_IF(unknowns.size() != num_rows,
                    "block solver unknowns size must match the number or residuals in block_solve");
      std::vector<mfem::Vector> residuals(num_rows);

      for (size_t row_i = 0; row_i < num_rows; ++row_i) {
        FEFieldPtr primal_field_row_i = diagonal_fields[row_i];
        *primal_field_row_i = *unknowns[row_i];
        applyBoundaryConditions(current_time_info.time(), bc_managers[row_i], primal_field_row_i,
                                (*current_bc_overrides)[row_i]);
      }
      for (size_t row_i = 0; row_i < num_rows; ++row_i) {
        residuals[row_i] = residual_evals[row_i]->residual(current_time_info, shape_disp_ptr.get(),
                                                           getConstFieldPointers(input_fields[row_i]));
        if (bc_managers[row_i]) {
          residuals[row_i].SetSubVector(bc_managers[row_i]->allEssentialTrueDofs(), 0.0);
        }
      }
      return residuals;
    };

    auto eval_jacobians = [=, &current_time_info](const std::vector<FEFieldPtr>& unknowns) {
      SLIC_ERROR_IF(unknowns.size() != num_rows,
                    "block solver unknown size must match the number or residuals in block_solve");
      std::vector<std::vector<std::unique_ptr<mfem::HypreParMatrix>>> jacobians(num_rows);

      for (size_t row_i = 0; row_i < num_rows; ++row_i) {
        FEFieldPtr primal_field_row_i = diagonal_fields[row_i];
        *primal_field_row_i = *unknowns[row_i];
        applyBoundaryConditions(current_time_info.time(), bc_managers[row_i], primal_field_row_i,
                                (*current_bc_overrides)[row_i]);
      }

      for (size_t row_i = 0; row_i < num_rows; ++row_i) {
        std::vector<FEFieldPtr> row_field_inputs = input_fields[row_i];
        std::vector<double> tangent_weights(row_field_inputs.size(), 0.0);
        for (size_t col_j = 0; col_j < num_rows; ++col_j) {
          size_t field_index_to_diff = block_indices[row_i][col_j];
          if (field_index_to_diff != invalid_block_index) {
            tangent_weights[field_index_to_diff] = 1.0;
            auto jac_ij = residual_evals[row_i]->jacobian(current_time_info, shape_disp_ptr.get(),
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
        if (jacobians[row_i][row_i]) {
          jacobians[row_i][row_i]->EliminateBC(bc_managers[row_i]->allEssentialTrueDofs(),
                                               mfem::Operator::DiagonalPolicy::DIAG_ONE);
        }
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
      return jacobians;
    };

    if (bc_field_overrides.empty()) {
      warmStartBlockSolve(solver, diagonal_fields, eval_residuals, eval_jacobians, current_time_info, time_info,
                          num_rows);
      diagonal_fields = solveWithOptionalBcRamp(solver, diagonal_fields, eval_residuals, eval_jacobians, bc_managers,
                                                current_bc_overrides, time_info.time(), time_info.dt(), num_rows);
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
