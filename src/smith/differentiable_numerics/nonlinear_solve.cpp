#include "smith/physics/weak_form.hpp"
#include "smith/physics/field_types.hpp"
#include "smith/physics/boundary_conditions/boundary_condition_manager.hpp"
#include "smith/differentiable_numerics/nonlinear_solve.hpp"
#include "smith/differentiable_numerics/differentiable_solver.hpp"
#include "smith/differentiable_numerics/dirichlet_boundary_conditions.hpp"
#include "smith/differentiable_numerics/nonlinear_solve.hpp"

namespace smith {

/// @brief apply boundary conditions
void applyBoundaryConditions(double time, const smith::BoundaryConditionManager* bc_manager,
                             smith::FEFieldPtr& primal_field, const smith::FEFieldPtr& bc_field_ptr)
{
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

/// @brief Solve a nonlinear system of equations as defined by the weak form
/// @param residual_eval The weak form which defines the equations to be solved
/// @param shape_disp The mesh-morphed shape displacement
/// @param states The time varying states as inputs to the weak form
/// @param params The fixed field parameters as inputs to the weak form
/// @param state_update_weights Specifies how to blend the arguments of the weak form into a unique unknown.  The
/// primary unknown, p, is states[primal_solve_state_index].  However, other arguments to the weak form may also depend
/// on this unknown. The state_update_weights specify how this is done.  For a nonzero weight, the field argument i is
/// assumed to change as states[i] + (p - p_initial) * state_update_weights[i].
/// @param primal_solve_state_index Index specifying which of the states is the primary unknown.
/// @param dirichlet_state_index Index specifying which field has the Dirichlet boundary conditions applied to it.
/// Typically this will be the same as the primal_solve_state_index, but it can be different.  An example is explicit
/// dynamics, where the unknown is the acceleration, and the boundary conditions are often applied directly to the
/// displacement field.
/// @param time_info Timestep information (time, dt, cycle)
/// @param solver The differentiable, potentially nonlinear, equation solver used to solve the system of equations
/// @param bc_manager Holds information about which degrees of freedom (DOFS)
/// @param bc_field A field which holds to desired values for the fixed degrees of freedom as specified by the
/// bc_manager
/// @return The field solution to the weak form
FieldState nonlinearSolve(const WeakForm* residual_eval, const FieldState& shape_disp,
                          const std::vector<FieldState>& states, const std::vector<FieldState>& params,
                          const std::vector<double>& state_update_weights, size_t primal_solve_state_index,
                          size_t dirichlet_state_index, const TimeInfo& time_info, const DifferentiableSolver* solver,
                          const BoundaryConditionManager* bc_manager, const FieldState* bc_field = nullptr)
{
  SMITH_MARK_FUNCTION;
  SLIC_ERROR_IF(states.size() != state_update_weights.size(), "State and state weight fields are inconsistent");
  SLIC_ERROR_IF(state_update_weights[primal_solve_state_index] != 1.0, "Primal state must have a weight of 1.0");

  std::vector<gretl::StateBase> allFields;
  for (auto& s : states) allFields.push_back(s);
  for (auto& p : params) allFields.push_back(p);
  allFields.push_back(shape_disp);

  bool have_bc_field = bc_field;
  if (have_bc_field) {
    allFields.push_back(*bc_field);
  }

  FieldState sol = states[primal_solve_state_index].clone(allFields);

  sol.set_eval([=](const gretl::UpstreamStates& inputs, gretl::DownstreamState& output) {
    SMITH_MARK_BEGIN("solve forward");

    const size_t num_states = state_update_weights.size();

    std::vector<size_t> non_primal_to_state_index;
    for (size_t i = 0; i < num_states; ++i) {
      if (i != primal_solve_state_index) {
        non_primal_to_state_index.push_back(i);
      }
    }

    const size_t num_extra_args = have_bc_field ? 2 : 1;
    const size_t num_fields = inputs.size() - num_extra_args;

    std::vector<FEFieldPtr> corrected_fields(num_fields);
    for (size_t field_index = 0; field_index < num_fields; ++field_index) {
      if (field_index < state_update_weights.size() && state_update_weights[field_index] != 0.0) {
        corrected_fields[field_index] = std::make_shared<FiniteElementState>(*inputs[field_index].get<FEFieldPtr>());
      } else {
        corrected_fields[field_index] = inputs[field_index].get<FEFieldPtr>();
      }
    }

    const FEFieldPtr shape_disp_ptr = inputs[num_fields].get<FEFieldPtr>();

    FEFieldPtr bc_field_ptr;
    if (have_bc_field) {
      bc_field_ptr = inputs[num_fields + num_extra_args - 1].get<FEFieldPtr>();
    }

    FEFieldPtr s0 = corrected_fields[primal_solve_state_index];
    FEFieldPtr s = std::make_shared<FiniteElementState>(s0->space(), "s");

    if (bc_manager && (dirichlet_state_index == primal_solve_state_index)) {
      applyBoundaryConditions(time_info.time(), bc_manager, s0, bc_field_ptr);
    }

    s = solver->solve(
        *s0,  // initial guess when solving for the primal index field
        [=](const FiniteElementState& s_) {
          FEFieldPtr primal_field = corrected_fields[primal_solve_state_index];
          *primal_field = s_;

          if (bc_manager && (dirichlet_state_index == primal_solve_state_index)) {
            applyBoundaryConditions(time_info.time(), bc_manager, primal_field, bc_field_ptr);
          }

          for (size_t corrected_field_index : non_primal_to_state_index) {
            if (state_update_weights[corrected_field_index] != 0.0) {
              *corrected_fields[corrected_field_index] = *inputs[corrected_field_index].get<FEFieldPtr>();
              corrected_fields[corrected_field_index]->Add(state_update_weights[corrected_field_index], *primal_field);
              corrected_fields[corrected_field_index]->Add(-state_update_weights[corrected_field_index], *s0);
            }
          }

          auto r = residual_eval->residual(time_info, shape_disp_ptr.get(), getConstFieldPointers(corrected_fields));

          if (bc_manager) {
            if (dirichlet_state_index == primal_solve_state_index) {
              auto constrained_dofs = bc_manager->allEssentialTrueDofs();
              r.SetSubVector(constrained_dofs, 0.0);
            }
          }

          return r;
        },
        [=](const FiniteElementState& s_) {
          FEFieldPtr primal_field = corrected_fields[primal_solve_state_index];
          *primal_field = s_;

          if (bc_manager && (dirichlet_state_index == primal_solve_state_index)) {
            applyBoundaryConditions(time_info.time(), bc_manager, primal_field, bc_field_ptr);
          }

          for (size_t corrected_field_index : non_primal_to_state_index) {
            if (state_update_weights[corrected_field_index] != 0.0) {
              *corrected_fields[corrected_field_index] = *inputs[corrected_field_index].get<FEFieldPtr>();
              corrected_fields[corrected_field_index]->Add(state_update_weights[corrected_field_index], *primal_field);
              corrected_fields[corrected_field_index]->Add(-state_update_weights[corrected_field_index], *s0);
            }
          }

          auto J = residual_eval->jacobian(time_info, shape_disp_ptr.get(), getConstFieldPointers(corrected_fields),
                                           state_update_weights);

          if (bc_manager) {
            if (dirichlet_state_index == primal_solve_state_index) {
              J->EliminateBC(bc_manager->allEssentialTrueDofs(), mfem::Operator::DiagonalPolicy::DIAG_ONE);
            }
          }
          return J;
        });

    output.set<FEFieldPtr, FEDualPtr>(s);

    SMITH_MARK_END("solve forward");
  });

  sol.set_vjp([=](gretl::UpstreamStates& inputs, const gretl::DownstreamState& output) {
    SMITH_MARK_BEGIN("solve reverse");
    const FEFieldPtr s = output.get<FEFieldPtr>();                      // get the final solution
    const FEDualPtr s_dual = output.get_dual<FEDualPtr, FEFieldPtr>();  // get the dual load

    const size_t num_states = state_update_weights.size();

    std::vector<size_t> non_primal_to_state_index;
    for (size_t i = 0; i < num_states; ++i) {
      if (i != primal_solve_state_index) {
        non_primal_to_state_index.push_back(i);
      }
    }

    const size_t num_extra_args = have_bc_field ? 2 : 1;
    const size_t num_fields = inputs.size() - num_extra_args;

    std::vector<FEFieldPtr> corrected_fields(num_fields);
    for (size_t field_index = 0; field_index < num_fields; ++field_index) {
      if (field_index < state_update_weights.size() && state_update_weights[field_index] != 0.0) {
        corrected_fields[field_index] = std::make_shared<FiniteElementState>(*inputs[field_index].get<FEFieldPtr>());
      } else {
        corrected_fields[field_index] = inputs[field_index].get<FEFieldPtr>();
      }
    }

    const FEFieldPtr shape_disp_ptr = inputs[num_fields].get<FEFieldPtr>();

    const FEFieldPtr s0 = inputs[primal_solve_state_index].get<FEFieldPtr>();

    *corrected_fields[primal_solve_state_index] = *s;
    for (size_t corrected_field_index : non_primal_to_state_index) {
      if (state_update_weights[corrected_field_index] != 0.0) {
        corrected_fields[corrected_field_index]->Add(state_update_weights[corrected_field_index], *s);
        corrected_fields[corrected_field_index]->Add(-state_update_weights[corrected_field_index], *s0);
      }
    }

    solver->clearMemory();
    auto J = residual_eval->jacobian(time_info, shape_disp_ptr.get(), getConstFieldPointers(corrected_fields),
                                     state_update_weights, {});

    if (bc_manager) {
      if (dirichlet_state_index == primal_solve_state_index) {
        J->EliminateBC(bc_manager->allEssentialTrueDofs(), mfem::Operator::DiagonalPolicy::DIAG_ONE);
      }
    }

    auto J_T = std::unique_ptr<mfem::HypreParMatrix>(J->Transpose());
    J.reset();

    auto s_adjoint_ptr = solver->solveAdjoint(*s_dual, std::move(J_T));

    if (bc_manager) {
      if (dirichlet_state_index == primal_solve_state_index) {
        s_adjoint_ptr->SetSubVector(bc_manager->allEssentialTrueDofs(), 0.0);
      }
    }

    *s_adjoint_ptr *= -1.0;

    std::vector<DualFieldPtr> field_sensitivities(num_fields, nullptr);
    FEDualPtr shape_disp_sensitivity = inputs[num_fields].get_dual<FEDualPtr, FEFieldPtr>();
    for (size_t state_index = 0; state_index < num_states; ++state_index) {
      field_sensitivities[state_index] = inputs[state_index].get_dual<FEDualPtr, FEFieldPtr>().get();
    }
    for (size_t param_index = num_states; param_index < num_fields; ++param_index) {
      field_sensitivities[param_index] = inputs[param_index].get_dual<FEDualPtr, FEFieldPtr>().get();
    }

    auto primal_sensitivity = std::make_shared<FiniteElementDual>(*field_sensitivities[primal_solve_state_index]);
    field_sensitivities[primal_solve_state_index] = primal_sensitivity.get();
    *field_sensitivities[primal_solve_state_index] = *s_dual;

    residual_eval->vjp(time_info, shape_disp_ptr.get(), getConstFieldPointers(corrected_fields), {},
                       s_adjoint_ptr.get(), shape_disp_sensitivity.get(), field_sensitivities, {});

    if (bc_manager && have_bc_field && dirichlet_state_index == primal_solve_state_index) {
      auto bc_dual_ptr = inputs[num_fields + num_extra_args - 1].get_dual<FEDualPtr, FEFieldPtr>();
      field_sensitivities[primal_solve_state_index]->SetSubVectorComplement(bc_manager->allEssentialTrueDofs(), 0.0);
      *bc_dual_ptr += *field_sensitivities[primal_solve_state_index];
    }

    SMITH_MARK_END("solve reverse");
  });

  sol.finalize();

  return sol;
}

FieldState solve(const WeakForm& weak_form, const FieldState& shape_disp, const std::vector<FieldState>& states,
                 const std::vector<FieldState>& params, const TimeInfo& time_info, const DifferentiableSolver& solver,
                 const DirichletBoundaryConditions& bcs, size_t unknown_state_index)
{
  std::vector<double> state_update_weights(states.size(), 0.0);
  state_update_weights[unknown_state_index] = 1.0;
  return nonlinearSolve(&weak_form, shape_disp, states, params, state_update_weights, unknown_state_index,
                        unknown_state_index, time_info, &solver, &bcs.getBoundaryConditionManager());
}

std::vector<FieldState> block_solve(const std::vector<WeakForm*>& residual_evals,
                                    const std::vector<std::vector<size_t>> block_indices, const FieldState& shape_disp,
                                    const std::vector<std::vector<FieldState>>& states,
                                    const std::vector<std::vector<FieldState>>& params, const TimeInfo& time_info,
                                    const DifferentiableBlockSolver* solver,
                                    const std::vector<BoundaryConditionManager*> bc_managers)
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
    SLIC_ERROR_IF(num_rows != num_param_inputs.size(), "row count for params and columns are inconsistent");

    // The order of inputs in upstreams seems to be:
    // states of residual 0 -> states of residual 1 -> params of residual 0 -> params of residual 1
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
      diagonal_fields[row_i] = std::make_shared<FiniteElementState>(*input_fields[row_i][block_indices[row_i][row_i]]);
    }

    for (size_t row_i = 0; row_i < num_rows; ++row_i) {
      for (size_t col_j = 0; col_j < num_rows; ++col_j) {
        input_fields[row_i][block_indices[row_i][col_j]] = diagonal_fields[col_j];
      }
    }

    const FEFieldPtr shape_disp_ptr = upstreams[field_count].get<FEFieldPtr>();

    auto eval_residuals = [=](const std::vector<FEFieldPtr>& unknowns) {
      SLIC_ERROR_IF(unknowns.size() != num_rows,
                    "block solver unknowns size must match the number or residuals in block_solve");
      std::vector<mfem::Vector> residuals(num_rows);

      for (size_t row_i = 0; row_i < num_rows; ++row_i) {
        FEFieldPtr primal_field_row_i = diagonal_fields[row_i];
        *primal_field_row_i = *unknowns[row_i];
        applyBoundaryConditions(time_info.time(), bc_managers[row_i], primal_field_row_i, nullptr);
      }

      for (size_t row_i = 0; row_i < num_rows; ++row_i) {
        residuals[row_i] = residual_evals[row_i]->residual(time_info, shape_disp_ptr.get(),
                                                           getConstFieldPointers(input_fields[row_i]));
        residuals[row_i].SetSubVector(bc_managers[row_i]->allEssentialTrueDofs(), 0.0);
      }

      return residuals;
    };

    auto eval_jacobians = [=](const std::vector<FEFieldPtr>& unknowns) {
      SLIC_ERROR_IF(unknowns.size() != num_rows,
                    "block solver unknown size must match the number or residuals in block_solve");
      std::vector<std::vector<std::unique_ptr<mfem::HypreParMatrix>>> jacobians(num_rows);

      for (size_t row_i = 0; row_i < num_rows; ++row_i) {
        FEFieldPtr primal_field_row_i = diagonal_fields[row_i];
        *primal_field_row_i = *unknowns[row_i];
        applyBoundaryConditions(time_info.time(), bc_managers[row_i], primal_field_row_i, nullptr);
      }

      for (size_t row_i = 0; row_i < num_rows; ++row_i) {
        std::vector<FEFieldPtr> row_field_inputs = input_fields[row_i];
        std::vector<double> tangent_weights(row_field_inputs.size(), 0.0);
        for (size_t col_j = 0; col_j < num_rows; ++col_j) {
          size_t field_index_to_diff = block_indices[row_i][col_j];
          tangent_weights[field_index_to_diff] = 1.0;
          auto jac_ij = residual_evals[row_i]->jacobian(time_info, shape_disp_ptr.get(),
                                                        getConstFieldPointers(row_field_inputs), tangent_weights);
          jacobians[row_i].emplace_back(std::move(jac_ij));
          tangent_weights[field_index_to_diff] = 0.0;
        }
      }

      // Apply BCs to the block system
      for (size_t row_i = 0; row_i < num_rows; ++row_i) {
        mfem::HypreParMatrix* Jii =
            jacobians[row_i][row_i]->EliminateRowsCols(bc_managers[row_i]->allEssentialTrueDofs());
        delete Jii;
        for (size_t col_j = 0; col_j < num_rows; ++col_j) {
          if (col_j != row_i) {
            jacobians[row_i][col_j]->EliminateRows(bc_managers[row_i]->allEssentialTrueDofs());
            mfem::HypreParMatrix* Jji =
                jacobians[col_j][row_i]->EliminateCols(bc_managers[row_i]->allEssentialTrueDofs());
            delete Jji;
          }
        }
      }

      return jacobians;
    };

    diagonal_fields = solver->solve(diagonal_fields, eval_residuals, eval_jacobians);

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
        input_fields[row_i][block_indices[row_i][col_j]] = diagonal_fields[col_j];
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
        tangent_weights[field_index_to_diff] = 1.0;
        auto jac_ij = residual_evals[row_i]->jacobian(time_info, shape_disp_ptr.get(),
                                                      getConstFieldPointers(row_field_inputs), tangent_weights);
        jacobians[row_i].emplace_back(std::move(jac_ij));
        tangent_weights[field_index_to_diff] = 0.0;
      }
    }

    // Apply BCs to the block system
    for (size_t row_i = 0; row_i < num_rows; ++row_i) {
      s_dual[row_i]->SetSubVector(bc_managers[row_i]->allEssentialTrueDofs(), 0.0);

      mfem::HypreParMatrix* Jii =
          jacobians[row_i][row_i]->EliminateRowsCols(bc_managers[row_i]->allEssentialTrueDofs());
      delete Jii;
      for (size_t col_j = 0; col_j < num_rows; ++col_j) {
        if (col_j != row_i) {
          jacobians[row_i][col_j]->EliminateRows(bc_managers[row_i]->allEssentialTrueDofs());
          mfem::HypreParMatrix* Jji =
              jacobians[col_j][row_i]->EliminateCols(bc_managers[row_i]->allEssentialTrueDofs());
          delete Jji;
        }
      }
    }

    // Take the transpose of the block system
    std::vector<std::vector<std::unique_ptr<mfem::HypreParMatrix>>> jacobians_T(num_rows);
    for (size_t col_j = 0; col_j < num_rows; ++col_j) {
      for (size_t row_i = 0; row_i < num_rows; ++row_i) {
        jacobians_T[col_j].emplace_back(std::unique_ptr<mfem::HypreParMatrix>(jacobians[row_i][col_j]->Transpose()));
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
        field_sensitivities[row_i][block_indices[row_i][col_j]] = nullptr;
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
        [i](const std::vector<FEFieldPtr>& sols) { return std::make_shared<FiniteElementState>(*sols[i]); },
        [i](const std::vector<FEFieldPtr>&, const FEFieldPtr&, std::vector<FEDualPtr>& sols_,
            const FEDualPtr& output_) { *sols_[i] += *output_; },
        sol);

    results.emplace_back(s);
  }

  return results;
}

}  // namespace smith
