// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#include <vector>
#include <memory>
#include "smith/differentiable_numerics/field_state.hpp"
#include "smith/physics/common.hpp"

namespace smith {

class WeakForm;
class DifferentiableBlockSolver;
class BoundaryConditionManager;

class SystemSolver {
public:
  struct Stage {
    std::vector<size_t> block_indices;  // which blocks (residuals) to solve in this stage
    std::shared_ptr<DifferentiableBlockSolver> solver;
  };

  SystemSolver(double global_rel_tol, int global_max_iterations);

  void addStage(const Stage& stage);
  void addStage(const std::vector<size_t>& block_indices, std::shared_ptr<DifferentiableBlockSolver> solver);

  std::vector<FieldState> solve(
      const std::vector<WeakForm*>& residual_evals,
      const std::vector<std::vector<size_t>>& block_indices, 
      const FieldState& shape_disp,
      const std::vector<std::vector<FieldState>>& states,
      const std::vector<std::vector<FieldState>>& params, 
      const TimeInfo& time_info,
      const std::vector<const BoundaryConditionManager*>& bc_managers) const;

private:
  double global_rel_tol_;
  int global_max_iterations_;
  std::vector<Stage> stages_;
};

} // namespace smith