#pragma once

namespace smith {

/// Cumulative wall-time buckets inside Functional Gradient::assemble(), used to
/// attribute assembleJacobian cost (element kernels vs scatter vs hypre/RAP).
struct GradientAssembleTimers {
  double kernels = 0.0;  ///< element matrix computation
  double scatter = 0.0;  ///< K_e -> local SparseMatrix scatter (SearchRow loop)
  double rap = 0.0;      ///< HypreParMatrix construction + RAP to true dofs
};

/// Process-local cumulative assembly timing counters.
inline GradientAssembleTimers gradient_assemble_timers;

}  // namespace smith
