// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

namespace smith {

/// @brief Options controlling adaptive cutback of prescribed BC updates.
///
/// When @c enabled is true, block_solve interpolates between BC values
/// evaluated at @c t_old = time - dt and @c t_new = time using a fraction
/// alpha in [0, 1]. The driver attempts alpha = 1 first; on solver
/// non-convergence or NaN, alpha is shrunk toward the last accepted alpha
/// until the Newton solve converges, then restepped toward 1.
struct BcRampOptions {
  bool enabled = false;                        ///< Master switch. Default off → behavior unchanged.
  double shrink_factor = 0.5;                  ///< Multiplier on (alpha - last_good_alpha) after a failed step.
  int max_cutbacks = 20;                       ///< Hard cap on cutback iterations per outer solve.
  int intermediate_max_iterations = 10;        ///< Nonlinear iteration cap for accepted intermediate ramp solves.
  double intermediate_relative_tol = 0.05;     ///< Relative tolerance for accepted intermediate ramp solves.
  double intermediate_absolute_tol_fac = 1e3;  ///< Absolute tolerance multiplier for intermediate ramp solves.
};

}  // namespace smith
