// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <cmath>
#include <functional>
#include <algorithm>
#include <iterator>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "mpi.h"
#include "gtest/gtest.h"
#include "mfem.hpp"

#include "smith/numerics/odes.hpp"
#include "smith/numerics/stdfunction_operator.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/numerics/equation_solver.hpp"
#include "smith/numerics/functional/finite_element.hpp"
#include "smith/numerics/solver_config.hpp"
#include "smith/physics/boundary_conditions/boundary_condition_manager.hpp"
#include "smith/physics/state/finite_element_state.hpp"

using namespace smith;
using namespace smith::mfem_ext;

const LinearSolverOptions linear_options{.linear_solver = LinearSolver::CG,
                                         .preconditioner = Preconditioner::None,
                                         .relative_tol = 1.0e-12,
                                         .absolute_tol = 1.0e-12,
                                         .max_iterations = 10,
                                         .print_level = 0};

const NonlinearSolverOptions nonlinear_options{
    .relative_tol = 1.0e-12, .absolute_tol = 1.0e-12, .max_iterations = 10, .print_level = -1};

const mfem::DenseMatrix M = []() {
  mfem::DenseMatrix M_mat(3, 3);
  M_mat(0, 0) = 2.0;
  M_mat(0, 1) = 1.0;
  M_mat(0, 2) = 0.0;

  M_mat(1, 0) = 1.0;
  M_mat(1, 1) = 4.0;
  M_mat(1, 2) = 1.0;

  M_mat(2, 0) = 0.0;
  M_mat(2, 1) = 1.0;
  M_mat(2, 2) = 2.0;
  return M_mat;
}();

const mfem::DenseMatrix C = []() {
  mfem::DenseMatrix C_mat(3, 3);
  C_mat(0, 0) = 1.0;
  C_mat(0, 1) = 0.0;
  C_mat(0, 2) = 0.0;

  C_mat(1, 0) = 0.0;
  C_mat(1, 1) = 1.0;
  C_mat(1, 2) = 0.0;

  C_mat(2, 0) = 0.0;
  C_mat(2, 1) = 0.0;
  C_mat(2, 2) = 1.0;
  return C_mat;
}();

const std::function stiffness_linear = [](const mfem::Vector& /*x*/) -> mfem::DenseMatrix {
  mfem::DenseMatrix K(3, 3);
  K(0, 0) = 1.0;
  K(0, 1) = -1.0;
  K(0, 2) = 0.0;

  K(1, 0) = -1.0;
  K(1, 1) = 2.0;
  K(1, 2) = -1.0;

  K(2, 0) = 0.0;
  K(2, 1) = -1.0;
  K(2, 2) = 1.0;
  return K;
};

const std::function internal_force_linear = [](const mfem::Vector& x) -> mfem::Vector {
  mfem::Vector force(x.Size());
  stiffness_linear(x).Mult(x, force);
  return force;
};

const std::function stiffness_nonlinear = [](const mfem::Vector& x) -> mfem::DenseMatrix {
  mfem::DenseMatrix K(3, 3);
  K(0, 0) = 1.0;
  K(0, 1) = -1.0;
  K(0, 2) = 0.0;

  K(1, 0) = -1.0;
  K(1, 1) = 2.0 - 2 * x(1) + x(2);
  K(1, 2) = -1.0 + x(1);

  K(2, 0) = 0.0;
  K(2, 1) = -1.0 + x(2);
  K(2, 2) = 1.0 + x(1) + x(2);
  return K;
};

const std::function internal_force_nonlinear = [](const mfem::Vector& x) -> mfem::Vector {
  mfem::Vector f(3);
  f(0) = x(0) - x(1);
  f(1) = (2.0 - x(1)) * x(1) + (x(1) - 1.0) * x(2) - x(0);
  f(2) = x(1) * (x(2) - 1.0) + (1.0 + 0.5 * x(2)) * x(2);
  return f;
};

const mfem::Vector f_ext = []() {
  mfem::Vector f(3);
  f(0) = 1.0;
  f(1) = 2.0;
  f(2) = 3.0;
  return f;
}();

// continuous constraint with continuous derivative
const auto sine_wave = [](const mfem::Vector& /*x*/, double t) { return 1.0 + sin(2.0 * M_PI * t); };

// continuous constraint with discontinuous derivative
const auto triangle_wave = [](const mfem::Vector& /*x*/, double t) {
  return 1.0 + (2.0 / M_PI) * asin(sin(2.0 * M_PI * t));
};

enum ode_type
{
  LINEAR,
  NONLINEAR
};

std::string to_string(ode_type o)
{
  if (o == LINEAR) return "linear";
  if (o == NONLINEAR) return "nonlinear";
  return "unknown";
}

enum constraint_type
{
  UNCONSTRAINED,
  SINE_WAVE,
  TRIANGLE_WAVE
};

std::string to_string(constraint_type c)
{
  if (c == UNCONSTRAINED) return "unconstrained";
  if (c == SINE_WAVE) return "sine wave";
  if (c == TRIANGLE_WAVE) return "triangle wave";
  return "unknown";
}

std::string to_string(smith::TimestepMethod m)
{
  // for first order odes
  if (m == smith::TimestepMethod::BackwardEuler) return "BackwardEuler";
  if (m == smith::TimestepMethod::SDIRK33) return "SDIRK33";
  if (m == smith::TimestepMethod::ForwardEuler) return "ForwardEuler";
  if (m == smith::TimestepMethod::RK2) return "RK2";
  if (m == smith::TimestepMethod::RK3SSP) return "RK3SSP";
  if (m == smith::TimestepMethod::RK4) return "RK4";
  if (m == smith::TimestepMethod::GeneralizedAlpha) return "GeneralizedAlpha";
  if (m == smith::TimestepMethod::ImplicitMidpoint) return "ImplicitMidpoint";
  if (m == smith::TimestepMethod::SDIRK23) return "SDIRK23";
  if (m == smith::TimestepMethod::SDIRK34) return "SDIRK34";

  // for second order odes
  if (m == smith::TimestepMethod::Newmark) return "Newmark";
  if (m == smith::TimestepMethod::HHTAlpha) return "HHTAlpha";
  if (m == smith::TimestepMethod::WBZAlpha) return "WBZAlpha";
  if (m == smith::TimestepMethod::AverageAcceleration) return "AverageAcceleration";
  if (m == smith::TimestepMethod::LinearAcceleration) return "LinearAcceleration";
  if (m == smith::TimestepMethod::CentralDifference) return "CentralDifference";
  if (m == smith::TimestepMethod::FoxGoodwin) return "FoxGoodwin";
  return "unknown";
}

std::string to_string(smith::DirichletEnforcementMethod m)
{
  // for first order odes
  if (m == smith::DirichletEnforcementMethod::DirectControl) return "direct control";
  if (m == smith::DirichletEnforcementMethod::RateControl) return "rate control";
  if (m == smith::DirichletEnforcementMethod::FullControl) return "full control";
  return "unknown";
}

int order_of_convergence(smith::TimestepMethod m)
{
  // for first order odes
  if (m == smith::TimestepMethod::BackwardEuler) return 1;
  if (m == smith::TimestepMethod::SDIRK33) return 3;
  if (m == smith::TimestepMethod::ForwardEuler) return 1;
  if (m == smith::TimestepMethod::RK2) return 2;
  if (m == smith::TimestepMethod::RK3SSP) return 3;
  if (m == smith::TimestepMethod::RK4) return 4;
  if (m == smith::TimestepMethod::GeneralizedAlpha) return 2;
  if (m == smith::TimestepMethod::ImplicitMidpoint) return 2;
  if (m == smith::TimestepMethod::SDIRK23) return 2;
  if (m == smith::TimestepMethod::SDIRK34) return 3;

  // for second order odes
  if (m == smith::TimestepMethod::Newmark) return 2;
  if (m == smith::TimestepMethod::HHTAlpha) return 2;
  if (m == smith::TimestepMethod::WBZAlpha) return 2;
  if (m == smith::TimestepMethod::AverageAcceleration) return 2;
  if (m == smith::TimestepMethod::LinearAcceleration) return 2;
  if (m == smith::TimestepMethod::CentralDifference) return 2;
  if (m == smith::TimestepMethod::FoxGoodwin) return 2;

  SLIC_ERROR(axom::fmt::format("Unsupported smith::TimestepMethod {0}", static_cast<int>(m)));
  return 0;
}

// does this method apply to a first- or second-order ode?
int which_kind_of_ode(smith::TimestepMethod m)
{
  // for first order odes
  if (m == smith::TimestepMethod::BackwardEuler) return 1;
  if (m == smith::TimestepMethod::SDIRK33) return 1;
  if (m == smith::TimestepMethod::ForwardEuler) return 1;
  if (m == smith::TimestepMethod::RK2) return 1;
  if (m == smith::TimestepMethod::RK3SSP) return 1;
  if (m == smith::TimestepMethod::RK4) return 1;
  if (m == smith::TimestepMethod::GeneralizedAlpha) return 1;
  if (m == smith::TimestepMethod::ImplicitMidpoint) return 1;
  if (m == smith::TimestepMethod::SDIRK23) return 1;
  if (m == smith::TimestepMethod::SDIRK34) return 1;

  // for second order odes
  if (m == smith::TimestepMethod::Newmark) return 2;
  if (m == smith::TimestepMethod::HHTAlpha) return 2;
  if (m == smith::TimestepMethod::WBZAlpha) return 2;
  if (m == smith::TimestepMethod::AverageAcceleration) return 2;
  if (m == smith::TimestepMethod::LinearAcceleration) return 2;
  if (m == smith::TimestepMethod::CentralDifference) return 2;
  if (m == smith::TimestepMethod::FoxGoodwin) return 2;

  SLIC_ERROR(axom::fmt::format("Unsupported smith::TimestepMethod {0}", static_cast<int>(m)));
  return 0;
}

double first_order_ode_test(int nsteps, ode_type type, constraint_type constraint, TimestepMethod timestepper,
                            DirichletEnforcementMethod enforcement)
{
  double t = 0.0;
  double ode_residual_eval_time = 0.0;
  double dt = 1.0 / nsteps;
  double previous_dt = -1.0;
  double c0 = 0.0;

  mfem::Vector x(3);
  mfem::Vector previous(3);
  previous = 0.0;

  mfem::DenseMatrix J(3, 3);

  auto mesh1D = mfem::Mesh::MakeCartesian1D(2);
  mfem::ParMesh mesh(MPI_COMM_WORLD, mesh1D);
  BoundaryConditionManager bcs(mesh);

  // although these test problems don't really apply to a finite element mesh,
  // the tools in Smith require that these finite element data structures exist
  smith::FiniteElementState dummy(mesh, H1<1>{}, "dummy");

  if (constraint == SINE_WAVE) {
    auto coef = std::make_shared<mfem::FunctionCoefficient>(sine_wave);
    bcs.addEssential(std::set<int>{1}, coef, dummy.space());
  }

  if (constraint == TRIANGLE_WAVE) {
    auto coef = std::make_shared<mfem::FunctionCoefficient>(triangle_wave);
    bcs.addEssential(std::set<int>{1}, coef, dummy.space());
  }

  std::function<mfem::Vector(const mfem::Vector&)> f_int;
  std::function<mfem::DenseMatrix(const mfem::Vector&)> K;

  if (type == LINEAR) {
    f_int = internal_force_linear;
    K = stiffness_linear;
  } else {
    f_int = internal_force_nonlinear;
    K = stiffness_nonlinear;
  }

  StdFunctionOperator residual(
      3,
      [&](const mfem::Vector& dx_dt, mfem::Vector& r) {
        // r = M * dx_dt + f_int(x + c0 * dx_dt) - f_ext;

        mfem::Vector x_next(x), M_dx_dt(x);
        x_next.Add(c0, dx_dt);
        auto force_internal = f_int(x_next);
        M.Mult(dx_dt, M_dx_dt);
        add(M_dx_dt, force_internal, r);
        r -= f_ext;
        if (constraint != UNCONSTRAINED) {
          r(0) = 0.0;
        }
      },
      [&](const mfem::Vector& dx_dt) -> mfem::Operator& {
        J = M;
        mfem::Vector x_next(x);
        x_next.Add(c0, dx_dt);
        J.Add(c0, K(x_next));
        if (constraint != UNCONSTRAINED) {
          // clang-format off
        J(0,0) = 1.0; J(0,1) = 0.0; J(0,2) = 0.0;
        J(1,0) = 0.0; 
        J(2,0) = 0.0;
          // clang-format on
        }
        return J;
      });

  EquationSolver solver(nonlinear_options, linear_options);
  solver.setOperator(residual);

  FirstOrderODE ode(dummy.space().TrueVSize(),
                    {.time = ode_residual_eval_time, .u = x, .dt = c0, .du_dt = previous, .previous_dt = previous_dt},
                    solver, bcs);

  ode.SetTimestepper(timestepper);
  ode.SetEnforcementMethod(enforcement);

  mfem::Vector soln(3);

  soln[0] = 1.0;
  soln[1] = 2.0;
  soln[2] = 3.0;

  for (int i = 0; i < nsteps; i++) {
    ode.Step(soln, t, dt);
  }

  // these solutions are computed to machine precision in
  // the wolfram language script: scripts/wolfram/smith_odes_solutions.wls
  //
  // clang-format off
  double exact_solutions[2][3][3] = {
    { // Linear
      {1.716166172752257,2.283833827247743,3.716166172752257},
      {1.00000000000000,2.281907223306997,3.791971854505491},
      {1.00000000000000,2.293071599499620,3.777196781634047}
    },
    { // Nonlinear
      {1.668978479765538,2.689094594698356,1.338823562848719},
      {1.00000000000000,2.888713406376719,1.345973814264978},
      {1.00000000000000,2.869493799926669,1.338893129624946}
    }
  };
  // clang-format on

  mfem::Vector exact_solution(exact_solutions[type][constraint], 3);

  mfem::Vector diff(exact_solution);
  diff -= soln;
  diff /= exact_solution.Norml2();

  return diff.Norml2();
}

double second_order_ode_test(int nsteps, ode_type type, constraint_type constraint, TimestepMethod timestepper,
                             DirichletEnforcementMethod enforcement)
{
  double t = 0.0;
  double ode_residual_eval_time = 0.0;
  double dt = 1.0 / nsteps;
  double c0, c1;

  mfem::Vector x(3);
  mfem::Vector dx_dt(3);
  mfem::Vector previous(3);
  previous = 0.0;

  mfem::DenseMatrix J(3, 3);

  auto mesh1D = mfem::Mesh::MakeCartesian1D(2);
  mfem::ParMesh mesh(MPI_COMM_WORLD, mesh1D);
  BoundaryConditionManager bcs(mesh);

  // although these test problems don't really apply to a finite element mesh,
  // the tools in Smith require that these finite element data structures exist
  smith::FiniteElementState dummy(mesh, H1<1>{}, "dummy");

  if (constraint == SINE_WAVE) {
    auto coef = std::make_shared<mfem::FunctionCoefficient>(sine_wave);
    bcs.addEssential(std::set<int>{1}, coef, dummy.space());
  }

  if (constraint == TRIANGLE_WAVE) {
    auto coef = std::make_shared<mfem::FunctionCoefficient>(triangle_wave);
    bcs.addEssential(std::set<int>{1}, coef, dummy.space());
  }

  std::function<mfem::Vector(const mfem::Vector&)> f_int;
  std::function<mfem::DenseMatrix(const mfem::Vector&)> K;

  if (type == LINEAR) {
    f_int = internal_force_linear;
    K = stiffness_linear;
  } else {
    f_int = internal_force_nonlinear;
    K = stiffness_nonlinear;
  }

  StdFunctionOperator residual(
      3,
      [&](const mfem::Vector& d2x_dt2, mfem::Vector& r) {
        // r = M * d2x_dt2 + C * (dx_dt + c1 * d2x_dt2) + f_int(x + c0 * d2x_dt2) - f_ext;

        mfem::Vector x_next(x), v_next(dx_dt), visc(x), M_d2x_dt2(x);
        x_next.Add(c0, d2x_dt2);
        v_next.Add(c1, d2x_dt2);

        auto force_internal = f_int(x_next);
        C.Mult(v_next, visc);
        M.Mult(d2x_dt2, M_d2x_dt2);
        add(M_d2x_dt2, visc, r);
        r += force_internal;
        r -= f_ext;

        if (constraint != UNCONSTRAINED) {
          r(0) = 0.0;
        }
      },
      [&](const mfem::Vector& d2x_dt2) -> mfem::Operator& {
        mfem::Vector x_next(x);
        x_next.Add(c0, d2x_dt2);
        J = M;
        J.Add(c1, C);
        J.Add(c0, K(x_next));
        if (constraint != UNCONSTRAINED) {
          // clang-format off
        J(0,0) = 1.0; J(0,1) = 0.0; J(0,2) = 0.0;
        J(1,0) = 0.0; 
        J(2,0) = 0.0;
          // clang-format on
        }
        return J;
      });

  EquationSolver solver(nonlinear_options, linear_options);
  solver.setOperator(residual);

  SecondOrderODE ode(dummy.space().TrueVSize(),
                     {.time = ode_residual_eval_time, .c0 = c0, .c1 = c1, .u = x, .du_dt = dx_dt, .d2u_dt2 = previous},
                     solver, bcs);

  ode.SetTimestepper(timestepper);
  ode.SetEnforcementMethod(enforcement);

  mfem::Vector displacement(3);
  displacement[0] = 1.0;
  displacement[1] = 2.0;
  displacement[2] = 3.0;

  mfem::Vector velocity(3);
  velocity[0] = 1.0;
  velocity[1] = -1.0;
  velocity[2] = 0.0;

  // ensure that initial conditions agree with the constraint
  if (constraint == SINE_WAVE) {
    velocity[0] = 2.0 * M_PI;
  }
  if (constraint == TRIANGLE_WAVE) {
    velocity[0] = 4.0;
  }

  for (int i = 0; i < nsteps; i++) {
    ode.Step(displacement, velocity, t, dt);
  }

  // these solutions are computed to machine precision in
  // the wolfram language script: scripts/wolfram/smith_odes_solutions.wls
  //
  // clang-format off
  double exact_displacements[2][3][3] = {
    { // Linear
      {1.890707657808391,1.440430856985806,3.167465070871345},
      {1.00000000000000,2.728890247925751,2.840954141998902},
      {1.00000000000000,1.397131173529700,3.189617467031896}
    },
    { // Nonlinear
      {1.843679427583925,1.603626250564599,1.427673131558248},
      {1.00000000000000,3.090535238075648,0.9902392742386004},
      {1.00000000000000,1.554092379030468,1.449923209556037}
    }
  };

  double exact_velocities[2][3][3] = {
    { // Linear
      {0.6308377824610962,0.002935983263988160,0.1970890557890808},
      {6.283185307179586,-1.339625209685756,1.572179558520680},
      {4.000000000000000,-0.2225975240781133,0.2976678735241257}
    },
    { // Nonlinear
      {0.7131365043381414,0.002179075289509970,-2.116046109865199},
      {6.283185307179586,-0.5523588186531320,-1.112916510614004},
      {4.000000000000000,-0.1879397476414202,-2.037365797741527}
    }
  };
  // clang-format on

  mfem::Vector exact_displacement(exact_displacements[type][constraint], 3);
  mfem::Vector exact_velocity(exact_velocities[type][constraint], 3);

  mfem::Vector displacement_diff(exact_displacement);
  displacement_diff -= displacement;
  displacement_diff /= exact_displacement.Norml2();

  mfem::Vector velocity_diff(exact_velocity);
  velocity_diff -= velocity;
  velocity_diff /= exact_velocity.Norml2();

  return std::max(displacement_diff.Norml2(), velocity_diff.Norml2());
}

using param_t = std::tuple<ode_type, constraint_type, TimestepMethod, DirichletEnforcementMethod>;

class FirstOrderODESuite : public testing::TestWithParam<param_t> {
 protected:
  void SetUp() override { std::tie(type, constraint, timestepper, enforcement) = GetParam(); }
  ode_type type;
  constraint_type constraint;
  TimestepMethod timestepper;
  DirichletEnforcementMethod enforcement;
};

TEST_P(FirstOrderODESuite, All)
{
  int o = order_of_convergence(timestepper);

  // there is enough variety in first order timestep methods
  // that we pick different timesteps for each of convergence
  constexpr int nsteps[5] = {0,  // unused, just here to make indexing work
                             30, 20, 10, 5};

  double errors[3] = {first_order_ode_test(nsteps[o] * 1, type, constraint, timestepper, enforcement),
                      first_order_ode_test(nsteps[o] * 2, type, constraint, timestepper, enforcement),
                      first_order_ode_test(nsteps[o] * 4, type, constraint, timestepper, enforcement)};

  std::string message = axom::fmt::format("running first order test({0}, {1}, {2}, {3}), errors: ({4}, {5}, {6})",
                                          to_string(type), to_string(constraint), to_string(timestepper),
                                          to_string(enforcement), errors[0], errors[1], errors[2]);
  SLIC_INFO(message);

  // this tolerance is pretty loose, since some some of the
  // methods converge pretty slowly, but we still want to
  // check that they're in the right ballpark
  EXPECT_LT(errors[2], 3.0e-3);

  // we check against slightly less than the expected order of convergence
  EXPECT_LT(pow(2.0, 0.9 * order_of_convergence(timestepper)), (errors[1] / errors[2]));
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(AllFirstOrderTests, FirstOrderODESuite,
  testing::Combine(
    testing::Values(
      LINEAR, 
      NONLINEAR
    ), 
    testing::Values(
      UNCONSTRAINED, 
      SINE_WAVE
    ), 
    testing::Values(
      smith::TimestepMethod::BackwardEuler,
      smith::TimestepMethod::SDIRK33,
      smith::TimestepMethod::ForwardEuler,
      smith::TimestepMethod::RK2,
      smith::TimestepMethod::RK3SSP,
      smith::TimestepMethod::RK4,
      smith::TimestepMethod::GeneralizedAlpha,
      smith::TimestepMethod::ImplicitMidpoint,
      smith::TimestepMethod::SDIRK23,
      smith::TimestepMethod::SDIRK34
    ),
    testing::Values(
      DirichletEnforcementMethod::DirectControl,       
      DirichletEnforcementMethod::RateControl,     
      DirichletEnforcementMethod::FullControl     
    )
  )
);
// clang-format on

class SecondOrderODESuite : public testing::TestWithParam<param_t> {
 protected:
  void SetUp() override { std::tie(type, constraint, timestepper, enforcement) = GetParam(); }
  ode_type type;
  constraint_type constraint;
  TimestepMethod timestepper;
  DirichletEnforcementMethod enforcement;
};

// these cases do not satisfy the stability criteria of the DirectControl option
std::vector unstable_cases = {
    std::tuple{SINE_WAVE, TimestepMethod::CentralDifference, DirichletEnforcementMethod::DirectControl},
    std::tuple{SINE_WAVE, TimestepMethod::FoxGoodwin, DirichletEnforcementMethod::DirectControl},
    std::tuple{SINE_WAVE, TimestepMethod::LinearAcceleration, DirichletEnforcementMethod::DirectControl}};

TEST_P(SecondOrderODESuite, All)
{
  bool expected_to_fail = std::find(std::begin(unstable_cases), std::end(unstable_cases),
                                    std::tuple{constraint, timestepper, enforcement}) != std::end(unstable_cases);

  if (!expected_to_fail) {
    double errors[3] = {second_order_ode_test(30, type, constraint, timestepper, enforcement),
                        second_order_ode_test(60, type, constraint, timestepper, enforcement),
                        second_order_ode_test(120, type, constraint, timestepper, enforcement)};

    std::string message = axom::fmt::format("running second order test({0}, {1}, {2}, {3}), errors: ({4}, {5}, {6})",
                                            to_string(type), to_string(constraint), to_string(timestepper),
                                            to_string(enforcement), errors[0], errors[1], errors[2]);

    SLIC_INFO(message);

    EXPECT_LT(errors[2], 1.0e-3);

    // we check against slightly less than the expected order of convergence
    EXPECT_LT(pow(2.0, 0.8 * order_of_convergence(timestepper)), (errors[1] / errors[2]));
  }
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(
    AllSecondOrderTsts, SecondOrderODESuite,
    testing::Combine(
      testing::Values(
        LINEAR, 
        NONLINEAR
      ), 
      testing::Values(
        UNCONSTRAINED, 
        SINE_WAVE
      ),
      testing::Values(
        smith::TimestepMethod::Newmark, 
        smith::TimestepMethod::HHTAlpha,
        smith::TimestepMethod::WBZAlpha, 
        smith::TimestepMethod::LinearAcceleration,
        smith::TimestepMethod::CentralDifference, 
        smith::TimestepMethod::FoxGoodwin
      ),
      testing::Values(
        DirichletEnforcementMethod::DirectControl, 
        DirichletEnforcementMethod::RateControl,
        DirichletEnforcementMethod::FullControl)
      )
    );
// clang-format on

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
