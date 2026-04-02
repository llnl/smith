#pragma once

#include "gretl/data_store.hpp"

#include "smith/smith_config.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/numerics/equation_solver.hpp"
#include "smith/numerics/solver_config.hpp"
#include "smith/mesh_utils/mesh_utils.hpp"

#include "smith/physics/state/state_manager.hpp"
#include "smith/physics/functional_objective.hpp"
#include "smith/physics/boundary_conditions/boundary_condition_manager.hpp"
#include "smith/physics/materials/parameterized_solid_material.hpp"

#include "smith/differentiable_numerics/differentiable_physics.hpp"
#include "smith/differentiable_numerics/dirichlet_boundary_conditions.hpp"
#include "smith/differentiable_numerics/differentiable_solver.hpp"
// #include "smith/differentiable_numerics/solid_mechanics_state_advancer.hpp"
#include "smith/differentiable_numerics/field_state.hpp"
#include "smith/differentiable_numerics/state_advancer.hpp"
#include "smith/differentiable_numerics/time_discretized_weak_form.hpp"
#include "smith/differentiable_numerics/time_integration_rule.hpp"
#include "smith/differentiable_numerics/tests/paraview_helper.hpp"
#include "smith/differentiable_numerics/reaction.hpp"
#include "smith/differentiable_numerics/nonlinear_solve.hpp"

using namespace smith;
struct ParameterizedNeoHookeanWithViscosity {
  using State = Empty;

  template <int d, typename DispGradType, typename VelGradType, typename BulkType, typename ShearType>
  SMITH_HOST_DEVICE auto operator()(State& /*state*/, const smith::tensor<DispGradType, d, d>& du_dX,
                                    const smith::tensor<VelGradType, d, d>& dv_dX, const BulkType& DeltaK,
                                    const ShearType& DeltaG) const
  {
    using std::log1p;
    constexpr auto I = Identity<d>();

    auto K_eff = K0 + get<0>(DeltaK);
    auto G_eff = G0 + get<0>(DeltaG);
    auto lambda = K_eff - (2.0 / d) * G_eff;

    auto F = du_dX + I;

    auto grad_v = dv_dX * inv(F);
    auto B_minus_I = du_dX * transpose(du_dX) + transpose(du_dX) + du_dX;
    auto logJ = log1p(detApIm1(du_dX));
    auto TK_elastic = lambda * logJ * I + G_eff * B_minus_I;

    auto D = 0.5 * (grad_v + transpose(grad_v));
    auto TK_viscous = 2.0 * eta * det(F) * D;

    auto TK = TK_elastic + TK_viscous;
    return dot(TK, inv(transpose(F)));
  }

  static constexpr int numParameters() { return 2; }

  double density;
  double K0;
  double G0;
  double eta;
};


struct ParameterizedHolzapfelViscoelastic {
  struct State {
     tensor<double,3,3> H1n{{{1.0, 0.0, 0.0},
                            {0.0, 1.0, 0.0},
                            {0.0, 0.0, 1.0}}}; // previous value of Hn tensor
   };

  template <int d, typename DispGradType, typename BulkType, typename ShearType>
  SMITH_HOST_DEVICE auto operator()(State& state, const smith::tensor<DispGradType, d, d>& du_dX,
                                    const BulkType& DeltaK, const ShearType& DeltaG) const
  {
    using std::pow;
    using std::exp;

    auto H1n = state.H1n; // previous value of the H^\alpha_n tensor, for alpha=1
  
    auto kappa = kappa0 + get<0>(DeltaK);
    auto mu = mu0 + get<0>(DeltaG);

    // get kinematics
    constexpr auto I = Identity<d>();
    auto F = du_dX + I;
    auto J = det(F);

    auto C = dot(transpose(F), F);
    auto trC = tr(C);
    auto Ci = inv(C);
    auto Ch = I - Ci*trC/3.;

    // calculate 2nd PK stress
    auto Svol = kappa*J*(J-1.)*Ci;
    auto Siso = Ch*mu*pow(J, -2./3.);
    auto Q1 = Siso*beta1*exp(-dt/(2.*tau1)) + H1n;
    auto S = Svol + Siso + Q1;

    // create the current H1
    auto H1 = exp(-dt/(2.*tau1))*(Q1*exp(-dt/(2.*tau1)) - beta1*Siso);

    // get kirchhoff stress
    auto TK = dot(F,dot(S,transpose(F)));
   
    // Pull back to Piola and store the current H1 as the state H1n
    state.H1n = get_value(H1);
    return dot(TK, inv(transpose(F)));
  }

  static constexpr int numParameters() { return 2; }

  double density;
  double kappa0;     ///< bulk modulus
  double mu0;        ///< shear modulus
  double beta1;      ///< strain energy factor - assuming only 1 term
  double tau1;       ///< viscoelastic time constant - assuming only 1 term
  double dt;         ///< fixed dt
};
