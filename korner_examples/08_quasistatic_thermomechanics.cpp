#include <iostream>

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
#include "smith/differentiable_numerics/nonlinear_block_solver.hpp"
// #include "smith/differentiable_numerics/solid_mechanics_state_advancer.hpp"
#include "smith/differentiable_numerics/field_state.hpp"
#include "smith/differentiable_numerics/state_advancer.hpp"
#include "smith/differentiable_numerics/time_discretized_weak_form.hpp"
#include "smith/differentiable_numerics/time_integration_rule.hpp"
#include "smith/differentiable_numerics/paraview_writer.hpp"
#include "smith/differentiable_numerics/reaction.hpp"
#include "smith/differentiable_numerics/nonlinear_solve.hpp"
#include "quasistatic_thermomechanics.hpp"
#include "calculate_reactions.hpp"

namespace smith {

smith::LinearSolverOptions solid_linear_options{.linear_solver = smith::LinearSolver::CG,
                                                //
                                                .preconditioner = smith::Preconditioner::HypreJacobi,
                                                .relative_tol = 1e-8,
                                                .absolute_tol = 1e-11,
                                                .max_iterations = 10000,
                                                .print_level = 2};

smith::NonlinearSolverOptions solid_nonlinear_opts{.nonlin_solver = NonlinearSolver::TrustRegion,
                                                   .relative_tol = 1.0e-8,
                                                   .absolute_tol = 1.0e-11,
                                                   .max_iterations = 500,
                                                   .print_level = 2};

struct NeoHookeanThermoelasticMaterial {
  static constexpr int dim = 3;
  double density;    ///< density
  double E;          ///< Young's modulus
  double nu;         ///< Poisson's ratio
  double C_v;        ///< volumetric heat capacity
  double alpha;      ///< thermal expansion coefficient
  double theta_ref;  ///< datum temperature for thermal expansion
  double kappa;      ///< thermal conductivity
  double mu;         ///< viscous parameter

  using State = Empty;
  /**
   * @brief Evaluate constitutive variables for thermomechanics
   *
   * @tparam T1 Type of the displacement gradient components (number-like)
   * @tparam T2 Type of the temperature (number-like)
   * @tparam T3 Type of the temperature gradient components (number-like)
   *
   * @param[in] grad_u Displacement gradient
   * @param[in] theta Temperature
   * @param[in] grad_theta Temperature gradient
   * @param[in,out] state State variables for this material
   *
   * @return[out] tuple of constitutive outputs. Contains the
   * First Piola stress, the volumetric heat capacity in the reference
   * configuration, the heat generated per unit volume during the time
   * step (units of energy), and the referential heat flux (units of
   * energy per unit time and per unit area).
   */
  template <typename T1, typename T2, typename T3, typename T4, int dim>
  auto operator()(State&, double /*dt*/, const tensor<T1, dim, dim>& grad_u, const tensor<T2, dim, dim>& grad_v,
                  T3 theta, const tensor<T4, dim>& grad_theta) const
  {
    using std::log1p;
    // std::cout << "dt = " << dt << "\n";
    // constexpr double eps = 1.0e-12;
    const double K = E / (3.0 * (1.0 - 2.0 * nu));
    const double G = 0.5 * E / (1.0 + nu);
    constexpr auto I = DenseIdentity<dim>();
    auto lambda = K - (2.0 / 3.0) * G;
    auto B_minus_I = dot(grad_u, transpose(grad_u)) + transpose(grad_u) + grad_u;

    auto logJ = log1p(detApIm1(grad_u));
    // Kirchoff stress, in form that avoids cancellation error when F is near I

    // Pull back to Piola
    auto F = grad_u + I;

    auto L = dot(grad_v, inv(F));
    auto D = sym(L);

    auto TK = lambda * logJ * I + G * B_minus_I + 0.5 * det(F) * mu * D;  // dot(L, inv(transpose(F)));

    // state.F_old = get_value(F);
    const auto S = -1.0 * K * (dim * alpha * (theta - theta_ref)) * I;
    auto Piola = dot(TK, inv(transpose(F))) + dot(F, S);
    // internal heat power
    auto greenStrainRate =
        0.5 * (grad_v + transpose(grad_v) + dot(transpose(grad_v), grad_u) + dot(transpose(grad_u), grad_v));
    const auto s0 = -dim * K * alpha * (theta + 273.1) * tr(greenStrainRate);
    // const auto s0 = -dim * K * alpha * (theta + 273.1);

    // heat flux
    const auto q0 = -kappa * grad_theta;
    return tuple{Piola, C_v, s0, q0};
  }
};

static constexpr int dim = 3;
static constexpr int order = 1;

using ShapeDispSpace = H1<1, dim>;
using VectorSpace = H1<order, dim>;
using ScalarSpace = H1<order, 1>;
using ScalarParameterSpace = L2<0>;
}  // namespace smith
int main(int argc, char* argv[])
{
  using namespace smith;
  smith::ApplicationManager applicationManager(argc, argv);
  SMITH_MARK_FUNCTION;

  axom::sidre::DataStore datastore;
  std::shared_ptr<smith::Mesh> mesh;
  std::string mesh_tag = "nominal";
  smith::StateManager::initialize(datastore, mesh_tag);

  std::string mesh_location = SMITH_REPO_DIR "/korner_examples/" + mesh_tag + ".g";
  int serial_refinement = 0;
  int parallel_refinement = 0;
  mesh = std::make_shared<smith::Mesh>(smith::buildMeshFromFile(mesh_location), mesh_tag, serial_refinement,
                                  parallel_refinement);
  mesh->addDomainOfBoundaryElements("fix_bottom", smith::by_attr<dim>(2));
  mesh->addDomainOfBoundaryElements("fix_top", smith::by_attr<dim>(3));
  mesh->addDomainOfBoundaryElements("fix_front", smith::by_attr<dim>(4));
  mesh->addDomainOfBoundaryElements("fix_back", smith::by_attr<dim>(5));
  mesh->addDomainOfBoundaryElements("fix_right", smith::by_attr<dim>(6));
  mesh->addDomainOfBoundaryElements("fix_left", smith::by_attr<dim>(7));

  std::string physics_name = "solid_" + mesh_tag;

  auto d_solid_nonlinear_solver = buildNonlinearBlockSolver(solid_nonlinear_opts, solid_linear_options, *mesh);

  auto d_thermal_nonlinear_solver = buildNonlinearBlockSolver(solid_nonlinear_opts, solid_linear_options, *mesh);

  smith::ImplicitNewmarkSecondOrderTimeIntegrationRule solid_time_rule;
  smith::ImplicitNewmarkSecondOrderTimeIntegrationRule thermal_time_rule;

  // warm-start.
  // implicit Newmark.
  //
  auto [physics, solid_mechanics_weak_form, thermal_mechanics_weak_form, vector_bcs, scalar_bcs] =
      custom_physics::buildThermoMechanics<dim, ShapeDispSpace, VectorSpace, ScalarSpace, ScalarParameterSpace>(
          mesh, d_solid_nonlinear_solver, d_thermal_nonlinear_solver, solid_time_rule, thermal_time_rule, physics_name,
          {"bulk"});

  vector_bcs->setFixedVectorBCs<dim>(mesh->domain("fix_bottom"));
  vector_bcs->setVectorBCs<dim>(mesh->domain("fix_top"), [](double t, smith::tensor<double, dim> X) {
    auto bc = 0.0 * X;
    bc[1] = -10.0 * t;
    return bc;
  });

  double rho = 1.0;
  double alpha = 1.0e-3;
  double theta_ref = 0.0;
  double k = 1.0;
  double mu = 0.0;
  double c = 1.0;
  double E = 100.0;
  double nu = 0.25;
  // auto K = E / (3.0 * (1.0 - 2 * nu));
  // auto G = E / (2.0 * (1.0 + nu));
  using MaterialType = NeoHookeanThermoelasticMaterial;
  MaterialType material = MaterialType{rho, E, nu, c, alpha, theta_ref, k, mu};
  // using MaterialType = ParameterizedNeoHookeanWithViscosity;
  // MaterialType material{.density = 10.0, .K0 = K, .G0 = G, .eta = 0.0e0};

  solid_mechanics_weak_form->addBodyIntegral(
      smith::DependsOn<0, 1, 2, 3>{}, mesh->entireBodyName(),
      [material](const auto& time_info, auto /*X*/, auto u, auto v, auto /*a*/, auto theta, auto /*theta_dot*/,
                 auto /*theta_dot_dot*/, auto /*bulk*/) {
        MaterialType::State state;
        const double dt = time_info.dt();
        auto Grad_u = get<DERIVATIVE>(u);
        auto Grad_v = get<DERIVATIVE>(v);
        auto theta_l = get<VALUE>(theta);
        auto Grad_theta = get<DERIVATIVE>(theta);
        auto [pk, C_v, s0, q0] = material(state, dt, Grad_u, Grad_v, theta_l, Grad_theta);
        // return smith::tuple{get<VALUE>(a) * material.density, pk_stress};
        return smith::tuple{smith::zero{}, pk};
      });

  thermal_mechanics_weak_form->addBodyIntegral(
      smith::DependsOn<0, 1, 2, 3>{}, mesh->entireBodyName(),
      [material](const auto& time_info, auto /*X*/, auto theta, auto theta_dot, auto /*theta_dot_dot*/, auto u, auto v,
                 auto /*a*/, auto /*bulk*/) {
        MaterialType::State state;
        const double dt = time_info.dt();
        auto Grad_u = get<DERIVATIVE>(u);
        auto Grad_v = get<DERIVATIVE>(v);
        auto theta_l = get<VALUE>(theta);
        auto Grad_theta = get<DERIVATIVE>(theta);
        auto [pk, C_v, s0, q0] = material(state, dt, Grad_u, Grad_v, theta_l, Grad_theta);
        auto dtheta_dt = get<VALUE>(theta_dot);
        // return smith::tuple{get<VALUE>(a) * material.density, pk_stress};
        return smith::tuple{C_v * dtheta_dt - s0, -q0};
      });

  auto params = physics->getFieldParams();

  params[0].get()->setFromFieldFunction([=](smith::tensor<double, dim>) {
    double scaling = 1.0;
    return scaling;
  });

  physics->resetStates();

  double time_increment = 1.0e-2;
  auto pv_writer = smith::createParaviewWriter(*mesh, physics->getFieldStatesAndParamStates(), physics_name);
  pv_writer.write(0, physics->time(), physics->getFieldStatesAndParamStates());
  double T = 1.0;
  int cnt = 0;
  while (physics->time() < T) {
    cnt++;

    if (mfem::Mpi::Root()) {
      std::cout << "Time Step: " << cnt << ", Time: " << physics->time() << std::endl;
    }
    physics->advanceTimestep(time_increment);

    auto reactions = physics->getReactionStates();
    double reaction = CalculateReaction(*reactions[0].get(), mesh, "fix_top", 1);
    if (mfem::Mpi::Root()) {
      std::cout << "Reaction: " << reaction << std::endl;
    }
    pv_writer.write(cnt, physics->time(), physics->getFieldStatesAndParamStates());
  }

  auto reactions = physics->getReactionStates();

  auto disp_squared = innerProduct(reactions[0], reactions[0]);

  gretl::set_as_objective(disp_squared);
  if (mfem::Mpi::Root()) {
    std::cout << "final disp norm2 = " << disp_squared.get() << std::endl;
  }

  return 0;
}
