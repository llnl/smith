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
#include "smith/differentiable_numerics/differentiable_solver.hpp"
// #include "smith/differentiable_numerics/solid_mechanics_state_advancer.hpp"
#include "smith/differentiable_numerics/field_state.hpp"
#include "smith/differentiable_numerics/state_advancer.hpp"
#include "smith/differentiable_numerics/time_discretized_weak_form.hpp"
#include "smith/differentiable_numerics/time_integration_rule.hpp"
#include "smith/differentiable_numerics/tests/paraview_helper.hpp"
#include "smith/differentiable_numerics/reaction.hpp"
#include "smith/differentiable_numerics/nonlinear_solve.hpp"

#include "calculate_reactions.hpp"
#include "viscous_solid_mechanics.hpp"
#include "custom_materials.hpp"

#include <vector>
#include <cmath>
#include <stdexcept>
#include <string>
#include <sstream>
#include <iomanip>

namespace smith {
static constexpr int dim = 3;
static constexpr int order = 1;
struct Problem_Params {
  std::string problem_name = "";
  std::string mesh_name = "";
  int N_steps = 200;
  double T = 2.0;
  std::function<double(double, tensor<double, dim>)> boundary_condition = [](double /*t*/, tensor<double, dim> /*X*/) {
    return 0.0;
  };
  bool use_2D = false;
  double Lx = 50.0;
  double Ly = 25.0;
  double Lz = 50.0;
  double viscosity_parameter = 1.0e1;
};

template <typename T>
std::vector<T> logspace(T a, T b, std::size_t N)
{
  static_assert(std::is_floating_point_v<T>, "logspace requires a floating-point type");

  if (N == 0) return {};

  if (N == 1) return {a};

  if (a <= T(0) || b <= T(0)) throw std::domain_error("logspace requires a > 0 and b > 0");

  std::vector<T> result;
  result.reserve(N);

  const T log_a = std::log10(a);
  const T log_b = std::log10(b);
  const T step = (log_b - log_a) / static_cast<T>(N - 1);

  for (std::size_t i = 0; i < N; ++i) {
    result.push_back(std::pow(T(10), log_a + step * static_cast<T>(i)));
  }

  return result;
}

inline std::string to_string_prec16(double value)
{
  std::ostringstream oss;
  oss << std::setprecision(16) << value;
  return oss.str();
}

smith::LinearSolverOptions solid_linear_options{.linear_solver = smith::LinearSolver::CG,
                                                //
                                                .preconditioner = smith::Preconditioner::HypreJacobi,
                                                .relative_tol = 1e-8,
                                                .absolute_tol = 1e-11,
                                                .max_iterations = 10000,
                                                .print_level = 1};

smith::NonlinearSolverOptions solid_nonlinear_opts{.nonlin_solver = NonlinearSolver::TrustRegion,
                                                   .relative_tol = 1.0e-8,
                                                   .absolute_tol = 1.0e-11,
                                                   .max_iterations = 500,
                                                   .print_level = 1};

using ShapeDispSpace = H1<1, dim>;
using VectorSpace = H1<order, dim>;
using ScalarParameterSpace = L2<0>;

int SolveProblem(const Problem_Params& problem_params)
{
  axom::sidre::DataStore datastore;
  std::shared_ptr<smith::Mesh> mesh;
  std::string mesh_tag = problem_params.mesh_name;
  smith::StateManager::initialize(datastore, mesh_tag);

  std::string mesh_location = SMITH_REPO_DIR "/korner_examples/" + mesh_tag + ".g";
  int serial_refinement = 0;
  int parallel_refinement = 0;
  mesh = make_shared<smith::Mesh>(smith::buildMeshFromFile(mesh_location), mesh_tag, serial_refinement,
                                  parallel_refinement);
  mesh->addDomainOfBoundaryElements("fix_bottom", smith::by_attr<dim>(2));
  mesh->addDomainOfBoundaryElements("fix_top", smith::by_attr<dim>(3));
  mesh->addDomainOfBoundaryElements("fix_front", smith::by_attr<dim>(4));
  mesh->addDomainOfBoundaryElements("fix_back", smith::by_attr<dim>(5));
  mesh->addDomainOfBoundaryElements("fix_right", smith::by_attr<dim>(6));
  mesh->addDomainOfBoundaryElements("fix_left", smith::by_attr<dim>(7));

  std::string physics_name = "solid_" + mesh_tag;

  std::shared_ptr<DifferentiableSolver> d_solid_nonlinear_solver =
      buildDifferentiableNonlinearSolve(solid_nonlinear_opts, solid_linear_options, *mesh);

  smith::SecondOrderTimeIntegrationRule time_rule(1.0);

  // warm-start.
  // implicit Newmark.

  auto [physics, weak_form, bcs] =
      custom_physics::buildSolidMechanics<dim, ShapeDispSpace, VectorSpace, ScalarParameterSpace, ScalarParameterSpace>(
          mesh, d_solid_nonlinear_solver, time_rule, physics_name, {"bulk", "shear"});

  bcs->setFixedVectorBCs<dim>(mesh->domain("fix_bottom"));
  bcs->setVectorBCs<dim>(mesh->domain("fix_top"), [problem_params](double t, tensor<double, dim> X) {
    auto output = 0.0 * X;
    output[1] = problem_params.boundary_condition(t, X);
    return output;
  });
  if (problem_params.use_2D) {
    bcs->setFixedVectorBCs<dim>(mesh->entireBody(), {2});
  }

  double E = 100.0;
  double nu = 0.25;
  auto K = E / (3.0 * (1.0 - 2 * nu));
  auto G = E / (2.0 * (1.0 + nu));
  using MaterialType = ParameterizedNeoHookeanWithViscosity;
  MaterialType material{.density = 10.0, .K0 = K, .G0 = G, .eta = problem_params.viscosity_parameter};

  weak_form->addBodyIntegral(
      smith::DependsOn<0, 1>{}, mesh->entireBodyName(),
      [material](const auto& /*time_info*/, auto /*X*/, auto u, auto v, auto /*a*/, auto bulk, auto shear) {
        MaterialType::State state;
        auto du_dX = get<DERIVATIVE>(u);
        auto Grad_v = get<DERIVATIVE>(v);
        auto pk_stress = material(state, du_dX, Grad_v, bulk, shear);
        // return smith::tuple{get<VALUE>(a) * material.density, pk_stress};
        return smith::tuple{smith::zero{}, pk_stress};
      });

  auto shape_disp = physics->getShapeDispFieldState();
  auto params = physics->getFieldParams();
  auto states = physics->getInitialFieldStates();

  params[0].get()->setFromFieldFunction([=](smith::tensor<double, dim>) {
    double scaling = 1.0;
    return scaling * material.K0;
  });

  params[1].get()->setFromFieldFunction([=](smith::tensor<double, dim>) {
    double scaling = 1.0;
    return scaling * material.G0;
  });

  physics->resetStates();

  double T = problem_params.T;
  double time_increment = T / (static_cast<double>(problem_params.N_steps));
  std::string paraview_output_name =
      physics_name + "/" + physics_name + to_string_prec16(problem_params.viscosity_parameter);
  auto pv_writer = smith::createParaviewOutput(*mesh, physics->getFieldStatesAndParamStates(), paraview_output_name);
  pv_writer.write(0, physics->time(), physics->getFieldStatesAndParamStates());

  std::ofstream file;

  std::string stress_strain_output =
      physics_name + "/" + to_string_prec16(problem_params.viscosity_parameter) + "_strain_curve.csv";

  if (mfem::Mpi::Root()) {
    file = std::ofstream(stress_strain_output);

    if (!file.is_open()) {
      MFEM_ABORT("Could Not Open File");
    }
    file << std::setprecision(16) << std::scientific;

    file << "time,strain,force\n";
  }

  for (int cnt = 1; cnt < (problem_params.N_steps + 1); cnt++) {
    if (mfem::Mpi::Root()) {
      std::cout << "Time Step: " << cnt << ", Time: " << physics->time() << std::endl;
    }
    physics->advanceTimestep(time_increment);

    TimeInfo time_info(physics->time(), time_increment);
    auto reactions =
        physics->getStateAdvancer()->computeResultants(shape_disp, physics->getFieldStates(), params, time_info);
    double reaction = CalculateReaction(*reactions[0].get(), mesh, "fix_top", 1);
    if (mfem::Mpi::Root()) {
      std::cout << "Reaction: " << reaction << std::endl;
      file << time_info.time() << ","
           << problem_params.boundary_condition(time_info.time(), {0, 0, 0}) / problem_params.Ly << ","
           << reaction / (problem_params.Lx * problem_params.Lz) << "\n";
      file.flush();
    }
    pv_writer.write(cnt, physics->time(), physics->getFieldStatesAndParamStates());
  }

  return 0;
}

};  // namespace smith
int main(int argc, char* argv[])
{
  using namespace smith;
  smith::ApplicationManager applicationManager(argc, argv);
  SMITH_MARK_FUNCTION;

  if (argc < 2) {
    std::cerr << "Usage: program <nominal, optimal, snap_array_nominal, snap_array_optimal>\n";

    return 1;
  }

  Problem_Params problem_params;
  std::string problem_tag = argv[1];
  if (problem_tag == "nominal") {
    // Solving Nominal Hole Array Cyclic Test
    problem_params.problem_name = "nominal";
    problem_params.mesh_name = "nominal";
    problem_params.boundary_condition = [](double t, tensor<double, dim> /*X*/) -> double {
      double def_mag = -10.;
      double tstar = 1.0;
      if (t < 1.0) {
        return def_mag * t;
      } else {
        return def_mag * (2.0 * tstar - t);
      }
    };
    problem_params.Lx = 52.0;
    problem_params.Ly = 52.0;
    problem_params.Lz = 25.0;

  } else if (problem_tag == "optimal") {
    // Solving Optimal Hole Array Cyclic Test
    problem_params.problem_name = "optimal";
    problem_params.mesh_name = "optimal";
    problem_params.boundary_condition = [](double t, tensor<double, dim> /*X*/) -> double {
      double def_mag = -10.;
      double tstar = 1.0;
      if (t < 1.0) {
        return def_mag * t;
      } else {
        return def_mag * (2.0 * tstar - t);
      }
    };
    problem_params.Lx = 52.0;
    problem_params.Ly = 52.0;
    problem_params.Lz = 25.0;

  } else if (problem_tag == "snap_array_nominal") {
    // Solving Snap Array Nominal Hole Array Cyclic Test
    problem_params.problem_name = "snap_array_nominal";
    problem_params.mesh_name = "snap_array_nominal";
    problem_params.boundary_condition = [](double t, tensor<double, dim> /*X*/) -> double {
      double def_mag = -159.6;
      double tstar = 1.0;
      if (t < 1.0) {
        return def_mag * t;
      } else {
        return def_mag * (2.0 * tstar - t);
      }
    };
    problem_params.Lx = 819.0;
    problem_params.Ly = 830.0;
    problem_params.Lz = 276.66;

  } else if (problem_tag == "snap_array_optimal") {
    // Solving Snap Array Optimal Hole Array Cyclic Test
    problem_params.problem_name = "snap_array_optimal";
    problem_params.mesh_name = "snap_array_optimal";
    problem_params.boundary_condition = [](double t, tensor<double, dim> /*X*/) -> double {
      double def_mag = -159.6;
      double tstar = 1.0;
      if (t < 1.0) {
        return def_mag * t;
      } else {
        return def_mag * (2.0 * tstar - t);
      }
    };
    problem_params.Lx = 819.0;
    problem_params.Ly = 830.0;
    problem_params.Lz = 276.66;
  } else {
    std::cerr << "Undefined Problem Input. Use: nominal, optimal, snap_array_nominal, snap_array_optimal";
  }

  double viscosity_min = 1.0e-4;
  double viscosity_max = 1.0e1;
  std::size_t N_solves = 10;

  std::vector<double> viscosity_vals = logspace(viscosity_min, viscosity_max, N_solves);
  viscosity_vals[0] = 0.0;
  for (std::size_t i = 0; i < N_solves; i++) {
    problem_params.viscosity_parameter = viscosity_vals[i];
    SolveProblem(problem_params);
  }

  return 0;
}
