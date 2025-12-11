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

namespace smith {

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

static constexpr int dim = 3;
static constexpr int order = 1;

using ShapeDispSpace = H1<1, dim>;
using VectorSpace = H1<order, dim>;
using ScalarParameterSpace = L2<0>;

};  // namespace smith

int main(int argc, char* argv[])
{
  using namespace smith;
  smith::ApplicationManager applicationManager(argc, argv);
  SMITH_MARK_FUNCTION;

  axom::sidre::DataStore datastore;
  std::shared_ptr<smith::Mesh> mesh;
  std::string mesh_tag = "snap_array_nominal";
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
  bcs->setVectorBCs<dim>(mesh->domain("fix_top"), [](double t, smith::tensor<double, dim> X) {
    auto bc = 0.0 * X;
    bc[1] = -200.0 * t;
    return bc;
  });

  double E = 100.0;
  double nu = 0.25;
  auto K = E / (3.0 * (1.0 - 2 * nu));
  auto G = E / (2.0 * (1.0 + nu));
  using MaterialType = ParameterizedNeoHookeanWithViscosity;
  MaterialType material{.density = 10.0, .K0 = K, .G0 = G, .eta = 0.0e0};

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

  double time_increment = 1.0e-2;
  auto pv_writer = smith::createParaviewOutput(*mesh, physics->getFieldStatesAndParamStates(), physics_name);
  pv_writer.write(0, physics->time(), physics->getFieldStatesAndParamStates());
  double T = 1.0;
  int cnt = 0;
  while (physics->time() < T) {
    cnt++;

    if (mfem::Mpi::Root()) {
      std::cout << "Time Step: " << cnt << ", Time: " << physics->time() << std::endl;
    }
    physics->advanceTimestep(time_increment);

    TimeInfo time_info(physics->time() - time_increment, time_increment);
    auto reactions = physics->getStateAdvancer()->computeResultants(shape_disp, physics->getFieldStates(),
                                                                    physics->getFieldStatesOld(), params, time_info);
    double reaction = CalculateReaction(*reactions[0].get(), mesh, "fix_top", 1);
    if (mfem::Mpi::Root()) {
      std::cout << "Reaction: " << reaction << std::endl;
    }
    pv_writer.write(cnt, physics->time(), physics->getFieldStatesAndParamStates());
  }

  TimeInfo time_info(physics->time() - time_increment, time_increment);

  auto final_states = physics->getFieldStates();
  auto previous_to_final_states = physics->getFieldStatesOld();

  auto state_advancer = physics->getStateAdvancer();
  printf("a\n");
  auto reactions =
      state_advancer->computeResultants(shape_disp, final_states, previous_to_final_states, params, time_info);

  printf("b\n");
  auto disp_squared = innerProduct(reactions[0], reactions[0]);

  gretl::set_as_objective(disp_squared);
  std::cout << "final disp norm2 = " << disp_squared.get() << std::endl;
}
