#include <gtest/gtest.h>

#include "gretl/src/data_store.hpp"

#include "smith/smith_config.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/numerics/equation_solver.hpp"
#include "smith/numerics/solver_config.hpp"
#include "smith/mesh_utils/mesh_utils.hpp"

#include "smith/physics/state/state_manager.hpp"
#include "smith/physics/functional_objective.hpp"
#include "smith/physics/boundary_conditions/boundary_condition_manager.hpp"
#include "smith/physics/materials/parameterized_solid_material.hpp"

#include "smith/differentiable_numerics/differentiable_utils.hpp"
#include "smith/differentiable_numerics/differentiable_physics.hpp"
#include "smith/differentiable_numerics/dirichlet_boundary_conditions.hpp"
#include "smith/differentiable_numerics/differentiable_solver.hpp"
#include "smith/differentiable_numerics/solid_mechanics_state_advancer.hpp"
#include "smith/differentiable_numerics/time_discretized_weak_form.hpp"
#include "smith/differentiable_numerics/tests/paraview_helper.hpp"

namespace smith {

smith::LinearSolverOptions solid_linear_options{.linear_solver = smith::LinearSolver::CG,
                                                //
                                                .preconditioner = smith::Preconditioner::HypreJacobi,
                                                .relative_tol = 1e-11,
                                                .absolute_tol = 1e-11,
                                                .max_iterations = 10000,
                                                .print_level = 0};

smith::NonlinearSolverOptions solid_nonlinear_opts{.nonlin_solver = NonlinearSolver::TrustRegion,
                                                   .relative_tol = 1.0e-9,
                                                   .absolute_tol = 1.0e-9,
                                                   .max_iterations = 500,
                                                   .print_level = 0};

static constexpr int dim = 3;
static constexpr int order = 1;

using ShapeDispSpace = H1<1, dim>;
using VectorSpace = H1<order, dim>;
using ScalarParameterSpace = L2<0>;

struct ThermoMechMeshFixture : public testing::Test {
  double length = 1.0;
  double width = 0.04;
  int num_elements_x = 21;
  int num_elements_y = 3;
  int num_elements_z = 3;
  double elem_size = length / num_elements_x;

  void SetUp()
  {
    smith::StateManager::initialize(datastore, "solid");
    auto mfem_shape = mfem::Element::QUADRILATERAL;
    mesh = std::make_shared<smith::Mesh>(
        mfem::Mesh::MakeCartesian3D(num_elements_x, num_elements_y, num_elements_z, mfem_shape, length, width, width),
        "mesh", 0, 0);
    mesh->addDomainOfBoundaryElements("left", smith::by_attr<dim>(3));
    mesh->addDomainOfBoundaryElements("right", smith::by_attr<dim>(5));
  }

  axom::sidre::DataStore datastore;
  std::shared_ptr<smith::Mesh> mesh;
};

TEST_F(ThermoMechMeshFixture, Test)
{
  SMITH_MARK_FUNCTION;

  std::shared_ptr<DifferentiableSolver> d_solid_nonlinear_solver =
      buildDifferentiableNonlinearSolve(solid_nonlinear_opts, solid_linear_options, *mesh);

  double E = 100.0;
  double nu = 0.25;
  auto K = E / (3.0 * (1.0 - 2 * nu));
  auto G = E / (2.0 * (1.0 + nu));

  using MaterialType = solid_mechanics::ParameterizedNeoHookeanSolid;
  MaterialType material{.density = 1.0, .K0 = K, .G0 = G};

  double time_increment = 0.5;

  smith::SecondOrderTimeIntegrationRule time_rule(1.0);

  auto graph = std::make_shared<gretl::DataStore>(100);
  std::string physics_name = "solid";
  auto [shape_disp, states, params, time, solid_mechanics_weak_form] =
      SolidMechanicsStateAdvancer::buildWeakFormAndStates<dim, ShapeDispSpace, VectorSpace, ScalarParameterSpace,
                                                          ScalarParameterSpace>(physics_name, mesh, graph, time_rule);

  params[0].get()->setFromFieldFunction([=](smith::tensor<double, dim>) {
    double scaling = 1.0;  //((x[0] < 3) && (x[0] > 2)) ? 0.99 : 0.001;
    return scaling * material.K0;
  });

  params[1].get()->setFromFieldFunction([=](smith::tensor<double, dim>) {
    double scaling = 1.0;  //((x[0] <= 3) && (x[0] >= 2)) ? 0.99 : 0.001;
    return scaling * material.G0;
  });

  auto vector_bcs = std::make_shared<DirichletBoundaryConditions>(
      mesh->mfemParMesh(), space(states[SolidMechanicsStateAdvancer::DISPLACEMENT]));
  vector_bcs->setVectorBCs<dim>(mesh->domain("left"), [](double t, smith::tensor<double, dim> X) {
    auto bc = 0.0 * X;
    bc[0] = 0.01 * t;
    bc[1] = -0.05 * t;
    return bc;
  });
  vector_bcs->setFixedVectorBCs<dim>(mesh->domain("right"));

  solid_mechanics_weak_form->addBodyIntegral(
      smith::DependsOn<0, 1>{}, mesh->entireBodyName(),
      [material](const auto& /*time_info*/, auto /*X*/, auto u, auto /*v*/, auto a, auto bulk, auto shear) {
        MaterialType::State state;
        auto pk_stress = material(state, get<DERIVATIVE>(u), bulk, shear);
        return smith::tuple{get<VALUE>(a) * material.density, pk_stress};
      });

  auto solid_mech_advancer = std::make_shared<SolidMechanicsStateAdvancer>(d_solid_nonlinear_solver, vector_bcs,
                                                                           solid_mechanics_weak_form, time_rule);

  DifferentiablePhysics physics(mesh, graph, shape_disp, states, params, solid_mech_advancer, physics_name);

  physics.resetStates();

  auto pv_writer = smith::createParaviewOutput(*mesh, physics.getAllFieldStates(), physics_name);
  pv_writer.write(0, physics.time(), physics.getAllFieldStates());
  for (size_t m = 0; m < 10; ++m) {
    physics.advanceTimestep(time_increment);
    pv_writer.write(m + 1, physics.time(), physics.getAllFieldStates());
  }

  auto objective = std::make_shared<smith::FunctionalObjective<dim, Parameters<VectorSpace> > >(
      "integrated_squared_temperature", mesh, spaces({states[SolidMechanicsStateAdvancer::DISPLACEMENT]}));
  objective->addBodyIntegral(smith::DependsOn<0>(), mesh->entireBodyName(), [](auto /*t*/, auto /*X*/, auto U) {
    auto u = get<VALUE>(U);
    return smith::inner(u, u);
  });

  auto final_states = physics.getAllFieldStates();
  DoubleState disp_squared =
      smith::evaluateObjective(objective, shape_disp, {final_states[SolidMechanicsStateAdvancer::DISPLACEMENT]});
  gretl::set_as_objective(disp_squared);

  std::cout << "final disp norm2 = " << disp_squared.get() << std::endl;

  EXPECT_GT(checkGradWrt(disp_squared, shape_disp, *graph, 1.1e-2, 4, true), 0.7);
  EXPECT_GT(checkGradWrt(disp_squared, params[0], *graph, 1.6e-1, 4, true), 0.7);
  EXPECT_GT(checkGradWrt(disp_squared, params[1], *graph, 1.6e-1, 4, true), 0.7);
}

}  // namespace smith

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
