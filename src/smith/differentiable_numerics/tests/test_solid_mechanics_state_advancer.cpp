#include <gtest/gtest.h>

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

#include "smith/differentiable_numerics/differentiable_solver.hpp"
#include "smith/differentiable_numerics/differentiable_solid_mechanics.hpp"
#include "smith/differentiable_numerics/dirichlet_boundary_conditions.hpp"

#include "smith/differentiable_numerics/tests/paraview_helper.hpp"
#include "smith/differentiable_numerics/differentiable_test_utils.hpp"

namespace smith {

smith::LinearSolverOptions solid_linear_options{.linear_solver = smith::LinearSolver::CG,
                                                //
                                                .preconditioner = smith::Preconditioner::HypreJacobi,
                                                .relative_tol = 1e-11,
                                                .absolute_tol = 1e-11,
                                                .max_iterations = 10000,
                                                .print_level = 0};

smith::NonlinearSolverOptions solid_nonlinear_opts{.nonlin_solver = NonlinearSolver::TrustRegion,
                                                   .relative_tol = 1.0e-10,
                                                   .absolute_tol = 1.0e-10,
                                                   .max_iterations = 500,
                                                   .print_level = 0};

static constexpr int dim = 3;
static constexpr int order = 1;

using ShapeDispSpace = H1<1, dim>;
using VectorSpace = H1<order, dim>;
using ScalarParameterSpace = L2<0>;

struct SolidMechanicsMeshFixture : public testing::Test {
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

void resetAndApplyInitialConditions(std::shared_ptr<BasePhysics> physics) { physics->resetStates(); }

double integrateForward(std::shared_ptr<BasePhysics> physics)
{
  resetAndApplyInitialConditions(physics);
  double time_increment = 0.5;
  for (size_t m = 0; m < 5; ++m) {
    physics->advanceTimestep(time_increment);
  }
  FiniteElementDual reaction = physics->dual("reactions");

  return 0.5 * innerProduct(reaction, reaction);
}

void adjointBackward(std::shared_ptr<BasePhysics> physics, smith::FiniteElementDual& shape_sensitivity,
                     std::vector<smith::FiniteElementDual>& parameter_sensitivities)
{
  smith::FiniteElementDual reaction = physics->dual("reactions");
  smith::FiniteElementState reaction_dual(reaction.space(), "reactions_dual");
  reaction_dual = reaction;

  physics->resetAdjointStates();

  physics->setDualAdjointBcs({{"reactions", reaction_dual}});

  while (physics->cycle() > 0) {
    physics->reverseAdjointTimestep();
    shape_sensitivity += physics->computeTimestepShapeSensitivity();
    for (size_t param_index = 0; param_index < parameter_sensitivities.size(); ++param_index) {
      parameter_sensitivities[param_index] += physics->computeTimestepSensitivity(param_index);
    }
  }
}

TEST_F(SolidMechanicsMeshFixture, SENSITIVITIES_GRETL)
{
  SMITH_MARK_FUNCTION;

  std::string physics_name = "solid";

  std::shared_ptr<DifferentiableSolver> d_solid_nonlinear_solver =
      buildDifferentiableNonlinearSolve(solid_nonlinear_opts, solid_linear_options, *mesh);

  smith::SecondOrderTimeIntegrationRule time_rule(1.0);

  // warm-start.
  // implicit Newmark.

  auto [physics, weak_form, bcs] =
      buildSolidMechanics<dim, ShapeDispSpace, VectorSpace, ScalarParameterSpace, ScalarParameterSpace>(
          mesh, d_solid_nonlinear_solver, time_rule, physics_name, {"bulk", "shear"});

  bcs->setFixedVectorBCs<dim>(mesh->domain("right"));
  bcs->setVectorBCs<dim>(mesh->domain("left"), [](double t, smith::tensor<double, dim> X) {
    auto bc = 0.0 * X;
    bc[0] = 0.01 * t;
    bc[1] = -0.05 * t;
    return bc;
  });

  double E = 100.0;
  double nu = 0.25;
  auto K = E / (3.0 * (1.0 - 2 * nu));
  auto G = E / (2.0 * (1.0 + nu));
  using MaterialType = solid_mechanics::ParameterizedNeoHookeanSolid;
  MaterialType material{.density = 10.0, .K0 = K, .G0 = G};

  weak_form->addBodyIntegral(
      smith::DependsOn<0, 1>{}, mesh->entireBodyName(),
      [material](const auto& /*time_info*/, auto /*X*/, auto u, auto /*v*/, auto a, auto bulk, auto shear) {
        MaterialType::State state;
        auto pk_stress = material(state, get<DERIVATIVE>(u), bulk, shear);
        return smith::tuple{get<VALUE>(a) * material.density, pk_stress};
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

  double time_increment = 0.5;
  auto pv_writer = smith::createParaviewOutput(*mesh, physics->getFieldStatesAndParamStates(), physics_name);
  pv_writer.write(0, physics->time(), physics->getFieldStatesAndParamStates());
  for (size_t m = 0; m < 5; ++m) {
    physics->advanceTimestep(time_increment);
    pv_writer.write(m + 1, physics->time(), physics->getFieldStatesAndParamStates());
  }

  TimeInfo time_info(physics->time() - time_increment, time_increment);

  auto state_advancer = physics->getStateAdvancer();
  auto reactions = state_advancer->computeResultants(shape_disp, physics->getFieldStates(),
                                                     physics->getFieldStatesOld(), params, time_info);
  auto reaction_squared = 0.5 * innerProduct(reactions[0], reactions[0]);

  gretl::set_as_objective(reaction_squared);
  std::cout << "final disp norm2 = " << reaction_squared.get() << std::endl;

  EXPECT_GT(checkGradWrt(reaction_squared, shape_disp, 1.1e-2, 4, true), 0.7);
  EXPECT_GT(checkGradWrt(reaction_squared, params[0], 3.2e-1, 4, true), 0.7);
  EXPECT_GT(checkGradWrt(reaction_squared, params[1], 3.2e-1, 4, true), 0.7);
}

TEST_F(SolidMechanicsMeshFixture, SENSITIVITIES_BASE_PHYSICS)
{
  SMITH_MARK_FUNCTION;

  std::string physics_name = "solid";

  std::shared_ptr<DifferentiableSolver> d_solid_nonlinear_solver =
      buildDifferentiableNonlinearSolve(solid_nonlinear_opts, solid_linear_options, *mesh);

  smith::SecondOrderTimeIntegrationRule time_rule(1.0);

  auto [physics, weak_form, bcs] =
      buildSolidMechanics<dim, ShapeDispSpace, VectorSpace, ScalarParameterSpace, ScalarParameterSpace>(
          mesh, d_solid_nonlinear_solver, time_rule, physics_name, {"bulk", "shear"});

  bcs->setFixedVectorBCs<dim>(mesh->domain("right"));
  bcs->setVectorBCs<dim>(mesh->domain("left"), [](double t, smith::tensor<double, dim> X) {
    auto bc = 0.0 * X;
    bc[0] = 0.01 * t;
    bc[1] = -0.05 * t;
    return bc;
  });

  double E = 100.0;
  double nu = 0.25;
  auto K = E / (3.0 * (1.0 - 2 * nu));
  auto G = E / (2.0 * (1.0 + nu));
  using MaterialType = solid_mechanics::ParameterizedNeoHookeanSolid;
  MaterialType material{.density = 10.0, .K0 = K, .G0 = G};

  weak_form->addBodyIntegral(
      smith::DependsOn<0, 1>{}, mesh->entireBodyName(),
      [material](const auto& /*time_info*/, auto /*X*/, auto u, auto /*v*/, auto a, auto bulk, auto shear) {
        MaterialType::State state;
        auto pk_stress = material(state, get<DERIVATIVE>(u), bulk, shear);
        return smith::tuple{get<VALUE>(a) * material.density, pk_stress};
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

  double qoi = integrateForward(physics);
  std::cout << "qoi = " << qoi << std::endl;

  size_t num_params = physics->parameterNames().size();

  smith::FiniteElementDual shape_sensitivity(*shape_disp.get_dual());
  std::vector<smith::FiniteElementDual> parameter_sensitivities;
  for (size_t p = 0; p < num_params; ++p) {
    parameter_sensitivities.emplace_back(*params[p].get_dual());
  }

  adjointBackward(physics, shape_sensitivity, parameter_sensitivities);

  auto state_sensitivities = physics->computeInitialConditionSensitivity();
  for (auto name_and_state_sensitivity : state_sensitivities) {
    std::cout << name_and_state_sensitivity.first << " " << name_and_state_sensitivity.second.Norml2() << std::endl;
  }

  std::cout << shape_sensitivity.name() << " " << shape_sensitivity.Norml2() << std::endl;

  for (size_t p = 0; p < num_params; ++p) {
    std::cout << parameter_sensitivities[p].name() << " " << parameter_sensitivities[p].Norml2() << std::endl;
  }
}

TEST_F(SolidMechanicsMeshFixture, TRANSIENT_CONSTANT_GRAVITY)
{
  SMITH_MARK_FUNCTION;

  const double dt = 1e-2;
  const size_t dt_reduction = 10;
  const size_t num_steps = 4;
  std::string physics_name = "solid";

  std::shared_ptr<DifferentiableSolver> d_solid_nonlinear_solver =
      buildDifferentiableNonlinearSolve(solid_nonlinear_opts, solid_linear_options, *mesh);

  smith::SecondOrderTimeIntegrationRule time_rule(1.0);

  auto [physics, weak_form, bcs] =
      buildSolidMechanics<dim, ShapeDispSpace, VectorSpace, ScalarParameterSpace, ScalarParameterSpace>(
          mesh, d_solid_nonlinear_solver, time_rule, physics_name, {"bulk", "shear"});

  double E = 100.0;
  double nu = 0.25;
  auto K = E / (3.0 * (1.0 - 2 * nu));
  auto G = E / (2.0 * (1.0 + nu));
  using MaterialType = solid_mechanics::ParameterizedNeoHookeanSolid;
  MaterialType material{.density = 10.0, .K0 = K, .G0 = G};

  weak_form->addBodyIntegral(
      smith::DependsOn<0, 1>{}, mesh->entireBodyName(),
      [material](const auto& /*time_info*/, auto /*X*/, auto u, auto /*v*/, auto a, auto bulk, auto shear) {
        MaterialType::State state;
        auto pk_stress = material(state, get<DERIVATIVE>(u), bulk, shear);
        return smith::tuple{get<VALUE>(a) * material.density, pk_stress};
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

  auto all_fields = physics->getFieldStatesAndParamStates();

  std::string pv_dir = std::string("paraview_") + physics->name();
  std::cout << "Writing output to " << pv_dir << std::endl;
  auto pv_writer = createParaviewOutput(*mesh, all_fields, pv_dir);
  pv_writer.write(physics->cycle(), physics->time(), all_fields);
  double time = 0.0;
  for (size_t m = 0; m < dt_reduction * num_steps; ++m) {
    double timestep = dt / double(dt_reduction);
    physics->advanceTimestep(timestep);
    all_fields = physics->getFieldStatesAndParamStates();
    pv_writer.write(physics->cycle(), physics->time(), all_fields);
    time += timestep;
  }

  // static constexpr double gravity = -9.0;
  // double a_exact = gravity;
  // double v_exact = gravity * time;
  // double u_exact = 0.5 * gravity * time * time;

  // FunctionalObjective<dim, Parameters<VectorSpace>> accel_error("accel_error", mesh,
  //                                                               spaces({all_fields[ACCEL]}));
  // accel_error.addBodyIntegral(DependsOn<0>{}, mesh->entireBodyName(), [a_exact](auto /*t*/, auto /*X*/, auto A) {
  //   auto a = get<VALUE>(A);
  //   auto da0 = a[0];
  //   auto da1 = a[1] - a_exact;
  //   return da0 * da0 + da1 * da1;
  // });
  // double a_err = accel_error.evaluate(TimeInfo(0.0, 1.0, 0), shape_disp->get().get(),
  //                                     getConstFieldPointers({all_fields[ACCEL]}));
  // EXPECT_NEAR(0.0, a_err, 1e-14);

  // FunctionalObjective<dim, Parameters<VectorSpace>> velo_error("velo_error", mesh,
  //                                                                            spaces({all_fields[VELO]}));
  // velo_error.addBodyIntegral(DependsOn<0>{}, mesh->entireBodyName(), [v_exact](auto /*t*/, auto /*X*/, auto V) {
  //   auto v = get<VALUE>(V);
  //   auto dv0 = v[0];
  //   auto dv1 = v[1] - v_exact;
  //   return dv0 * dv0 + dv1 * dv1;
  // });
  // double v_err = velo_error.evaluate(TimeInfo(0.0, 1.0, 0), shape_disp->get().get(),
  //                                    getConstFieldPointers({all_fields[VELO]}));
  // EXPECT_NEAR(0.0, v_err, 1e-14);

  // FunctionalObjective<dim, Parameters<VectorSpace>> disp_error("disp_error", mesh,
  //                                                                            spaces({all_fields[DISP]}));
  // disp_error.addBodyIntegral(DependsOn<0>{}, mesh->entireBodyName(), [u_exact](auto /*t*/, auto /*X*/, auto U) {
  //   auto u = get<VALUE>(U);
  //   auto du0 = u[0];
  //   auto du1 = u[1] - u_exact;
  //   return du0 * du0 + du1 * du1;
  // });
  // double u_err = disp_error.evaluate(TimeInfo(0.0, 1.0, 0), shape_disp->get().get(),
  //                                    getConstFieldPointers({all_fields[DISP]}));
  // EXPECT_NEAR(0.0, u_err, 1e-14);
}

}  // namespace smith

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
