// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "gtest/gtest.h"
#include "smith/smith_config.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/numerics/solver_config.hpp"
#include "smith/differentiable_numerics/solid_statics_with_L2_state_system.hpp"
#include "smith/differentiable_numerics/differentiable_test_utils.hpp"
#include "smith/differentiable_numerics/paraview_writer.hpp"
// #include "smith/physics/functional_objective.hpp"  // Temporarily commented to isolate build issue

namespace smith {

LinearSolverOptions solid_linear_options{.linear_solver = LinearSolver::Strumpack,
                                         .preconditioner = Preconditioner::HypreJacobi,
                                         .relative_tol = 1e-12,
                                         .absolute_tol = 1e-12,
                                         .max_iterations = 2000,
                                         .print_level = 0};

NonlinearSolverOptions solid_nonlinear_opts{.nonlin_solver = NonlinearSolver::NewtonLineSearch,
                                            .relative_tol = 1.0e-10,
                                            .absolute_tol = 1.0e-10,
                                            .max_iterations = 100,
                                            .max_line_search_iterations = 50,
                                            .print_level = 1,
                                            .force_monolithic = true};

static constexpr int dim = 3;
static constexpr int disp_order = 1;
static constexpr int state_order = 0;

// Temporarily use H1 to test if the issue is specific to L2 elements
using StateSpace = L2<state_order>; // L2<state_order>;

struct SolidStaticWithStateFixture : public testing::Test {
  void SetUp() override {
    StateManager::initialize(datastore, "solid_state");
    // Create a single element cube
    mesh = std::make_shared<smith::Mesh>(
        mfem::Mesh::MakeCartesian3D(4, 4, 4, mfem::Element::HEXAHEDRON, 1.0, 1.0, 1.0),
        "mesh", 0, 0);
    mesh->addDomainOfBoundaryElements("bottom", by_attr<dim>(1)); // z=0
    mesh->addDomainOfBoundaryElements("top", by_attr<dim>(6));    // z=1
  }

  axom::sidre::DataStore datastore;
  std::shared_ptr<smith::Mesh> mesh;
};

// Simple damage-like material: stiffness decreases with state variable alpha
struct DamageMaterial {
  using State = smith::QOI;

  double E = 100.0;
  double nu = 0.3;

  template <typename StateType, typename DerivType, typename AlphaType, typename... Params>
  SMITH_HOST_DEVICE auto operator()(StateType /*state*/, DerivType deriv_u, AlphaType alpha, Params... /*params*/) const {
    auto epsilon = sym(deriv_u);
    auto tr_eps = tr(epsilon);
    auto I = Identity<dim>();
    
    double lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    double mu = E / (2.0 * (1.0 + nu));

    // Stiffness degradation: (1 - alpha) * Stress_elastic
    // Ensure alpha doesn't exceed 0.99 for stability
    auto damage = alpha;
    if (damage > 1.0) damage = 1.0;
    if (damage < 0.0) damage = 0.0;
    
    auto factor = 1.0 - damage;
    auto sigma = factor * (lambda * tr_eps * I + 2.0 * mu * epsilon);
    
    return sigma;
  }
};

// Evolution law: alpha depends on strain norm
// R = alpha - strain_norm / (strain_norm + 0.1)
// Asymptotes to 1.0 as strain_norm -> infinity
// Equals 0.5 when strain_norm = 0.1
struct StrainNormEvolution {
  template <typename TimeInfo, typename AlphaType, typename AlphaOldType, typename DerivType, typename... Params>
  SMITH_HOST_DEVICE auto operator()(TimeInfo /*t_info*/, AlphaType alpha, AlphaOldType /*alpha_old*/, DerivType deriv_u, Params... /*params*/) const {
    using std::sqrt;
     auto epsilon = sym(deriv_u);
     auto eps_norm = sqrt(inner(epsilon, epsilon));
     
     // Residual:
     return alpha - eps_norm / (eps_norm + 0.1);
  }
};

TEST_F(SolidStaticWithStateFixture, CoupledSolve)
{
  auto solver = buildDifferentiableNonlinearBlockSolver(solid_nonlinear_opts, solid_linear_options, *mesh);
  
  auto system = buildSolidStaticsWithL2StateSystem<dim, disp_order, StateSpace>(
      mesh, solver, "coupled_system");

  // Material and Evolution
  system.setMaterial(DamageMaterial{}, mesh->entireBodyName());
  system.addStateEvolution(mesh->entireBodyName(), StrainNormEvolution{});

  // Boundary Conditions

  // Fix bottom face
  system.disp_bc->setFixedVectorBCs<dim>(mesh->domain("bottom"));
  
  // Pull top face
  double pull_rate = 0.05;
  system.disp_bc->setVectorBCs<dim>(mesh->domain("top"), [pull_rate](double t, tensor<double, dim> /*X*/) {
    tensor<double, dim> u{};
    u[2] = pull_rate * t;
    return u;
  });

  auto physics = system.createDifferentiablePhysics("physics");

  // Create ParaView writer
  auto writer = createParaviewWriter(*mesh, physics->getFieldStates(), "solid_state_output");
  writer.write(0, 0.0, physics->getFieldStates());
  // Advance multiple steps
  for (int step = 1; step <= 5; ++step) {
    physics->advanceTimestep(1.0);
    writer.write(step, step * 1.0, physics->getFieldStates());
    SLIC_INFO("Completed step " << step);
  }
}

} // namespace smith

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
