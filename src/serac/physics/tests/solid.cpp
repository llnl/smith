// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/solid_mechanics.hpp"

#include <algorithm>
#include <functional>
#include <fstream>
#include <set>
#include <string>

#include "axom/slic/core/SimpleLogger.hpp"
#include <gtest/gtest.h>
#include "mfem.hpp"

#include "serac/mesh/mesh_utils.hpp"
#include "serac/numerics/functional/domain.hpp"
#include "serac/physics/boundary_conditions/components.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/physics/materials/solid_material.hpp"
#include "serac/physics/materials/parameterized_solid_material.hpp"
#include "serac/serac_config.hpp"
#include "serac/infrastructure/initialize.hpp"
#include "serac/infrastructure/terminator.hpp"

namespace serac {

void functional_solid_test_static_J2()
{
  MPI_Barrier(MPI_COMM_WORLD);

  constexpr int p                   = 2;
  constexpr int dim                 = 3;
  int           serial_refinement   = 0;
  int           parallel_refinement = 0;

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "solid_mechanics_J2_test");

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename = SERAC_REPO_DIR "/data/meshes/beam-hex.mesh";

  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(filename), serial_refinement, parallel_refinement);

  std::string mesh_tag{"mesh"};

  auto& pmesh = serac::StateManager::setMesh(std::move(mesh), mesh_tag);

  // _solver_params_start
  serac::LinearSolverOptions linear_options{.linear_solver = LinearSolver::SuperLU};

  serac::NonlinearSolverOptions nonlinear_options{.nonlin_solver  = NonlinearSolver::Newton,
                                                  .relative_tol   = 1.0e-12,
                                                  .absolute_tol   = 1.0e-12,
                                                  .max_iterations = 5000,
                                                  .print_level    = 1};

  SolidMechanics<p, dim> solid_solver(nonlinear_options, linear_options, solid_mechanics::default_quasistatic_options,
                                      "solid_mechanics", mesh_tag);
  // _solver_params_end

  using Hardening = solid_mechanics::LinearHardening;
  using Material  = solid_mechanics::J2SmallStrain<Hardening>;

  Hardening hardening{.sigma_y = 50.0, .Hi = 50.0};
  Material  mat{
       .E         = 10000,  // Young's modulus
       .nu        = 0.25,   // Poisson's ratio
       .hardening = hardening,
       .Hk        = 5.0,  // kinematic hardening constant
       .density   = 1.0   // mass density
  };

  Material::State initial_state{};

  auto qdata = solid_solver.createQuadratureDataBuffer(initial_state);

  Domain whole_domain = EntireDomain(pmesh);
  solid_solver.setMaterial(mat, whole_domain, qdata);

  // prescribe zero displacement at the supported end of the beam,
  auto support = Domain::ofBoundaryElements(pmesh, by_attr<dim>(1));
  solid_solver.setFixedBCs(support);

  // apply a displacement along z to the the tip of the beam
  auto translated_in_z = [](tensor<double, dim>, double t) {
    tensor<double, dim> u{};
    u[2] = t * (t - 1);
    return u;
  };
  auto tip = Domain::ofBoundaryElements(pmesh, by_attr<dim>(2));
  solid_solver.setDisplacementBCs(translated_in_z, tip, Component::Z);

  // Finalize the data structures
  solid_solver.completeSetup();

  solid_solver.outputStateToDisk("paraview");

  // Perform the quasi-static solve
  int    num_steps = 10;
  double tmax      = 1.0;
  for (int i = 0; i < num_steps; i++) {
    solid_solver.advanceTimestep(tmax / num_steps);
    solid_solver.outputStateToDisk("paraview");
  }

  // this a qualitative test that just verifies
  // that plasticity models can have permanent
  // deformation after unloading
  // EXPECT_LT(norm(solid_solver.reactions()), 1.0e-5);
}
template <typename lambda>
struct ParameterizedBodyForce {
  template <int dim, typename T1, typename T2>
  auto operator()(const tensor<T1, dim> x, double /*t*/, T2 density) const
  {
    return get<0>(density) * acceleration(x);
  }
  lambda acceleration;
};

template <typename T>
ParameterizedBodyForce(T) -> ParameterizedBodyForce<T>;

template <int p, int dim>
void functional_parameterized_solid_test(double expected_disp_norm)
{
  MPI_Barrier(MPI_COMM_WORLD);

  int serial_refinement   = 0;
  int parallel_refinement = 0;

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "solid_functional_parameterized_solve");

  static_assert(dim == 2 || dim == 3, "Dimension must be 2 or 3 for solid functional parameterized test");

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename =
      (dim == 2) ? SERAC_REPO_DIR "/data/meshes/beam-quad.mesh" : SERAC_REPO_DIR "/data/meshes/beam-hex.mesh";

  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(filename), serial_refinement, parallel_refinement);

  std::string mesh_tag{"mesh"};

  auto& pmesh = serac::StateManager::setMesh(std::move(mesh), mesh_tag);

  // Construct and initialized the user-defined moduli to be used as a differentiable parameter in
  // the solid mechanics physics module.
  FiniteElementState user_defined_shear_modulus(pmesh, H1<1>{}, "parameterized_shear");

  user_defined_shear_modulus = 1.0;

  FiniteElementState user_defined_bulk_modulus(pmesh, H1<1>{}, "parameterized_bulk");

  user_defined_bulk_modulus = 1.0;

  // _custom_solver_start
  auto nonlinear_solver = std::make_unique<mfem::NewtonSolver>(pmesh.GetComm());
  nonlinear_solver->SetPrintLevel(1);
  nonlinear_solver->SetMaxIter(30);
  nonlinear_solver->SetAbsTol(1.0e-12);
  nonlinear_solver->SetRelTol(1.0e-10);

  auto linear_solver = std::make_unique<mfem::HypreGMRES>(pmesh.GetComm());
  linear_solver->SetPrintLevel(1);
  linear_solver->SetMaxIter(500);
  linear_solver->SetTol(1.0e-6);

  auto preconditioner = std::make_unique<mfem::HypreBoomerAMG>();
  linear_solver->SetPreconditioner(*preconditioner);

  auto equation_solver = std::make_unique<EquationSolver>(std::move(nonlinear_solver), std::move(linear_solver),
                                                          std::move(preconditioner));

  SolidMechanics<p, dim, Parameters<H1<1>, H1<1>>> solid_solver(std::move(equation_solver),
                                                                solid_mechanics::default_quasistatic_options,
                                                                "parameterized_solid", mesh_tag, {"shear", "bulk"});
  // _custom_solver_end

  solid_solver.setParameter(0, user_defined_bulk_modulus);
  solid_solver.setParameter(1, user_defined_shear_modulus);

  Domain whole_domain   = EntireDomain(pmesh);
  Domain whole_boundary = EntireBoundary(pmesh);

  solid_mechanics::ParameterizedLinearIsotropicSolid mat{1.0, 0.0, 0.0};
  solid_solver.setMaterial(DependsOn<0, 1>{}, mat, whole_domain);

  // Specify initial / boundary conditions
  Domain essential_boundary = Domain::ofBoundaryElements(pmesh, by_attr<dim>(1));

  solid_solver.setFixedBCs(essential_boundary);
  solid_solver.setDisplacement([](const mfem::Vector&, mfem::Vector& bc_vec) -> void { bc_vec = 0.0; });

  tensor<double, dim> constant_force;

  constant_force[0] = 0.0;
  constant_force[1] = 5.0e-4;

  if (dim == 3) {
    constant_force[2] = 0.0;
  }

  solid_mechanics::ConstantBodyForce<dim> force{constant_force};
  solid_solver.addBodyForce(force, whole_domain);

  // add some nonexistent body forces / tractions to check that
  // these parameterized versions compile and run without error
  solid_solver.addBodyForce(
      DependsOn<0>{}, [](const auto& x, double /*t*/, auto /* bulk */) { return x * 0.0; }, whole_domain);
  solid_solver.addBodyForce(DependsOn<1>{}, ParameterizedBodyForce{[](const auto& x) { return 0.0 * x; }},
                            whole_domain);
  solid_solver.setTraction(
      DependsOn<1>{}, [](const auto& x, auto...) { return 0 * x; }, whole_boundary);

  // Finalize the data structures
  solid_solver.completeSetup();

  // Perform the quasi-static solve
  solid_solver.advanceTimestep(1.0);

  // the calculations peformed in these lines of code
  // are not used, but running them as part of this test
  // checks the index-translation part of the derivative
  // kernels is working
  solid_solver.computeTimestepSensitivity(0);
  solid_solver.computeTimestepSensitivity(1);

  // Output the sidre-based plot files
  solid_solver.outputStateToDisk();

  // Check the final displacement norm
  EXPECT_NEAR(expected_disp_norm, norm(solid_solver.displacement()), 1.0e-6);
}

TEST(SolidMechanics, 2DQuadParameterizedStatic) { functional_parameterized_solid_test<2, 2>(2.2378592112148716); }

TEST(SolidMechanics, 3DQuadStaticJ2) { functional_solid_test_static_J2(); }

}  // namespace serac

int main(int argc, char* argv[])
{
  testing::InitGoogleTest(&argc, argv);

  serac::initialize(argc, argv);

  int result = RUN_ALL_TESTS();

  serac::exitGracefully(result);
}
