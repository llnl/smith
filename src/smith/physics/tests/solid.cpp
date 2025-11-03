// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <algorithm>
#include <string>
#include <memory>

#include "gtest/gtest.h"
#include "mpi.h"
#include "mfem.hpp"

#include "smith/physics/solid_mechanics.hpp"
#include "smith/numerics/functional/domain.hpp"
#include "smith/physics/boundary_conditions/components.hpp"
#include "smith/physics/state/state_manager.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/materials/solid_material.hpp"
#include "smith/physics/materials/parameterized_solid_material.hpp"
#include "smith/smith_config.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/mesh_utils/mesh_utils.hpp"
#include "smith/numerics/equation_solver.hpp"
#include "smith/numerics/functional/finite_element.hpp"  // for H1
#include "smith/numerics/functional/tensor.hpp"
#include "smith/numerics/solver_config.hpp"
#include "smith/physics/common.hpp"
#include "smith/physics/state/finite_element_state.hpp"

namespace smith {

void functional_solid_test_static_J2()
{
  MPI_Barrier(MPI_COMM_WORLD);

  constexpr int p = 2;
  constexpr int dim = 3;
  int serial_refinement = 0;
  int parallel_refinement = 0;

  // Create DataStore
  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, "solid_mechanics_J2_test");

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename = SMITH_REPO_DIR "/data/meshes/beam-hex.mesh";

  std::string mesh_tag{"mesh"};
  auto mesh =
      std::make_shared<smith::Mesh>(buildMeshFromFile(filename), mesh_tag, serial_refinement, parallel_refinement);

  // _solver_params_start
  smith::LinearSolverOptions linear_options{.linear_solver = LinearSolver::SuperLU};

  smith::NonlinearSolverOptions nonlinear_options{.nonlin_solver = NonlinearSolver::Newton,
                                                  .relative_tol = 1.0e-12,
                                                  .absolute_tol = 1.0e-12,
                                                  .max_iterations = 5000,
                                                  .print_level = 1};

  SolidMechanics<p, dim> solid_solver(nonlinear_options, linear_options, solid_mechanics::default_quasistatic_options,
                                      "solid_mechanics", mesh);
  // _solver_params_end

  using Hardening = solid_mechanics::LinearHardening;
  using Material = solid_mechanics::J2SmallStrain<Hardening>;

  Hardening hardening{.sigma_y = 50.0, .Hi = 50.0, .eta = 0.0};
  Material mat{
      .E = 10000,  // Young's modulus
      .nu = 0.25,  // Poisson's ratio
      .hardening = hardening,
      .Hk = 5.0,      // kinematic hardening constant
      .density = 1.0  // mass density
  };

  Material::State initial_state{};

  auto qdata = solid_solver.createQuadratureDataBuffer(initial_state);

  solid_solver.setRateDependentMaterial(mat, mesh->entireBody(), qdata);

  // prescribe zero displacement at the supported end of the beam,
  mesh->addDomainOfBoundaryElements("support", by_attr<dim>(1));
  solid_solver.setFixedBCs(mesh->domain("support"));

  // apply a displacement along z to the the tip of the beam
  auto translated_in_z = [](tensor<double, dim>, double t) {
    tensor<double, dim> u{};
    u[2] = t * (t - 1);
    return u;
  };
  mesh->addDomainOfBoundaryElements("tip", by_attr<dim>(2));
  solid_solver.setDisplacementBCs(translated_in_z, mesh->domain("tip"), Component::Z);

  // Finalize the data structures
  solid_solver.completeSetup();

  solid_solver.outputStateToDisk("paraview");

  // Perform the quasi-static solve
  int num_steps = 10;
  double tmax = 1.0;
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

  int serial_refinement = 0;
  int parallel_refinement = 0;

  // Create DataStore
  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, "solid_functional_parameterized_solve");

  static_assert(dim == 2 || dim == 3, "Dimension must be 2 or 3 for solid functional parameterized test");

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename =
      (dim == 2) ? SMITH_REPO_DIR "/data/meshes/beam-quad.mesh" : SMITH_REPO_DIR "/data/meshes/beam-hex.mesh";

  std::string mesh_tag{"mesh"};
  auto mesh =
      std::make_shared<smith::Mesh>(buildMeshFromFile(filename), mesh_tag, serial_refinement, parallel_refinement);

  // Construct and initialized the user-defined moduli to be used as a differentiable parameter in
  // the solid mechanics physics module.
  FiniteElementState user_defined_shear_modulus(mesh->mfemParMesh(), H1<1>{}, "parameterized_shear");

  user_defined_shear_modulus = 1.0;

  FiniteElementState user_defined_bulk_modulus(mesh->mfemParMesh(), H1<1>{}, "parameterized_bulk");

  user_defined_bulk_modulus = 1.0;

  // _custom_solver_start
  auto nonlinear_solver = std::make_unique<mfem::NewtonSolver>(mesh->getComm());
  nonlinear_solver->SetPrintLevel(1);
  nonlinear_solver->SetMaxIter(30);
  nonlinear_solver->SetAbsTol(1.0e-12);
  nonlinear_solver->SetRelTol(1.0e-10);

  auto linear_solver = std::make_unique<mfem::HypreGMRES>(mesh->getComm());
  linear_solver->SetPrintLevel(1);
  linear_solver->SetMaxIter(500);
  linear_solver->SetTol(1.0e-6);

  auto preconditioner = std::make_unique<mfem::HypreBoomerAMG>();
  linear_solver->SetPreconditioner(*preconditioner);

  auto equation_solver = std::make_unique<EquationSolver>(std::move(nonlinear_solver), std::move(linear_solver),
                                                          std::move(preconditioner));

  SolidMechanics<p, dim, Parameters<H1<1>, H1<1>>> solid_solver(std::move(equation_solver),
                                                                solid_mechanics::default_quasistatic_options,
                                                                "parameterized_solid", mesh, {"shear", "bulk"});
  // _custom_solver_end

  solid_solver.setParameter(0, user_defined_bulk_modulus);
  solid_solver.setParameter(1, user_defined_shear_modulus);

  solid_mechanics::ParameterizedLinearIsotropicSolid mat{1.0, 0.0, 0.0};
  solid_solver.setMaterial(DependsOn<0, 1>{}, mat, mesh->entireBody());

  // Specify initial / boundary conditions
  mesh->addDomainOfBoundaryElements("essential_boundary", by_attr<dim>(1));

  solid_solver.setFixedBCs(mesh->domain("essential_boundary"));

  tensor<double, dim> constant_force;

  constant_force[0] = 0.0;
  constant_force[1] = 5.0e-4;

  if (dim == 3) {
    constant_force[2] = 0.0;
  }

  solid_mechanics::ConstantBodyForce<dim> force{constant_force};
  solid_solver.addBodyForce(force, mesh->entireBody());

  // add some nonexistent body forces / tractions to check that
  // these parameterized versions compile and run without error
  solid_solver.addBodyForce(
      DependsOn<0>{}, [](const auto& x, double /*t*/, auto /* bulk */) { return x * 0.0; }, mesh->entireBody());
  solid_solver.addBodyForce(DependsOn<1>{}, ParameterizedBodyForce{[](const auto& x) { return 0.0 * x; }},
                            mesh->entireBody());
  solid_solver.setTraction(DependsOn<1>{}, [](const auto& x, auto...) { return 0 * x; }, mesh->entireBoundary());

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

TEST(SolidMechanics, TDofBoundaryCondition)
{
  /*
    Verifies that the solution obtained by specifying displacement BCs by tdof
    is identical to the solution obtained by specifying the BCs by domains.
  */
  MPI_Barrier(MPI_COMM_WORLD);

  constexpr int p = 1;
  constexpr int dim = 2;
  int serial_refinement = 2;
  int parallel_refinement = 0;

  // Create DataStore
  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, "tdof_bc_test");

  // Construct the mesh
  std::string filename = SMITH_REPO_DIR "/data/meshes/square.mesh";
  std::string mesh_tag{"mesh"};
  auto pmesh =
      std::make_shared<smith::Mesh>(buildMeshFromFile(filename), mesh_tag, serial_refinement, parallel_refinement);

  pmesh->addDomainOfBoundaryElements("essential_boundary_x", by_attr<dim>(1));
  pmesh->addDomainOfBoundaryElements("essential_boundary_y", by_attr<dim>(2));

  smith::LinearSolverOptions linear_options{.linear_solver = LinearSolver::SuperLU};

  smith::NonlinearSolverOptions nonlinear_options{.nonlin_solver = NonlinearSolver::Newton,
                                                  .relative_tol = 1.0e-12,
                                                  .absolute_tol = 1.0e-12,
                                                  .max_iterations = 1,
                                                  .print_level = 1};

  using Material = solid_mechanics::LinearIsotropic;
  Material mat{.density = 1.0, .K = 0.5, .G = 1.0};

  auto body_force_function = [](auto, auto) { return tensor<double, dim>{0.1, 0.0}; };
  auto displacement_bc_function = [](auto X, auto) { return tensor<double, dim>{X[0], 0.0}; };

  // GENERATE REFERENCE SOLUTION
  // Make a solid mechanics module, specify the disp BCs by domain (ie, the standard method)
  SolidMechanics<p, dim> solid(nonlinear_options, linear_options, solid_mechanics::default_quasistatic_options,
                               "solid_mechanics", pmesh);
  solid.setMaterial(mat, pmesh->entireBody());
  solid.setDisplacementBCs(displacement_bc_function, pmesh->domain("essential_boundary_x"), Component::X);
  solid.setDisplacementBCs(displacement_bc_function, pmesh->domain("essential_boundary_y"), Component::Y);
  solid.addBodyForce(body_force_function, pmesh->entireBody());
  solid.completeSetup();
  solid.advanceTimestep(1.0);

  // GENERATE THE SOLUTION TO TEST
  // find the tdofs in the domain-based BCs
  auto ldofs = pmesh->domain("essential_boundary_x").dof_list(&solid.displacement().space());
  solid.displacement().space().DofsToVDofs(0, ldofs);

  auto ldofs2 = pmesh->domain("essential_boundary_y").dof_list(&solid.displacement().space());
  solid.displacement().space().DofsToVDofs(1, ldofs2);

  ldofs.Append(ldofs2);

  mfem::Array<int> true_dofs;
  for (int j = 0; j < ldofs.Size(); ++j) {
    int tdof = solid.displacement().space().GetLocalTDofNumber(ldofs[j]);
    if (tdof >= 0) true_dofs.Append(tdof);
  }

  // make another solver and set bcs by tdof
  SolidMechanics<p, dim> solid_by_tdof(nonlinear_options, linear_options, solid_mechanics::default_quasistatic_options,
                                       "solid_mechanics_by_tdof", pmesh);

  solid_by_tdof.setMaterial(mat, pmesh->entireBody());
  solid_by_tdof.setDisplacementBCsByDofList(displacement_bc_function, true_dofs);
  solid_by_tdof.addBodyForce(body_force_function, pmesh->entireBody());
  solid_by_tdof.completeSetup();
  solid_by_tdof.advanceTimestep(1.0);

  // compare solutions
  for (int i = 0; i < solid.displacement().Size(); i++) {
    EXPECT_DOUBLE_EQ(solid.displacement()[i], solid_by_tdof.displacement()[i]);
  }
}

}  // namespace smith

int main(int argc, char* argv[])
{
  testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
