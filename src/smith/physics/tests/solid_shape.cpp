// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <memory>
#include <string>

#include "gtest/gtest.h"
#include "mpi.h"
#include "mfem.hpp"

#include "smith/physics/solid_mechanics.hpp"
#include "smith/physics/materials/solid_material.hpp"
#include "smith/smith_config.hpp"
#include "smith/physics/state/state_manager.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/mesh_utils/mesh_utils.hpp"
#include "smith/numerics/functional/finite_element.hpp"  // for H1
#include "smith/numerics/functional/tensor.hpp"
#include "smith/numerics/solver_config.hpp"
#include "smith/physics/state/finite_element_state.hpp"

namespace smith {

void shape_test()
{
  MPI_Barrier(MPI_COMM_WORLD);

  int serial_refinement = 0;
  int parallel_refinement = 0;

  constexpr int p = 1;
  constexpr int dim = 2;

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename = SMITH_REPO_DIR "/data/meshes/patch2D_tris_and_quads.mesh";

  // Create DataStore
  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, "solid_functional_shape_solve");

  std::string mesh_tag{"mesh"};
  auto mesh =
      std::make_shared<smith::Mesh>(buildMeshFromFile(filename), mesh_tag, serial_refinement, parallel_refinement);

  mfem::Vector shape_displacement;
  mfem::Vector pure_displacement;

  // Define the boundary where essential boundary conditions will be prescribed
  mesh->addDomainOfBoundaryElements("ess_bdr", by_attr<dim>(1));

  // Use a krylov solver for the Jacobian solve

  auto linear_options = solid_mechanics::default_linear_options;

  // Use tight tolerances as this is a machine precision test
#ifdef SMITH_USE_PETSC
  linear_options.linear_solver = LinearSolver::PetscCG;
  linear_options.preconditioner = Preconditioner::Petsc;
  linear_options.petsc_preconditioner = PetscPCType::HMG;
#else
  linear_options.preconditioner = Preconditioner::HypreJacobi;
#endif
  linear_options.relative_tol = 1.0e-15;
  linear_options.absolute_tol = 1.0e-15;

  auto nonlinear_options = solid_mechanics::default_nonlinear_options;

  nonlinear_options.absolute_tol = 8.0e-15;
  nonlinear_options.relative_tol = 8.0e-15;
  nonlinear_options.max_iterations = 10;

  solid_mechanics::LinearIsotropic mat{1.0, 1.0, 1.0};

  double shape_factor = 2.0;

  auto applied_displacement = [](tensor<double, dim> x, double) {
    tensor<double, dim> u{};
    u[1] = x[0] * 0.1;
    return u;
  };

  auto applied_displacement_pure = [shape_factor](tensor<double, dim> x, double) {
    tensor<double, dim> u{};
    u[1] = (x[0] * 0.1) / (shape_factor + 1.0);
    return u;
  };

  // Construct and apply a uniform body load
  tensor<double, dim> constant_force;

  constant_force[0] = 0.0e-3;
  constant_force[1] = 1.0e-4;

  if (dim == 3) {
    constant_force[2] = 0.0;
  }

  solid_mechanics::ConstantBodyForce<dim> force{constant_force};

  // Project a non-affine transformation with an affine transformation on the boundary
  mfem::VectorFunctionCoefficient shape_coef(2, [shape_factor](const mfem::Vector& x, mfem::Vector& shape) {
    shape[0] = x[0] * shape_factor;
    shape[1] = 0.0;
  });

  {
    // Construct and initialized the user-defined shape velocity to offset the computational mesh
    FiniteElementState user_defined_shape_displacement(mesh->mfemParMesh(), H1<SHAPE_ORDER, dim>{});

    user_defined_shape_displacement.project(shape_coef);

    // Construct a functional-based solid mechanics solver including references to the shape velocity field.
    SolidMechanics<p, dim> solid_solver(nonlinear_options, linear_options, solid_mechanics::default_quasistatic_options,
                                        "solid_functional", mesh);

    // Set the initial displacement and boundary condition
    solid_solver.setDisplacementBCs(applied_displacement, mesh->domain("ess_bdr"));

    // For consistency of the problem, this value should match the one in the BCs
    solid_solver.setDisplacement(
        [applied_displacement](tensor<double, dim> X) { return applied_displacement(X, 0.0); });

    solid_solver.setShapeDisplacement(user_defined_shape_displacement);

    solid_solver.setMaterial(mat, mesh->entireBody());
    solid_solver.addBodyForce(force, mesh->entireBody());

    // Finalize the data structures
    solid_solver.completeSetup();

    // Perform the quasi-static solve
    solid_solver.advanceTimestep(1.0);

    shape_displacement = solid_solver.displacement().gridFunction();
  }

  axom::sidre::DataStore new_datastore;
  StateManager::reset();
  smith::StateManager::initialize(new_datastore, "solid_functional_pure_solve");

  std::string new_mesh_tag{"new_mesh"};
  auto new_mesh =
      std::make_shared<smith::Mesh>(buildMeshFromFile(filename), new_mesh_tag, serial_refinement, parallel_refinement);

  new_mesh->addDomainOfBoundaryElements("ess_bdr", by_attr<dim>(1));

  {
    // Construct and initialized the user-defined shape velocity to offset the computational mesh
    FiniteElementState user_defined_shape_displacement(new_mesh->mfemParMesh(), H1<SHAPE_ORDER, dim>{});

    user_defined_shape_displacement.project(shape_coef);

    // Delete the pre-computed geometry factors as we are mutating the mesh
    new_mesh->mfemParMesh().DeleteGeometricFactors();
    auto* mesh_nodes = new_mesh->mfemParMesh().GetNodes();
    *mesh_nodes += user_defined_shape_displacement.gridFunction();

    // Construct a functional-based solid mechanics solver including references to the shape velocity field.
    SolidMechanics<p, dim> solid_solver_no_shape(
        nonlinear_options, linear_options, solid_mechanics::default_quasistatic_options, "solid_functional", new_mesh);

    mfem::VisItDataCollection visit_dc("pure_version",
                                       const_cast<mfem::ParMesh*>(&solid_solver_no_shape.mfemParMesh()));
    visit_dc.RegisterField("displacement", &solid_solver_no_shape.displacement().gridFunction());
    visit_dc.Save();

    // Set the initial displacement and boundary condition
    solid_solver_no_shape.setDisplacementBCs(applied_displacement_pure, new_mesh->domain("ess_bdr"));
    solid_solver_no_shape.setDisplacement(
        [applied_displacement_pure](tensor<double, dim> X) { return applied_displacement_pure(X, 0.0); });

    solid_solver_no_shape.setMaterial(mat, new_mesh->entireBody());
    solid_solver_no_shape.addBodyForce(force, new_mesh->entireBody());

    // Finalize the data structures
    solid_solver_no_shape.completeSetup();

    // Perform the quasi-static solve
    solid_solver_no_shape.advanceTimestep(1.0);

    pure_displacement = solid_solver_no_shape.displacement().gridFunction();
    visit_dc.SetCycle(1);
    visit_dc.Save();
  }

  double error = pure_displacement.DistanceTo(shape_displacement.GetData());
  double relative_error = error / pure_displacement.Norml2();
  EXPECT_LT(relative_error, 4.5e-12);
}

TEST(SolidMechanics, MoveShape) { shape_test(); }

}  // namespace smith

int main(int argc, char* argv[])
{
  testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
