// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/heat_transfer.hpp"
#include "serac/physics/materials/thermal_material.hpp"
#include "serac/physics/materials/parameterized_thermal_material.hpp"

#include <fstream>

#include <gtest/gtest.h>
#include "mfem.hpp"

#include "serac/serac_config.hpp"
#include "serac/mesh/mesh_utils.hpp"
#include "serac/physics/state/state_manager.hpp"

namespace serac {

TEST(HeatTransfer, MoveShape)
{
  MPI_Barrier(MPI_COMM_WORLD);

  int serial_refinement = 2;
  int parallel_refinement = 0;

  constexpr int p = 1;
  constexpr int dim = 2;

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename = SERAC_REPO_DIR "/data/meshes/patch2D_tris_and_quads.mesh";

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "thermal_shape_solve");

  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(filename), serial_refinement, parallel_refinement);

  std::string mesh_tag{"mesh"};

  auto& pmesh = serac::StateManager::setMesh(std::move(mesh), mesh_tag);

  mfem::Vector shape_temperature;
  mfem::Vector pure_temperature;

  // Define a boundary attribute set
  std::set<int> ess_bdr = {1};

  auto nonlinear_options = heat_transfer::default_nonlinear_options;
  nonlinear_options.absolute_tol = 1.0e-14;
  nonlinear_options.relative_tol = 1.0e-14;

  // Use tight tolerances as this is a machine precision test
  //
  // Sam: we're setting a really small abs tolerance here to
  //      work around https://github.com/mfem/mfem/issues/3641
  // TODO: adopt solution to issue above once implemented
  auto linear_options = heat_transfer::default_linear_options;
  linear_options.absolute_tol = 1.0e-30;

  auto time_integration_options = TimesteppingOptions{TimestepMethod::QuasiStatic};

  // Define an anisotropic conductor material model
  tensor<double, 2, 2> cond{{{5.0, 0.4}, {0.4, 1.0}}};
  heat_transfer::LinearConductor<dim> mat(1.0, 1.0, cond);

  heat_transfer::ConstantSource source{1.0};

  double shape_factor_1 = 200.0;
  double shape_factor_2 = 0.0;
  // Project a non-affine transformation
  mfem::VectorFunctionCoefficient shape_coef(
      2, [shape_factor_1, shape_factor_2](const mfem::Vector& x, mfem::Vector& shape) {
        shape[0] = x[1] * shape_factor_1;
        shape[1] = x[0] * shape_factor_2;
      });

  // Define the function for the initial temperature and boundary condition
  auto zero = [](const mfem::Vector&, double) -> double { return 0.0; };

  {
    // Construct a functional-based thermal solver including references to the shape displacement field.
    HeatTransfer<p, dim> thermal_solver(nonlinear_options, linear_options, time_integration_options, "thermal_shape",
                                        mesh_tag);

    // Set the initial temperature and boundary condition
    thermal_solver.setTemperatureBCs(ess_bdr, zero);
    thermal_solver.setTemperature(zero);

    FiniteElementState shape_displacement(pmesh, H1<SHAPE_ORDER, dim>{});

    shape_displacement.project(shape_coef);
    thermal_solver.setShapeDisplacement(shape_displacement);

    Domain whole_domain = EntireDomain(pmesh);
    thermal_solver.setMaterial(mat, whole_domain);
    thermal_solver.setSource(source, whole_domain);

    // Finalize the data structures
    thermal_solver.completeSetup();

    // Perform the quasi-static solve
    thermal_solver.advanceTimestep(1.0);

    thermal_solver.outputStateToDisk();

    shape_temperature = thermal_solver.temperature().gridFunction();
  }

  axom::sidre::DataStore new_datastore;
  StateManager::reset();
  serac::StateManager::initialize(new_datastore, "thermal_pure_solve");

  auto new_mesh = mesh::refineAndDistribute(buildMeshFromFile(filename), serial_refinement, parallel_refinement);

  std::string pure_mesh_tag{"pure_mesh"};

  auto& new_pmesh = serac::StateManager::setMesh(std::move(new_mesh), pure_mesh_tag);

  {
    // Construct and initialized the user-defined shape displacement to offset the computational mesh
    FiniteElementState user_defined_shape_displacement(new_pmesh, H1<SHAPE_ORDER, dim>{});

    user_defined_shape_displacement.project(shape_coef);

    // Delete the pre-computed geometry factors as we are mutating the mesh
    new_pmesh.DeleteGeometricFactors();
    auto* mesh_nodes = new_pmesh.GetNodes();
    *mesh_nodes += user_defined_shape_displacement.gridFunction();

    // Construct a functional-based thermal solver including references to the shape displacement field.
    HeatTransfer<p, dim> thermal_solver_no_shape(nonlinear_options, linear_options, time_integration_options,
                                                 "thermal_pure", pure_mesh_tag);

    // Set the initial temperature and boundary condition
    thermal_solver_no_shape.setTemperatureBCs(ess_bdr, zero);
    thermal_solver_no_shape.setTemperature(zero);

    Domain whole_domain = EntireDomain(new_pmesh);
    thermal_solver_no_shape.setMaterial(mat, whole_domain);
    thermal_solver_no_shape.setSource(source, whole_domain);

    // Finalize the data structures
    thermal_solver_no_shape.completeSetup();

    // Perform the quasi-static solve
    thermal_solver_no_shape.advanceTimestep(1.0);

    thermal_solver_no_shape.outputStateToDisk();

    pure_temperature = thermal_solver_no_shape.temperature().gridFunction();
  }

  double error = pure_temperature.DistanceTo(shape_temperature.GetData());
  double relative_error = error / pure_temperature.Norml2();
  EXPECT_LT(relative_error, 5.0e-14);
}

}  // namespace serac

//------------------------------------------------------------------------------
#include "axom/slic/core/SimpleLogger.hpp"

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);

  axom::slic::SimpleLogger logger;

  int result = RUN_ALL_TESTS();
  MPI_Finalize();

  return result;
}
