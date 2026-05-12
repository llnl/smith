// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <memory>
#include <set>
#include <string>

#include "gtest/gtest.h"
#include "mpi.h"
#include "axom/sidre.hpp"
#include "mfem.hpp"

#include "smith/smith_config.hpp"
#include "smith/physics/heat_transfer.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/materials/thermal_material.hpp"
#include "smith/physics/materials/parameterized_thermal_material.hpp"
#include "smith/physics/state/state_manager.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/mesh_utils/mesh_utils.hpp"
#include "smith/numerics/functional/finite_element.hpp"
#include "smith/physics/common.hpp"
#include "smith/physics/state/finite_element_dual.hpp"
#include "smith/physics/state/finite_element_state.hpp"

namespace smith {

TEST(Thermal, ParameterizedMaterial)
{
  MPI_Barrier(MPI_COMM_WORLD);

  int serial_refinement = 1;
  int parallel_refinement = 0;

  // Create DataStore
  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, "thermal_functional_parameterized_sensitivities");

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename = SMITH_REPO_DIR "/data/meshes/star.mesh";
  std::string mesh_tag{"mesh"};
  auto mesh =
      std::make_shared<smith::Mesh>(buildMeshFromFile(filename), mesh_tag, serial_refinement, parallel_refinement);

  constexpr int p = 1;
  constexpr int dim = 2;

  // Define a boundary attribute set
  std::set<int> ess_bdr = {1};

  // Construct and initialize the user-defined conductivity to be used as a differentiable parameter in
  // the heat transfer physics module.
  FiniteElementState user_defined_conductivity(mesh->mfemParMesh(), H1<1>{}, "parameterized_conductivity");

  user_defined_conductivity = 1.0;

  // We must know the index of the parameter finite element state in our parameter pack to take sensitivities.
  // As we only have one parameter in this example, the index is zero.
  constexpr int conductivity_parameter_index = 0;

  // Construct a functional-based heat transfer solver
  //
  // Note that we now include an extra template parameter indicating the finite element space for the parameterized
  // field, in this case the thermal conductivity. We also pass an array of finite element states for each of the
  // requested parameterized fields.
  HeatTransfer<p, dim, Parameters<H1<1>>> thermal_solver(
      heat_transfer::default_nonlinear_options, heat_transfer::direct_linear_options,
      heat_transfer::default_static_options, "thermal_functional", mesh, {"conductivity"});

  thermal_solver.setParameter(0, user_defined_conductivity);

  // Construct a potentially user-defined parameterized material and send it to the thermal module
  heat_transfer::ParameterizedLinearIsotropicConductor mat;
  thermal_solver.setMaterial(DependsOn<0>{}, mat, mesh->entireBody());

  // Define the function for the initial temperature and boundary condition
  auto bdr_temp = [](const mfem::Vector& x, double) -> double {
    if (x[0] < 0.5 || x[1] < 0.5) {
      return 1.0;
    }
    return 0.0;
  };

  // Set the initial temperature and boundary condition
  thermal_solver.setTemperatureBCs(ess_bdr, bdr_temp);
  thermal_solver.setTemperature(bdr_temp);

  // Define a constant source term
  heat_transfer::ConstantSource source{-1.0};
  thermal_solver.setSource(source, mesh->entireBody());

  // Set the flux term to zero for testing code paths
  heat_transfer::ConstantFlux flux_bc{0.0};
  thermal_solver.setFluxBCs(flux_bc, mesh->entireBoundary());

  // Finalize the data structures
  thermal_solver.completeSetup();

  // Perform the quasi-static solve
  thermal_solver.advanceTimestep(1.0);

  // Output the sidre-based plot files
  thermal_solver.outputStateToDisk();

  // Construct a dummy adjoint load (this would come from a QOI downstream).
  // This adjoint load is equivalent to a discrete L1 norm on the temperature.
  FiniteElementDual adjoint_load(mesh->mfemParMesh(), H1<p>{}, "adjoint_load");

  adjoint_load = 1.0;

  thermal_solver.setAdjointLoad({{"temperature", adjoint_load}});

  // Solve the adjoint problem
  thermal_solver.reverseAdjointTimestep();

  // Compute the sensitivity (d QOI/ d state * d state/d parameter) given the current adjoint solution
  auto sensitivity = thermal_solver.computeTimestepSensitivity(conductivity_parameter_index);

  EXPECT_NEAR(1.7890782925134845, mfem::ParNormlp(sensitivity, 2, MPI_COMM_WORLD), 1.0e-6);
}

}  // namespace smith

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
