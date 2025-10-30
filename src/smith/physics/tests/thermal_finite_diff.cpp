// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <algorithm>
#include <memory>
#include <set>
#include <string>

#include "gtest/gtest.h"
#include "mpi.h"
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
#include "smith/numerics/solver_config.hpp"
#include "smith/physics/common.hpp"
#include "smith/physics/state/finite_element_dual.hpp"
#include "smith/physics/state/finite_element_state.hpp"

namespace smith {

TEST(Thermal, FiniteDifference)
{
  MPI_Barrier(MPI_COMM_WORLD);

  int serial_refinement = 1;
  int parallel_refinement = 0;

  // Create DataStore
  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, "thermal_functional_parameterized_sensitivities");

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename = SMITH_REPO_DIR "/data/meshes/square.mesh";

  std::string mesh_tag{"mesh"};

  auto mesh =
      std::make_shared<smith::Mesh>(buildMeshFromFile(filename), mesh_tag, serial_refinement, parallel_refinement);

  constexpr int p = 1;
  constexpr int dim = 2;

  // Define a boundary attribute set
  std::set<int> ess_bdr = {1, 2};

  // We must know the index of the parameter finite element state in our parameter pack to take sensitivities.
  // As we only have one parameter in this example, the index is zero.
  constexpr int conductivity_parameter_index = 0;

  // Construct a functional-based heat transfer solver
  //
  // Note that we now include an extra template parameter indicating the finite element space for the parameterized
  // field, in this case the thermal conductivity. We also pass an array of finite element states for each of the
  // requested parameterized fields.
  HeatTransfer<p, dim, Parameters<H1<1>>> thermal_solver(
      heat_transfer::default_nonlinear_options, heat_transfer::default_linear_options,
      heat_transfer::default_static_options, "thermal_functional", mesh, {"conductivity"});

  FiniteElementState user_defined_conductivity(mesh->mfemParMesh(), H1<1>{}, "user_defined_conductivity");

  double conductivity_value = 1.2;
  user_defined_conductivity = conductivity_value;

  thermal_solver.setParameter(0, user_defined_conductivity);

  // Construct a potentially user-defined parameterized material and send it to the thermal module
  heat_transfer::ParameterizedLinearIsotropicConductor mat;
  thermal_solver.setMaterial(DependsOn<0>{}, mat, mesh->entireBody());

  // Define a constant source term
  heat_transfer::ConstantSource source{1.0};
  thermal_solver.setSource(source, mesh->entireBody());

  // Set the flux term to zero for testing code paths
  heat_transfer::ConstantFlux flux_bc{0.0};
  thermal_solver.setFluxBCs(flux_bc, mesh->entireBoundary());

  // Define the function for the initial temperature and boundary condition
  auto bdr_temp = [](const mfem::Vector& x, double) -> double { return (x[0] < 0.5 || x[1] < 0.5) ? 1.0 : 0.0; };

  // Set the initial temperature and boundary condition
  thermal_solver.setTemperatureBCs(ess_bdr, bdr_temp);
  thermal_solver.setTemperature(bdr_temp);

  // Finalize the data structures
  thermal_solver.completeSetup();

  // Perform the quasi-static solve
  thermal_solver.advanceTimestep(1.0);

  // Output the sidre-based plot files
  thermal_solver.outputStateToDisk();

  // Make up an adjoint load which can also be viewed as a
  // sensitivity of some qoi with respect to displacement
  mfem::ParLinearForm adjoint_load_form(
      const_cast<mfem::ParFiniteElementSpace*>(&thermal_solver.temperature().space()));
  adjoint_load_form = 1.0;

  // Construct a dummy adjoint load (this would come from a QOI downstream).
  // This adjoint load is equivalent to a discrete L1 norm on the temperature.
  smith::FiniteElementDual adjoint_load(thermal_solver.temperature().space(), "adjoint_load");
  std::unique_ptr<mfem::HypreParVector> assembled_vector(adjoint_load_form.ParallelAssemble());
  adjoint_load = *assembled_vector;

  thermal_solver.setAdjointLoad({{"temperature", adjoint_load}});

  // Solve the adjoint problem
  thermal_solver.reverseAdjointTimestep();

  // Compute the sensitivity (d QOI/ d state * d state/d parameter) given the current adjoint solution
  [[maybe_unused]] auto sensitivity = thermal_solver.computeTimestepSensitivity(conductivity_parameter_index);

  // Perform finite difference on each conduction value
  // to check if computed qoi sensitivity is consistent
  // with finite difference on the temperature
  double eps = 1.0e-4;
  for (int i = 0; i < user_defined_conductivity.gridFunction().Size(); ++i) {
    // Perturb the conductivity
    (user_defined_conductivity)(i) = conductivity_value + eps;

    thermal_solver.setParameter(0, user_defined_conductivity);
    thermal_solver.advanceTimestep(1.0);
    mfem::ParGridFunction temperature_plus = thermal_solver.temperature().gridFunction();

    (user_defined_conductivity)(i) = conductivity_value - eps;

    thermal_solver.setParameter(0, user_defined_conductivity);
    thermal_solver.advanceTimestep(1.0);
    mfem::ParGridFunction temperature_minus = thermal_solver.temperature().gridFunction();

    // Reset to the original conductivity value
    (user_defined_conductivity)(i) = conductivity_value;

    // Finite difference to compute sensitivity of temperature with respect to conductivity
    mfem::ParGridFunction dtemp_dconductivity(
        const_cast<mfem::ParFiniteElementSpace*>(&thermal_solver.temperature().space()));
    for (int i2 = 0; i2 < temperature_plus.Size(); ++i2) {
      dtemp_dconductivity(i2) = (temperature_plus(i2) - temperature_minus(i2)) / (2.0 * eps);
    }

    // Compute numerical value of sensitivity of qoi with respect to conductivity
    // by taking the inner product between adjoint load and displacement sensitivity
    double dqoi_dconductivity = adjoint_load_form(dtemp_dconductivity);

    // See if these are similar
    SLIC_INFO(axom::fmt::format("dqoi_dconductivity: {}", dqoi_dconductivity));
    SLIC_INFO(axom::fmt::format("sensitivity: {}", sensitivity(i)));
    EXPECT_NEAR((sensitivity(i) - dqoi_dconductivity) / dqoi_dconductivity, 0.0, 1.0e-3);
  }
}

TEST(HeatTransfer, FiniteDifferenceShape)
{
  MPI_Barrier(MPI_COMM_WORLD);

  int serial_refinement = 0;
  int parallel_refinement = 0;

  // Create DataStore
  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, "thermal_parameterized_shape_sensitivities");

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename = SMITH_REPO_DIR "/data/meshes/star.mesh";

  std::string mesh_tag{"mesh"};

  auto mesh =
      std::make_shared<smith::Mesh>(buildMeshFromFile(filename), mesh_tag, serial_refinement, parallel_refinement);

  constexpr int p = 1;
  constexpr int dim = 2;

  // Define a boundary attribute set
  std::set<int> ess_bdr = {1};

  double shape_displacement_value = 1.0;

  // The nonlinear solver must have tight tolerances to ensure at least one Newton step occurs
  smith::NonlinearSolverOptions nonlin_opts{
      .relative_tol = 1.0e-8, .absolute_tol = 1.0e-14, .max_iterations = 10, .print_level = 1};

  // Construct a functional-based thermal solver
  HeatTransfer<p, dim> thermal_solver(nonlin_opts, heat_transfer::direct_linear_options,
                                      heat_transfer::default_static_options, "thermal_functional_shape", mesh);

  heat_transfer::LinearIsotropicConductor mat(1.0, 1.0, 1.0);

  thermal_solver.setMaterial(mat, mesh->entireBody());

  heat_transfer::ConstantSource source{1.0};
  thermal_solver.setSource(source, mesh->entireBody());

  FiniteElementState shape_displacement(mesh->mfemParMesh(), H1<SHAPE_ORDER, dim>{});

  shape_displacement = shape_displacement_value;
  thermal_solver.setShapeDisplacement(shape_displacement);

  // Define the function for the initial displacement and boundary condition
  auto one = [](const mfem::Vector&, double) -> double { return 1.0; };

  // Set the initial displacement and boundary condition
  thermal_solver.setTemperatureBCs(ess_bdr, one);
  thermal_solver.setTemperature(one);

  // Finalize the data structures
  thermal_solver.completeSetup();

  // Perform the quasi-static solve
  thermal_solver.advanceTimestep(1.0);

  // Output the sidre-based plot files
  thermal_solver.outputStateToDisk();

  // Make up an adjoint load which can also be viewed as a
  // sensitivity of some qoi with respect to temperature
  mfem::ParLinearForm adjoint_load_form(
      const_cast<mfem::ParFiniteElementSpace*>(&thermal_solver.temperature().space()));
  adjoint_load_form = 1.0;

  // Construct a dummy adjoint load (this would come from a QOI downstream).
  // This adjoint load is equivalent to a discrete L1 norm on the temperature.
  smith::FiniteElementDual adjoint_load(thermal_solver.temperature().space(), "adjoint_load");
  std::unique_ptr<mfem::HypreParVector> assembled_vector(adjoint_load_form.ParallelAssemble());
  adjoint_load = *assembled_vector;

  thermal_solver.setAdjointLoad({{"temperature", adjoint_load}});

  // Solve the adjoint problem
  thermal_solver.reverseAdjointTimestep();

  // Compute the sensitivity (d QOI/ d state * d state/d parameter) given the current adjoint solution
  [[maybe_unused]] auto& sensitivity = thermal_solver.computeTimestepShapeSensitivity();

  // Perform finite difference on each shape displacement value
  // to check if computed qoi sensitivity is consistent
  // with finite difference on the temperature
  double eps = 1.0e-6;
  for (int i = 0; i < shape_displacement.Size(); ++i) {
    // Perturb the shape field
    shape_displacement(i) = shape_displacement_value + eps;
    thermal_solver.setShapeDisplacement(shape_displacement);

    thermal_solver.advanceTimestep(1.0);
    mfem::ParGridFunction temperature_plus = thermal_solver.temperature().gridFunction();

    shape_displacement(i) = shape_displacement_value - eps;
    thermal_solver.setShapeDisplacement(shape_displacement);

    thermal_solver.advanceTimestep(1.0);
    mfem::ParGridFunction temperature_minus = thermal_solver.temperature().gridFunction();

    // Reset to the original bulk modulus value
    shape_displacement(i) = shape_displacement_value;
    thermal_solver.setShapeDisplacement(shape_displacement);

    // Finite difference to compute sensitivity of displacement with respect to bulk modulus
    mfem::ParGridFunction dtemp_dshape(const_cast<mfem::ParFiniteElementSpace*>(&thermal_solver.temperature().space()));
    for (int i2 = 0; i2 < temperature_plus.Size(); ++i2) {
      dtemp_dshape(i2) = (temperature_plus(i2) - temperature_minus(i2)) / (2.0 * eps);
    }

    // Compute numerical value of sensitivity of qoi with respect to bulk modulus
    // by taking the inner product between adjoint load and displacement sensitivity
    double dqoi_dshape = adjoint_load_form(dtemp_dshape);

    // See if these are similar
    SLIC_INFO(axom::fmt::format("dqoi_dshape: {}", dqoi_dshape));
    SLIC_INFO(axom::fmt::format("sensitivity: {}", sensitivity(i)));
    EXPECT_NEAR((sensitivity(i) - dqoi_dshape) / std::max(dqoi_dshape, 1.0e-3), 0.0, 1.0e-4);
  }
}

}  // namespace smith

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
