// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <fstream>

#include "axom/slic/core/SimpleLogger.hpp"
#include <gtest/gtest.h>
#include "mfem.hpp"

#include "serac/serac_config.hpp"
#include "serac/mesh/mesh_utils.hpp"
#include "serac/numerics/functional/domain.hpp"
#include "serac/physics/solid_mechanics.hpp"
#include "serac/physics/materials/solid_material.hpp"
#include "serac/physics/materials/parameterized_solid_material.hpp"
#include "serac/physics/state/state_manager.hpp"

namespace serac {

void periodic_test(mfem::Element::Type element_type)
{
  MPI_Barrier(MPI_COMM_WORLD);

  int serial_refinement   = 0;
  int parallel_refinement = 0;

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "solid_periodic_dir");

  // Construct the appropriate dimension mesh and give it to the data store
  int    nElem = 2;
  double lx = 3.0e-1, ly = 3.0e-1, lz = 0.25e-1;
  auto   initial_mesh = mfem::Mesh(mfem::Mesh::MakeCartesian3D(4 * nElem, 4 * nElem, nElem, element_type, lx, ly, lz));

  // Create translation vectors defining the periodicity
  mfem::Vector              x_translation({lx, 0.0, 0.0});
  std::vector<mfem::Vector> translations = {x_translation};
  double                    tol          = 1e-6;

  std::vector<int> periodicMap = initial_mesh.CreatePeriodicVertexMapping(translations, tol);

  // Create the periodic mesh using the vertex mapping defined by the translation vectors
  auto periodic_mesh = mfem::Mesh::MakePeriodic(initial_mesh, periodicMap);
  auto mesh          = mesh::refineAndDistribute(std::move(periodic_mesh), serial_refinement, parallel_refinement);

  std::string mesh_tag{"mesh"};

  auto& pmesh = serac::StateManager::setMesh(std::move(mesh), mesh_tag);

  constexpr int p   = 1;
  constexpr int dim = 3;

  // Construct and initialized the user-defined moduli to be used as a differentiable parameter in
  // the solid physics module.
  FiniteElementState user_defined_shear_modulus(pmesh, L2<1>{}, "parameterized_shear");

  double shear_modulus_value = 1.0;

  user_defined_shear_modulus = shear_modulus_value;

  FiniteElementState user_defined_bulk_modulus(pmesh, L2<1>{}, "parameterized_bulk");

  double bulk_modulus_value = 1.0;

  user_defined_bulk_modulus = bulk_modulus_value;

  // Construct a functional-based solid solver
  SolidMechanics<p, dim, Parameters<L2<p>, L2<p>>> solid_solver(
      solid_mechanics::default_nonlinear_options, solid_mechanics::default_linear_options,
      solid_mechanics::default_quasistatic_options, "solid_periodic", mesh_tag, {"bulk", "shear"});

  solid_solver.setParameter(0, user_defined_bulk_modulus);
  solid_solver.setParameter(1, user_defined_shear_modulus);

  Domain whole_mesh = EntireDomain(pmesh);

  solid_mechanics::ParameterizedNeoHookeanSolid mat{1.0, 0.0, 0.0};
  solid_solver.setMaterial(DependsOn<0, 1>{}, mat, whole_mesh);

  // Boundary conditions
  Domain support = Domain::ofBoundaryElements(pmesh, by_attr<dim>(2));
  solid_solver.setFixedBCs(support);

  double iniDispVal       = 5.0e-6;
  auto   ini_displacement = [iniDispVal](const mfem::Vector&, mfem::Vector& u) -> void { u = iniDispVal; };
  solid_solver.setDisplacement(ini_displacement);

  tensor<double, dim> constant_force{};
  constant_force[1] = 1.0e-2;

  solid_mechanics::ConstantBodyForce<dim> force{constant_force};
  solid_solver.addBodyForce(force, whole_mesh);

  // Finalize the data structures
  solid_solver.completeSetup();

  // Perform the quasi-static solve
  solid_solver.advanceTimestep(1.0);

  [[maybe_unused]] auto [K, K_e] = solid_solver.stiffnessMatrix();

  // Output the sidre-based plot files
  solid_solver.outputStateToDisk();
}

// note: these tests aren't checking correctness, just that periodic meshes
//       don't crash the physics modules / output routines
TEST(SolidMechanics, PeriodicTets) { periodic_test(mfem::Element::TETRAHEDRON); }
TEST(SolidMechanics, PeriodicHexes) { periodic_test(mfem::Element::HEXAHEDRON); }

}  // namespace serac

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);

  axom::slic::SimpleLogger logger;

  int result = RUN_ALL_TESTS();
  MPI_Finalize();

  return result;
}
