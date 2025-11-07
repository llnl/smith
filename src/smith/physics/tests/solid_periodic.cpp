// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "mpi.h"
#include "mfem.hpp"

#include "smith/smith_config.hpp"
#include "smith/numerics/functional/domain.hpp"
#include "smith/physics/solid_mechanics.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/materials/solid_material.hpp"
#include "smith/physics/materials/parameterized_solid_material.hpp"
#include "smith/physics/state/state_manager.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/numerics/functional/finite_element.hpp"  // for L2
#include "smith/numerics/functional/tensor.hpp"
#include "smith/physics/common.hpp"
#include "smith/physics/state/finite_element_state.hpp"

namespace smith {

void periodic_test(mfem::Element::Type element_type)
{
  MPI_Barrier(MPI_COMM_WORLD);

  int serial_refinement = 0;
  int parallel_refinement = 0;

  // Create DataStore
  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, "solid_periodic_dir");

  // Construct the appropriate dimension mesh and give it to the data store
  int nElem = 2;
  double lx = 3.0e-1, ly = 3.0e-1, lz = 0.25e-1;
  auto initial_mesh = mfem::Mesh(mfem::Mesh::MakeCartesian3D(4 * nElem, 4 * nElem, nElem, element_type, lx, ly, lz));

  // Create translation vectors defining the periodicity
  mfem::Vector x_translation({lx, 0.0, 0.0});
  std::vector<mfem::Vector> translations = {x_translation};
  double tol = 1e-6;

  std::vector<int> periodicMap = initial_mesh.CreatePeriodicVertexMapping(translations, tol);

  std::string mesh_tag{"mesh"};
  // Create the periodic mesh using the vertex mapping defined by the translation vectors
  auto mesh = std::make_shared<smith::Mesh>(mfem::Mesh::MakePeriodic(initial_mesh, periodicMap), mesh_tag,
                                            serial_refinement, parallel_refinement);

  constexpr int p = 1;
  constexpr int dim = 3;

  // Construct and initialized the user-defined moduli to be used as a differentiable parameter in
  // the solid physics module.
  FiniteElementState user_defined_shear_modulus(mesh->mfemParMesh(), L2<1>{}, "parameterized_shear");

  double shear_modulus_value = 1.0;

  user_defined_shear_modulus = shear_modulus_value;

  FiniteElementState user_defined_bulk_modulus(mesh->mfemParMesh(), L2<1>{}, "parameterized_bulk");

  double bulk_modulus_value = 1.0;

  user_defined_bulk_modulus = bulk_modulus_value;

  // Construct a functional-based solid solver
  SolidMechanics<p, dim, Parameters<L2<p>, L2<p>>> solid_solver(
      solid_mechanics::default_nonlinear_options, solid_mechanics::default_linear_options,
      solid_mechanics::default_quasistatic_options, "solid_periodic", mesh, {"bulk", "shear"});

  solid_solver.setParameter(0, user_defined_bulk_modulus);
  solid_solver.setParameter(1, user_defined_shear_modulus);

  solid_mechanics::ParameterizedNeoHookeanSolid mat{1.0, 0.0, 0.0};
  solid_solver.setMaterial(DependsOn<0, 1>{}, mat, mesh->entireBody());

  mesh->addDomainOfBoundaryElements("support", by_attr<dim>(2));
  solid_solver.setFixedBCs(mesh->domain("support"));

  constexpr double iniDispVal = 5.0e-6;
  auto initial_displacement = [](tensor<double, dim>) { return make_tensor<dim>([](int) { return iniDispVal; }); };
  solid_solver.setDisplacement(initial_displacement);

  tensor<double, dim> constant_force{};
  constant_force[1] = 1.0e-2;

  solid_mechanics::ConstantBodyForce<dim> force{constant_force};
  solid_solver.addBodyForce(force, mesh->entireBody());

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

}  // namespace smith

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
