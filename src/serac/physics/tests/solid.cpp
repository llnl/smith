// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "serac/physics/solid_mechanics.hpp"

#include <functional>
#include <fstream>
#include <set>
#include <string>

#include "axom/slic/core/SimpleLogger.hpp"
#include <gtest/gtest.h>
#include "mfem.hpp"

#include "serac/mesh/mesh_utils.hpp"
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

  serac::StateManager::setMesh(std::move(mesh), mesh_tag);

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

  solid_solver.setMaterial(mat, qdata);

  // prescribe zero displacement at the supported end of the beam,
  std::set<int> support           = {1};
  auto          zero_displacement = [](const mfem::Vector&, mfem::Vector& u) -> void { u = 0.0; };
  solid_solver.setDisplacementBCs(support, zero_displacement);

  // apply a displacement along z to the the tip of the beam
  auto translated_in_z = [](const mfem::Vector&, double t, mfem::Vector& u) -> void {
    u    = 0.0;
    u[2] = t * (t - 1);
  };
  std::set<int> tip = {2};
  solid_solver.setDisplacementBCs(tip, translated_in_z);

  solid_solver.setDisplacement(zero_displacement);

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

// The purpose of this test is to check that the spatial function-defined essential boundary conditions are
// working appropriately. It takes a 4 hex cube mesh and pins it in one corner. The z-direction displacement
// is set to zero on the bottom face and a constant negative value on the top face.
void functional_solid_spatial_essential_bc()
{
  MPI_Barrier(MPI_COMM_WORLD);

  constexpr int p                   = 1;
  constexpr int dim                 = 3;
  int           serial_refinement   = 1;
  int           parallel_refinement = 0;

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "solid_mechanics_spatial_essential");

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename = SERAC_REPO_DIR "/data/meshes/onehex.mesh";

  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(filename), serial_refinement, parallel_refinement);

  std::string mesh_tag{"mesh"};

  serac::StateManager::setMesh(std::move(mesh), mesh_tag);

  // Construct a functional-based solid mechanics solver
  SolidMechanics<p, dim> solid_solver(solid_mechanics::default_nonlinear_options,
                                      solid_mechanics::direct_linear_options,
                                      solid_mechanics::default_quasistatic_options, "solid_mechanics", mesh_tag);

  solid_mechanics::LinearIsotropic mat{1.0, 1.0, 1.0};
  solid_solver.setMaterial(mat);

  // Set up
  auto zero_vector = [](const mfem::Vector&, mfem::Vector& u) { u = 0.0; };

  // We want to test both the scalar displacement functions including the time argument and the scalar displacement
  // functions without
  auto zero_scalar   = [](const mfem::Vector&, double) { return 0.0; };
  auto scalar_offset = [](const mfem::Vector&) { return -0.1; };

  auto is_on_bottom = [](const mfem::Vector& x) {
    if (x(2) < 0.01) {
      return true;
    }
    return false;
  };

  auto is_on_bottom_corner = [](const mfem::Vector& x) {
    if (x(0) < 0.01 && x(1) < 0.01 && x(2) < 0.01) {
      return true;
    }
    return false;
  };

  auto is_on_top = [](const mfem::Vector& x) {
    if (x(2) > 0.95) {
      return true;
    }
    return false;
  };

  solid_solver.setDisplacementBCs(is_on_bottom_corner, zero_vector);
  solid_solver.setDisplacementBCs(is_on_bottom, zero_scalar, 2);
  solid_solver.setDisplacementBCs(is_on_top, scalar_offset, 2);

  // Set a zero initial guess
  solid_solver.setDisplacement(zero_vector);

  // Finalize the data structures
  solid_solver.completeSetup();

  // Perform the quasi-static solve
  solid_solver.advanceTimestep(1.0);
  solid_solver.outputStateToDisk();

  auto [size, rank] = serac::getMPIInfo();

  // This exact solution is only correct when two MPI ranks are used
  // It is based on a poisson ratio of 0.125 with a prescribed z strain of 10%
  if (size == 2) {
    auto vdim  = solid_solver.displacement().space().GetVDim();
    auto ndofs = solid_solver.displacement().space().GetTrueVSize() / vdim;
    auto dof   = [ndofs, vdim](auto node, auto component) {
      return mfem::Ordering::Map<serac::ordering>(ndofs, vdim, node, component);
    };

    // This is a vector of pairs containing the exact solution index and value for the known analytical dofs.
    // These exact indices and values are chosen to avoid dependence on solver tolerances.
    if (rank == 0) {
      std::vector<std::pair<int, double>> solution = {
          {dof(0, 0), 0.0},      {dof(1, 0), 0.0125},   {dof(4, 0), 0.00625}, {dof(8, 0), 0.0},     {dof(9, 0), 0.0125},
          {dof(12, 0), 0.00625}, {dof(0, 1), 0.0},      {dof(2, 1), 0.0125},  {dof(7, 1), 0.00625}, {dof(8, 1), 0.0},
          {dof(11, 1), 0.0125},  {dof(15, 1), 0.00625}, {dof(0, 2), -0.1},    {dof(1, 2), -0.1},    {dof(2, 2), -0.1},
          {dof(3, 2), -0.1},     {dof(4, 2), -0.1},     {dof(5, 2), -0.1},    {dof(6, 2), -0.1},    {dof(7, 2), -0.1},
          {dof(8, 2), -0.05},    {dof(9, 2), -0.05},    {dof(10, 2), -0.05},  {dof(11, 2), -0.05},  {dof(12, 2), -0.05},
          {dof(13, 2), -0.05},   {dof(14, 2), -0.05},   {dof(15, 2), -0.05},  {dof(16, 2), -0.1},   {dof(17, 2), -0.05},
      };
      for (auto exact_entry : solution) {
        EXPECT_NEAR(exact_entry.second, solid_solver.displacement()(exact_entry.first), 1.0e-8);
      }
    }

    if (rank == 1) {
      std::vector<std::pair<int, double>> solution = {
          {dof(0, 0), 0.0},     {dof(1, 0), 0.0125}, {dof(4, 0), 0.00625}, {dof(0, 1), 0.0}, {dof(2, 1), 0.0125},
          {dof(7, 1), 0.00625}, {dof(0, 2), 0.0},    {dof(1, 2), 0.0},     {dof(2, 2), 0.0}, {dof(3, 2), 0.0},
          {dof(4, 2), 0.0},     {dof(5, 2), 0.0},    {dof(6, 2), 0.0},     {dof(7, 2), 0.0}, {dof(8, 2), 0.0},
      };
      for (auto exact_entry : solution) {
        EXPECT_NEAR(exact_entry.second, solid_solver.displacement()(exact_entry.first), 1.0e-8);
      }
    }
  }
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

  solid_mechanics::ParameterizedLinearIsotropicSolid mat{1.0, 0.0, 0.0};
  solid_solver.setMaterial(DependsOn<0, 1>{}, mat);

  // Define the function for the initial displacement and boundary condition
  auto bc = [](const mfem::Vector&, mfem::Vector& bc_vec) -> void { bc_vec = 0.0; };

  // Define a boundary attribute set and specify initial / boundary conditions
  std::set<int> ess_bdr = {1};

  // Generate a true dof set from the boundary attribute
  mfem::Array<int> bdr_attr_marker(pmesh.bdr_attributes.Max());
  bdr_attr_marker    = 0;
  bdr_attr_marker[0] = 1;
  mfem::Array<int> true_dofs;
  auto             fe_space = const_cast<mfem::ParFiniteElementSpace*>(&solid_solver.displacement().space());
  fe_space->GetEssentialTrueDofs(bdr_attr_marker, true_dofs);

  solid_solver.setDisplacementBCsByDofList(true_dofs, bc);
  solid_solver.setDisplacement(bc);

  tensor<double, dim> constant_force;

  constant_force[0] = 0.0;
  constant_force[1] = 5.0e-4;

  if (dim == 3) {
    constant_force[2] = 0.0;
  }

  solid_mechanics::ConstantBodyForce<dim> force{constant_force};
  solid_solver.addBodyForce(force, EntireDomain<dim>(pmesh));

  // add some nonexistent body forces / tractions to check that
  // these parameterized versions compile and run without error
  solid_solver.addBodyForce(
      DependsOn<0>{}, [](const auto& x, double /*t*/, auto /* bulk */) { return x * 0.0; }, EntireDomain<dim>(pmesh));
  solid_solver.addBodyForce(DependsOn<1>{}, ParameterizedBodyForce{[](const auto& x) { return 0.0 * x; }},
                            EntireDomain<dim>(pmesh));
  solid_solver.setTraction(
      DependsOn<1>{}, [](const auto& x, auto...) { return 0 * x; }, EntireBoundary<dim>(pmesh));

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

TEST(SolidMechanics, SpatialBoundaryCondition) { functional_solid_spatial_essential_bc(); }

}  // namespace serac

int main(int argc, char* argv[])
{
  testing::InitGoogleTest(&argc, argv);

  serac::initialize(argc, argv);

  int result = RUN_ALL_TESTS();

  serac::exitGracefully(result);
}
