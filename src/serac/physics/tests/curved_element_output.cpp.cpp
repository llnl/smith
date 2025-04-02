#include "serac/physics/solid_mechanics.hpp"

#include "axom/slic/core/SimpleLogger.hpp"
#include <gtest/gtest.h>
#include "mfem.hpp"

#include "serac/infrastructure/terminator.hpp"
#include "serac/mesh/mesh_utils.hpp"
#include "serac/numerics/functional/domain.hpp"
#include "serac/physics/boundary_conditions/components.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/physics/materials/solid_material.hpp"
#include "serac/serac_config.hpp"


namespace serac {

TEST(SolidMechanics, CurvedElementOutput)
{
  constexpr int dim = 2;
  constexpr int p = 1;
  
  const std::string mesh_tag = "mesh";
  const std::string physics_prefix = "solid";
  
  int serial_refinement = 1;
  int parallel_refinement = 0;

  // Create DataStore
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, "curved_element_output_test");

  // Construct the appropriate dimension mesh and give it to the data store
  std::string filename = SERAC_REPO_DIR "/data/meshes/single_curved_quad.g";
  auto mesh = mesh::refineAndDistribute(buildMeshFromFile(filename), serial_refinement, parallel_refinement);
  auto& pmesh = serac::StateManager::setMesh(std::move(mesh), mesh_tag);

  Domain whole_domain = EntireDomain(pmesh);
  Domain left_boundary = Domain::ofBoundaryElements(pmesh, by_attr<dim>(4));
  Domain bottom_boundary = Domain::ofBoundaryElements(pmesh, by_attr<dim>(1));
  Domain right_boundary = Domain::ofBoundaryElements(pmesh, by_attr<dim>(2));

  NonlinearSolverOptions nonlinear_opts = solid_mechanics::default_nonlinear_options;
  LinearSolverOptions linear_opts = solid_mechanics::default_linear_options;
  TimesteppingOptions time_opts = solid_mechanics::default_quasistatic_options;

  auto solid = SolidMechanics<p, dim>(nonlinear_opts, linear_opts, time_opts, "curved_element", mesh_tag);

  using Material = solid_mechanics::NeoHookean;
  constexpr double E = 1.0;
  constexpr double nu = 0.0;
  Material material{.K = E/3.0/(1.0 - 2.0*nu), .G = 0.5*E/(1.0 + nu)};
  solid.setMaterial(serac::DependsOn<>{}, material, whole_domain);

  solid.setFixedBCs(left_boundary, Component::X);
  solid.setFixedBCs(bottom_boundary, Component::Y);
  
  solid.setTraction([](auto, auto, auto) { return tensor<double, dim>{0.01, 0.0}; }, right_boundary);

  solid.completeSetup();

  solid.outputStateToDisk("curved_element_paraview");
}
    
}

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  serac::initialize(argc, argv);
  int result = RUN_ALL_TESTS();
  serac::exitGracefully(result);
  return result;
}
