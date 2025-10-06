// Copyright (c) Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <array>
#include <complex>
#include <memory>
#include <set>
#include <string>
#include <tuple>

#include "mpi.h"
#include "mfem.hpp"

#include "serac/physics/thermomechanics.hpp"
#include "serac/physics/thermomechanics_monolithic.hpp"
#include "serac/physics/materials/green_saint_venant_thermoelastic.hpp"
#include "serac/serac_config.hpp"
#include "serac/physics/mesh.hpp"
#include "serac/physics/state/state_manager.hpp"
#include "serac/infrastructure/application_manager.hpp"
#include "serac/mesh_utils/mesh_utils.hpp"
#include "serac/numerics/functional/domain.hpp"
#include "serac/numerics/functional/dual.hpp"
#include "serac/numerics/functional/finite_element.hpp"
#include "serac/numerics/functional/geometry.hpp"
#include "serac/numerics/functional/tensor.hpp"
#include "serac/numerics/functional/tuple.hpp"
#include "serac/physics/boundary_conditions/components.hpp"
#include "../lua_loader/lua_loader.hpp"

namespace serac {

// Heat driven solid deformation test
// Fixed bcs for displacement: u(0, y) = 0, u(x, 0) = 0
template <int p, int dim, typename TempBC, typename FluxBC, typename ThermalSource>
void ThermomechHeatedDeform(const std::set<int>& temp_ess_bcs, const TempBC& temp_bc_function,
                            const FluxBC& flux_bc_function, const ThermalSource& source_function)
{
  MPI_Barrier(MPI_COMM_WORLD);

  std::string simulation_tag;
  std::string mesh_location;
  bool lua_verbose = true;
  std::filesystem::path script_path = SERAC_REPO_DIR "/examples/lattices/thermomech.lua";

  // Loading Parameters/Functions From Lua Tables
  LuaLoader::LuaLoader lua_loader(script_path, lua_verbose);  // maintain this for the lifetime of the simulation

  std::string problem_type = lua_loader.ExtractLuaParameter<std::string>("problem_type", "nominal");

  simulation_tag = problem_type;
  if (problem_type == "nominal") {
    mesh_location = SERAC_REPO_DIR "/data/meshes/hole_array_nominal.g";
  } else if (problem_type == "optimized") {
    mesh_location = SERAC_REPO_DIR "/data/meshes/hole_array_optimized.g";
  } else if (problem_type == "synthetic") {
    mesh_location = SERAC_REPO_DIR "/data/meshes/circle_lattice.g";
  } else {
    MFEM_ABORT("Error wrong problem type specified");
  }

  std::function<serac::vec3(const serac::vec3&, const double)> default_func =
      [](const serac::vec3&, const double) -> serac::vec3 { return serac::vec3{0.0, 0.0, 0.0}; };

  std::function<serac::vec3(const serac::vec3&, const double)> applied_displacement_func =
      lua_loader.ExtractLuaSpaceTimeFunction("applied_displacement", default_func);

  double max_time = lua_loader.ExtractLuaParameter<double>("simulation_time", 1.0);

  int N_Steps = lua_loader.ExtractLuaParameter<int>("simulation_steps", 100);

  int serial_refinement = lua_loader.ExtractLuaParameter<int>("serial_refinement", 0);

  int parallel_refinement = lua_loader.ExtractLuaParameter<int>("parallel_refinement", 0);

  std::vector<double> material_params =
      lua_loader.ExtractLuaTable<double>("material_parameters", "values", {1.0, 1.0, 1.0});

  bool use_custom_location = false;

  std::string custom_output = lua_loader.ExtractLuaParameter<std::string>("output_location", "");

  if (custom_output != "") {
    use_custom_location = true;
  }

  axom::sidre::DataStore datastore;
  std::string output_directory;
  if (use_custom_location) {
    output_directory = custom_output + "/" + simulation_tag;
  } else {
    output_directory = simulation_tag;
  }

  serac::StateManager::initialize(datastore, "thermomechHeatedDeform");

  // std::string filename = SERAC_REPO_DIR "/data/meshes/square_attribute.mesh";

  const std::string mesh_tag = "mesh";
  auto mesh =
      std::make_shared<serac::Mesh>(buildMeshFromFile(mesh_location), mesh_tag, serial_refinement, parallel_refinement);

  auto linear_opts = thermomechanics::direct_linear_options;
  auto nonlinear_opts = thermomechanics::default_nonlinear_options;
  ThermomechanicsMonolithic<p, dim> thermomech_solver(nonlinear_opts, linear_opts, "thermomechHeatedDeform", mesh);

  double rho = 1.0;
  double E = 100.0;
  double nu = 0.25;
  double c = 1.0;
  double alpha = 1.0e-3;
  double theta_ref = 0.0;
  double k = 1.0;
  thermomechanics::GreenSaintVenantThermoelasticMaterial material{rho, E, nu, c, alpha, theta_ref, k};

  thermomech_solver.setMaterial(material, mesh->entireBody());

  auto zero = [](const mfem::Vector&, double) -> double { return 0.0; };
  thermomech_solver.setTemperatureBCs(temp_ess_bcs, temp_bc_function);
  thermomech_solver.setFluxBCs(flux_bc_function, mesh->entireBoundary());

  thermomech_solver.setSource(source_function, mesh->entireBody());
  thermomech_solver.setTemperature(zero);

  // std::set<int> disp_ess_bdr_y = {4};
  std::set<int> disp_ess_bdr_x = {2};
  mesh->addDomainOfBoundaryElements("ess_y_bdr", by_attr<dim>(disp_ess_bdr_x));
  mesh->addDomainOfBoundaryElements("ess_x_bdr", by_attr<dim>(disp_ess_bdr_x));
  mesh->addDomainOfBoundaryElements("ess_z_bdr", by_attr<dim>(disp_ess_bdr_x));

  thermomech_solver.setFixedBCs(mesh->domain("ess_y_bdr"), Component::Y);
  thermomech_solver.setFixedBCs(mesh->domain("ess_x_bdr"), Component::X);
  thermomech_solver.setFixedBCs(mesh->domain("ess_z_bdr"), Component::Z);

  auto zeroVector = [](const mfem::Vector&, mfem::Vector& u) { u = 0.0; };
  thermomech_solver.setDisplacement(zeroVector);

  std::string paraview_tag = output_directory + "_paraview";
  if (use_custom_location) {
    paraview_tag = custom_output + "/" + simulation_tag;
  } else {
    paraview_tag = simulation_tag;
  }
  double dt = max_time / (static_cast<double>(N_Steps - 1));
  SLIC_INFO_ROOT(axom::fmt::format("Paraview Tag: {}", paraview_tag));
  thermomech_solver.outputStateToDisk(paraview_tag);

  thermomech_solver.completeSetup();

  // thermomech_solver.advanceTimestep(1.0);
  // thermomech_solver.outputStateToDisk(paraview_tag);

  for (int i = 1; i < N_Steps; ++i) {
    SLIC_INFO_ROOT("------------------------------------------");
    SLIC_INFO_ROOT(axom::fmt::format("TIME STEP {}", i));
    SLIC_INFO_ROOT(axom::fmt::format("time = {} (out of {})", thermomech_solver.time() + dt, max_time));
    serac::logger::flush();
    thermomech_solver.advanceTimestep(dt);
    thermomech_solver.outputStateToDisk(paraview_tag);
  }
}

}  // namespace serac

int main(int argc, char* argv[])
{
  serac::ApplicationManager applicationManager(argc, argv);
  serac::ThermomechHeatedDeform<1, 3>(
      std::set<int>{2, 4}, [](auto /* X */, auto /* time */) -> double { return 0.0; },
      [](auto /* X */, auto /* n */, auto /* time */, auto /* T */) { return 0.0; },
      [](auto /* X */, auto /* time */, auto /* T */, auto /* dT_dx */) { return 1.0; });
}
