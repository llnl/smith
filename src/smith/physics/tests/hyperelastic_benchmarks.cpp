// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <cmath>
#include <format>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>

#include "mpi.h"
#include "mfem.hpp"

#include "smith/smith_config.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/infrastructure/logger.hpp"
#include "smith/numerics/functional/domain.hpp"
#include "smith/numerics/functional/functional.hpp"
#include "smith/numerics/functional/tensor.hpp"
#include "smith/numerics/solver_config.hpp"
#include "smith/physics/boundary_conditions/components.hpp"
#include "smith/physics/materials/solid_material.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/solid_mechanics.hpp"
#include "smith/physics/state/state_manager.hpp"

namespace smith {
namespace {

constexpr int p = 2;
constexpr int dim = 3;
constexpr int max_benchmark_elements = 200000;

std::string nonlinear_solver_name = "NewtonLineSearch";
std::string linear_solver_name = "GMRES";
std::string preconditioner_name = "HypreAMG";
std::string deflation_order_name = "affine";
std::string deflation_coarse_mode_name = "global";
int nonlinear_max_iterations = 30;
int linear_max_iterations = 800;
int print_level = 1;
int num_steps = 3;
bool write_output = false;
std::string selected_case = "all";
int trust_subspace_option = static_cast<int>(SubSpaceOptions::WHEN_INDEFINITE_OR_BOUNDARY);
int trust_num_leftmost = 2;
int trust_num_previous_steps = 4;
int trust_work_quadrature_points = 2;
int trust_num_lanczos = 0;
int trust_num_lanczos_iters = 0;
double cg_model_stagnation_tol = 0.0;
int cg_model_stagnation_window = 0;
bool cg_eisenstat_walker = false;
bool use_bsr_spmv = false;

template <typename T>
T parseEnum(const std::map<std::string, T>& values, const std::string& value, const std::string& option)
{
  auto found = values.find(value);
  if (found == values.end()) {
    throw std::runtime_error(std::format("Unknown {} value '{}'", option, value));
  }
  return found->second;
}

DeflationOrder selectedDeflationOrder()
{
  if (deflation_order_name == "affine") {
    return DeflationOrder::Affine;
  }
  if (deflation_order_name == "quadratic") {
    return DeflationOrder::Quadratic;
  }
  throw std::runtime_error("Unknown --deflation-order value '" + deflation_order_name + "'");
}

CoarseMode selectedCoarseMode()
{
  if (deflation_coarse_mode_name == "global") {
    return CoarseMode::Additive;
  }
  if (deflation_coarse_mode_name == "local") {
    return CoarseMode::AdditiveLocal;
  }
  if (deflation_coarse_mode_name == "schwarz") {
    return CoarseMode::AdditiveSchwarz;
  }
  throw std::runtime_error("Unknown --deflation-coarse-mode value '" + deflation_coarse_mode_name + "'");
}

LinearSolverOptions linearOptions()
{
  return {.linear_solver = parseEnum(linearSolverMap, linear_solver_name, "--linear-solver"),
          .preconditioner = parseEnum(preconditionerMap, preconditioner_name, "--preconditioner"),
          .deflation_order = selectedDeflationOrder(),
          .deflation_coarse_mode = selectedCoarseMode(),
          .relative_tol = 1.0e-7,
          .absolute_tol = 1.0e-14,
          .max_iterations = linear_max_iterations,
          .cg_model_stagnation_tol = cg_model_stagnation_tol,
          .cg_model_stagnation_window = cg_model_stagnation_window,
          .cg_eisenstat_walker = cg_eisenstat_walker,
          .print_level = 0,
          .use_bsr_spmv = use_bsr_spmv,
          .bsr_block_size = dim};
}

NonlinearSolverOptions nonlinearOptions()
{
  return {.nonlin_solver = parseEnum(nonlinearSolverMap, nonlinear_solver_name, "--nonlinear-solver"),
          .relative_tol = 1.0e-7,
          .absolute_tol = 1.0e-9,
          .max_iterations = nonlinear_max_iterations,
          .print_level = print_level,
          .subspace_option = static_cast<SubSpaceOptions>(trust_subspace_option),
          .num_leftmost = trust_num_leftmost,
          .num_previous_steps = trust_num_previous_steps,
          .trust_work_quadrature_points = trust_work_quadrature_points,
          .trust_num_lanczos = trust_num_lanczos,
          .trust_num_lanczos_iters = trust_num_lanczos_iters};
}

int globalElementCount(const Mesh& mesh)
{
  int global_elements = 0;
  const int local_elements = mesh.mfemParMesh().GetNE();
  MPI_Allreduce(&local_elements, &global_elements, 1, MPI_INT, MPI_SUM, mesh.getComm());
  return global_elements;
}

void checkElementCount(const std::string& name, const Mesh& mesh)
{
  const int elements = globalElementCount(mesh);
  SLIC_INFO_ROOT(std::format("{}: global elements = {}", name, elements));
  SLIC_ERROR_ROOT_IF(
      elements >= max_benchmark_elements,
      std::format("{} has {} elements, exceeding the benchmark cap of {}", name, elements, max_benchmark_elements));
}

void advance(SolidMechanics<p, dim>& solid, const std::string& output_name)
{
  if (write_output) {
    solid.outputStateToDisk(output_name);
  }

  for (int step = 0; step < num_steps; ++step) {
    MPI_Barrier(MPI_COMM_WORLD);
    const double start_time = MPI_Wtime();
    solid.advanceTimestep(1.0 / num_steps);
    MPI_Barrier(MPI_COMM_WORLD);
    SLIC_INFO_ROOT(
        std::format("{} step {}/{} wall = {:.3f} s", output_name, step + 1, num_steps, MPI_Wtime() - start_time));
    if (write_output) {
      solid.outputStateToDisk(output_name);
    }
  }
}

void checkDisplacement(const std::string& name, const SolidMechanics<p, dim>& solid)
{
  const double displacement_norm = mfem::ParNormlp(solid.displacement(), 2, MPI_COMM_WORLD);
  SLIC_INFO_ROOT(std::format("{}: final displacement l2 = {:.8e}", name, displacement_norm));
  SLIC_ERROR_ROOT_IF(!std::isfinite(displacement_norm), name + " produced a non-finite displacement norm");
  SLIC_ERROR_ROOT_IF(displacement_norm <= 0.0, name + " produced a zero displacement norm");
}

void parseCommandLine(int& argc, char** argv)
{
  int write_arg = 1;
  for (int read_arg = 1; read_arg < argc; ++read_arg) {
    const std::string arg = argv[read_arg];
    if (arg.rfind("--nonlinear-solver=", 0) == 0) {
      nonlinear_solver_name = arg.substr(std::string("--nonlinear-solver=").size());
    } else if (arg.rfind("--linear-solver=", 0) == 0) {
      linear_solver_name = arg.substr(std::string("--linear-solver=").size());
    } else if (arg.rfind("--preconditioner=", 0) == 0) {
      preconditioner_name = arg.substr(std::string("--preconditioner=").size());
    } else if (arg.rfind("--deflation-order=", 0) == 0) {
      deflation_order_name = arg.substr(std::string("--deflation-order=").size());
    } else if (arg.rfind("--deflation-coarse-mode=", 0) == 0) {
      deflation_coarse_mode_name = arg.substr(std::string("--deflation-coarse-mode=").size());
    } else if (arg.rfind("--nonlinear-max-iterations=", 0) == 0) {
      nonlinear_max_iterations = std::stoi(arg.substr(std::string("--nonlinear-max-iterations=").size()));
    } else if (arg.rfind("--linear-max-iterations=", 0) == 0) {
      linear_max_iterations = std::stoi(arg.substr(std::string("--linear-max-iterations=").size()));
    } else if (arg.rfind("--trust-subspace-option=", 0) == 0) {
      trust_subspace_option = std::stoi(arg.substr(std::string("--trust-subspace-option=").size()));
    } else if (arg.rfind("--trust-num-leftmost=", 0) == 0) {
      trust_num_leftmost = std::stoi(arg.substr(std::string("--trust-num-leftmost=").size()));
    } else if (arg.rfind("--trust-num-previous-steps=", 0) == 0) {
      trust_num_previous_steps = std::stoi(arg.substr(std::string("--trust-num-previous-steps=").size()));
    } else if (arg.rfind("--trust-work-quadrature=", 0) == 0) {
      trust_work_quadrature_points = std::stoi(arg.substr(std::string("--trust-work-quadrature=").size()));
    } else if (arg.rfind("--trust-num-lanczos=", 0) == 0) {
      trust_num_lanczos = std::stoi(arg.substr(std::string("--trust-num-lanczos=").size()));
    } else if (arg.rfind("--trust-num-lanczos-iters=", 0) == 0) {
      trust_num_lanczos_iters = std::stoi(arg.substr(std::string("--trust-num-lanczos-iters=").size()));
    } else if (arg.rfind("--cg-stagnation-tol=", 0) == 0) {
      cg_model_stagnation_tol = std::stod(arg.substr(std::string("--cg-stagnation-tol=").size()));
    } else if (arg.rfind("--cg-stagnation-window=", 0) == 0) {
      cg_model_stagnation_window = std::stoi(arg.substr(std::string("--cg-stagnation-window=").size()));
    } else if (arg == "--cg-eisenstat-walker") {
      cg_eisenstat_walker = true;
    } else if (arg == "--use-bsr-spmv") {
      use_bsr_spmv = true;
    } else if (arg.rfind("--print-level=", 0) == 0) {
      print_level = std::stoi(arg.substr(std::string("--print-level=").size()));
    } else if (arg.rfind("--steps=", 0) == 0) {
      num_steps = std::stoi(arg.substr(std::string("--steps=").size()));
    } else if (arg == "--write-output") {
      write_output = true;
    } else if (arg.rfind("--case=", 0) == 0) {
      selected_case = arg.substr(std::string("--case=").size());
    } else {
      argv[write_arg] = argv[read_arg];
      ++write_arg;
    }
  }
  argc = write_arg;
}

}  // namespace

void runNearIncompressibleBlockCompression()
{
  axom::sidre::DataStore datastore;
  StateManager::initialize(datastore, "hyperelastic_block_compression");

  constexpr int nx = 16;
  constexpr int ny = 5;
  constexpr int nz = 5;
  constexpr double length = 3.0;
  constexpr double width = 0.8;
  constexpr double height = 0.8;

  auto mesh = std::make_shared<Mesh>(
      mfem::Mesh(mfem::Mesh::MakeCartesian3D(nx, ny, nz, mfem::Element::HEXAHEDRON, length, width, height)),
      "hyperelastic_block_mesh", 0, 0);
  checkElementCount("near-incompressible block", *mesh);

  mesh->addDomainOfBoundaryElements("fixed_face", by_attr<dim>(5));
  mesh->addDomainOfBoundaryElements("driven_face", by_attr<dim>(3));

  SolidMechanics<p, dim> solid(nonlinearOptions(), linearOptions(), solid_mechanics::default_quasistatic_options,
                               "hyperelastic_block", mesh);
  solid_mechanics::NeoHookean material{.density = 1.0, .K = 1000.0, .G = 1.0};
  solid.setMaterial(material, mesh->entireBody());
  solid.setFixedBCs(mesh->domain("fixed_face"));
  solid.setDisplacementBCs(
      [](tensor<double, dim>, double time) {
        tensor<double, dim> displacement{};
        displacement[0] = -0.35 * time;
        return displacement;
      },
      mesh->domain("driven_face"), Component::X);

  solid.completeSetup();
  advance(solid, "hyperelastic_block_compression");
  checkDisplacement("near-incompressible block", solid);
}

void runSpherePenaltyContact()
{
  axom::sidre::DataStore datastore;
  StateManager::initialize(datastore, "hyperelastic_sphere_penalty_contact");

  constexpr int nx = 26;
  constexpr int ny = 26;
  constexpr int nz = 8;
  constexpr double length = 1.0;
  constexpr double width = 1.0;
  constexpr double height = 0.4;
  constexpr double sphere_radius = 0.22;
  constexpr double contact_penalty = 1.0e5;
  const vec3 sphere_center{{0.5 * length, 0.5 * width, -0.26}};

  auto mesh = std::make_shared<Mesh>(
      mfem::Mesh(mfem::Mesh::MakeCartesian3D(nx, ny, nz, mfem::Element::HEXAHEDRON, length, width, height)),
      "hyperelastic_contact_mesh", 0, 0);
  checkElementCount("sphere penalty contact", *mesh);

  mesh->addDomainOfBoundaryElements("driven_face", by_attr<dim>(6));
  mesh->addDomainOfBoundaryElements("contact_face", by_attr<dim>(1));

  SolidMechanics<p, dim> solid(nonlinearOptions(), linearOptions(), solid_mechanics::default_quasistatic_options,
                               "hyperelastic_contact", mesh);
  solid_mechanics::NeoHookean material{.density = 1.0, .K = 100.0, .G = 1.0};
  solid.setMaterial(material, mesh->entireBody());
  solid.setDisplacementBCs(
      [](tensor<double, dim>, double time) {
        tensor<double, dim> displacement{};
        displacement[2] = -0.06 * time;
        return displacement;
      },
      mesh->domain("driven_face"));

  solid.addCustomBoundaryIntegral(
      DependsOn<>{},
      [=](double, auto X, auto displacement, auto) {
        auto x = get<VALUE>(X) + get<VALUE>(displacement);
        auto offset = x - sphere_center;
        auto distance = norm(offset);
        auto phi = distance - sphere_radius;
        auto contact_residual = 0.0 * x;
        if (phi < 0.0) {
          contact_residual = contact_penalty * phi * offset / distance;
        }
        return contact_residual;
      },
      mesh->domain("contact_face"));

  solid.completeSetup();
  advance(solid, "hyperelastic_sphere_penalty_contact");
  checkDisplacement("sphere penalty contact", solid);

  Functional<double(H1<p, dim>)> contact_energy({&solid.displacement().space()});
  contact_energy.AddSurfaceIntegral(
      DependsOn<0>{},
      [=](double, auto X, auto displacement) {
        auto x = get<VALUE>(X) + get<VALUE>(displacement);
        auto phi = norm(x - sphere_center) - sphere_radius;
        auto energy = 0.0 * phi;
        if (phi < 0.0) {
          energy = 0.5 * contact_penalty * phi * phi;
        }
        return energy;
      },
      mesh->domain("contact_face"));

  Functional<double(H1<p, dim>)> active_area({&solid.displacement().space()});
  active_area.AddSurfaceIntegral(
      DependsOn<0>{},
      [=](double, auto X, auto displacement) {
        auto x = get<VALUE>(X) + get<VALUE>(displacement);
        auto phi = norm(x - sphere_center) - sphere_radius;
        auto active = 0.0 * phi;
        if (phi < 0.0) {
          active = 1.0 + 0.0 * phi;
        }
        return active;
      },
      mesh->domain("contact_face"));

  const double energy = contact_energy(solid.time(), solid.displacement());
  const double area = active_area(solid.time(), solid.displacement());
  SLIC_INFO_ROOT(std::format("sphere penalty contact: active area = {:.8e}, contact energy = {:.8e}", area, energy));
  SLIC_ERROR_ROOT_IF(area <= 0.0, "sphere penalty contact produced no active contact area");
  SLIC_ERROR_ROOT_IF(!std::isfinite(energy), "sphere penalty contact produced non-finite contact energy");
}

void runTwistedBeam()
{
  axom::sidre::DataStore datastore;
  StateManager::initialize(datastore, "hyperelastic_twisted_beam");

  constexpr int nx = 50;
  constexpr int ny = 5;
  constexpr int nz = 5;
  constexpr double length = 8.0;
  constexpr double width = 0.5;
  constexpr double height = 0.5;
  constexpr double twist_angle = 1.75 * M_PI;
  constexpr double axial_compression = 0.18;

  auto mesh = std::make_shared<Mesh>(
      mfem::Mesh(mfem::Mesh::MakeCartesian3D(nx, ny, nz, mfem::Element::HEXAHEDRON, length, width, height)),
      "hyperelastic_twisted_beam_mesh", 0, 0);
  checkElementCount("twisted beam", *mesh);

  mesh->addDomainOfBoundaryElements("fixed_face", by_attr<dim>(5));
  mesh->addDomainOfBoundaryElements("twist_face", by_attr<dim>(3));

  SolidMechanics<p, dim> solid(nonlinearOptions(), linearOptions(), solid_mechanics::default_quasistatic_options,
                               "hyperelastic_twisted_beam", mesh);
  solid_mechanics::NeoHookean material{.density = 1.0, .K = 50.0, .G = 1.0};
  solid.setMaterial(material, mesh->entireBody());
  solid.setFixedBCs(mesh->domain("fixed_face"));
  solid.setDisplacementBCs(
      [=](tensor<double, dim> X, double time) {
        const double theta = twist_angle * time;
        const double y0 = X[1] - 0.5 * width;
        const double z0 = X[2] - 0.5 * height;
        tensor<double, dim> displacement{};
        displacement[0] = -axial_compression * time;
        displacement[1] = std::cos(theta) * y0 - std::sin(theta) * z0 - y0;
        displacement[2] = std::sin(theta) * y0 + std::cos(theta) * z0 - z0;
        return displacement;
      },
      mesh->domain("twist_face"));

  solid.completeSetup();
  advance(solid, "hyperelastic_twisted_beam");
  checkDisplacement("twisted beam", solid);
}

}  // namespace smith

int main(int argc, char* argv[])
{
  smith::parseCommandLine(argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  if (smith::selected_case == "all" || smith::selected_case == "block") {
    smith::runNearIncompressibleBlockCompression();
  }
  if (smith::selected_case == "all" || smith::selected_case == "contact") {
    smith::runSpherePenaltyContact();
  }
  if (smith::selected_case == "all" || smith::selected_case == "twist") {
    smith::runTwistedBeam();
  }
  SLIC_ERROR_ROOT_IF(smith::selected_case != "all" && smith::selected_case != "block" &&
                         smith::selected_case != "contact" && smith::selected_case != "twist",
                     "Unknown --case value '" + smith::selected_case + "'; use all, block, contact, or twist");
  return 0;
}
