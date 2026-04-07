// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

#include "mfem.hpp"

#include "smith/smith_config.hpp"
#include "smith/differentiable_numerics/coupled_system_solver.hpp"
#include "smith/differentiable_numerics/nonlinear_block_solver.hpp"
#include "smith/differentiable_numerics/paraview_writer.hpp"
#include "smith/differentiable_numerics/thermo_mechanics_system.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/numerics/solver_config.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/state/state_manager.hpp"

namespace example_tm {

static constexpr int dim = 3;
static constexpr int displacement_order = 1;
static constexpr int temperature_order = 1;

struct Options {
  int nx = 24;
  int ny = 4;
  int nz = 4;
  int steps = 100;
  double dt = 0.01;
  double length = 1.0;
  double width = 0.15;
  double height = 0.15;
  double pull_rate = 0.2;
  std::string output_name = "paraview_green_saint_venant_staggered_thermomechanics";
};

bool parseIntFlag(const std::string& arg, const std::string& name, int& value)
{
  const auto prefix = name + "=";
  if (arg.rfind(prefix, 0) != 0) {
    return false;
  }
  value = std::stoi(arg.substr(prefix.size()));
  return true;
}

bool parseDoubleFlag(const std::string& arg, const std::string& name, double& value)
{
  const auto prefix = name + "=";
  if (arg.rfind(prefix, 0) != 0) {
    return false;
  }
  value = std::stod(arg.substr(prefix.size()));
  return true;
}

bool parseStringFlag(const std::string& arg, const std::string& name, std::string& value)
{
  const auto prefix = name + "=";
  if (arg.rfind(prefix, 0) != 0) {
    return false;
  }
  value = arg.substr(prefix.size());
  return true;
}

Options parseOptions(int argc, char* argv[])
{
  Options options;

  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--help") {
      if (mfem::Mpi::Root()) {
        std::cout << "Usage: green_saint_venant_staggered_thermomechanics "
                  << "[--nx=<int>] [--ny=<int>] [--nz=<int>] [--steps=<int>] [--dt=<real>] "
                  << "[--length=<real>] [--width=<real>] [--height=<real>] [--pull-rate=<real>] "
                  << "[--output=<name>]\n";
      }
      std::exit(0);
    }

    if (parseIntFlag(arg, "--nx", options.nx) || parseIntFlag(arg, "--ny", options.ny) ||
        parseIntFlag(arg, "--nz", options.nz) || parseIntFlag(arg, "--steps", options.steps) ||
        parseDoubleFlag(arg, "--dt", options.dt) || parseDoubleFlag(arg, "--length", options.length) ||
        parseDoubleFlag(arg, "--width", options.width) || parseDoubleFlag(arg, "--height", options.height) ||
        parseDoubleFlag(arg, "--pull-rate", options.pull_rate) ||
        parseStringFlag(arg, "--output", options.output_name)) {
      continue;
    }

    if (mfem::Mpi::Root()) {
      std::cerr << "Unknown option: " << arg << "\n";
    }
    std::exit(1);
  }

  return options;
}

template <typename T>
auto greenStrain(const smith::tensor<T, dim, dim>& grad_u)
{
  return 0.5 * (grad_u + transpose(grad_u) + dot(transpose(grad_u), grad_u));
}

struct LocalGreenSaintVenantThermoelasticMaterial {
  using State = smith::Empty;

  template <typename T1, typename T2, typename T3, typename T4>
  auto operator()(double /*dt*/, State&, const smith::tensor<T1, dim, dim>& grad_u,
                  const smith::tensor<T2, dim, dim>& grad_v, T3 theta, const smith::tensor<T4, dim>& grad_theta) const
  {
    const double bulk_modulus = youngs_modulus / (3.0 * (1.0 - 2.0 * poissons_ratio));
    const double shear_modulus = 0.5 * youngs_modulus / (1.0 + poissons_ratio);
    static constexpr auto I = smith::Identity<dim>();

    const auto F = grad_u + I;
    const auto strain = greenStrain(grad_u);
    const auto volumetric_strain = tr(strain);
    const auto second_piola = 2.0 * shear_modulus * dev(strain) +
                              bulk_modulus * (volumetric_strain - dim * thermal_expansion * (theta - theta_ref)) * I;
    const auto piola = dot(F, second_piola);

    const auto strain_rate =
        0.5 * (grad_v + transpose(grad_v) + dot(transpose(grad_v), grad_u) + dot(transpose(grad_u), grad_v));
    const auto absolute_temperature = ambient_temperature + theta;
    const auto thermoelastic_source = -dim * bulk_modulus * thermal_expansion * absolute_temperature * tr(strain_rate);
    const auto heat_flux = -thermal_conductivity * grad_theta;

    return smith::tuple{piola, heat_capacity, thermoelastic_source, heat_flux};
  }

  double density = 1.0;
  double youngs_modulus = 100.0;
  double poissons_ratio = 0.25;
  double heat_capacity = 25.0;
  double thermal_expansion = 1.0e-4;
  double theta_ref = 0.0;
  double thermal_conductivity = 0.5;
  double ambient_temperature = 293.15;
};

smith::LinearSolverOptions makeMechanicsLinearOptions()
{
  return smith::LinearSolverOptions{.linear_solver = smith::LinearSolver::SuperLU};
}

smith::NonlinearSolverOptions makeMechanicsNonlinearOptions()
{
  return smith::NonlinearSolverOptions{.nonlin_solver = smith::NonlinearSolver::NewtonLineSearch,
                                       .relative_tol = 1.0e-8,
                                       .absolute_tol = 1.0e-8,
                                       .max_iterations = 20,
                                       .max_line_search_iterations = 6,
                                       .print_level = 1};
}

smith::LinearSolverOptions makeThermalLinearOptions()
{
  return smith::LinearSolverOptions{.linear_solver = smith::LinearSolver::SuperLU};
}

smith::NonlinearSolverOptions makeThermalNonlinearOptions()
{
  return smith::NonlinearSolverOptions{.nonlin_solver = smith::NonlinearSolver::NewtonLineSearch,
                                       .relative_tol = 1.0e-8,
                                       .absolute_tol = 1.0e-8,
                                       .max_iterations = 15,
                                       .max_line_search_iterations = 6,
                                       .print_level = 1};
}

template <typename SystemType>
void initializeStates(SystemType& system, double initial_temperature)
{
  auto zero_vector = [](smith::tensor<double, dim>) { return smith::tensor<double, dim>{}; };
  auto constant_temperature = [initial_temperature](smith::tensor<double, dim>) { return initial_temperature; };

  auto& disp_solve = const_cast<smith::FiniteElementState&>(
      *system.field_store->getField(system.prefix("displacement_solve_state")).get());
  auto& disp =
      const_cast<smith::FiniteElementState&>(*system.field_store->getField(system.prefix("displacement")).get());
  auto& velocity =
      const_cast<smith::FiniteElementState&>(*system.field_store->getField(system.prefix("velocity")).get());
  auto& acceleration =
      const_cast<smith::FiniteElementState&>(*system.field_store->getField(system.prefix("acceleration")).get());
  auto& temp_solve = const_cast<smith::FiniteElementState&>(
      *system.field_store->getField(system.prefix("temperature_solve_state")).get());
  auto& temperature =
      const_cast<smith::FiniteElementState&>(*system.field_store->getField(system.prefix("temperature")).get());

  disp_solve.setFromFieldFunction(zero_vector);
  disp = disp_solve;
  velocity.setFromFieldFunction(zero_vector);
  acceleration.setFromFieldFunction(zero_vector);
  temp_solve.setFromFieldFunction(constant_temperature);
  temperature = temp_solve;
}

}  // namespace example_tm

int main(int argc, char* argv[])
{
  smith::ApplicationManager application_manager(argc, argv);
  const auto options = example_tm::parseOptions(argc, argv);

  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, "green_saint_venant_staggered_thermomechanics");

  auto mesh = std::make_shared<smith::Mesh>(
      mfem::Mesh::MakeCartesian3D(options.nx, options.ny, options.nz, mfem::Element::HEXAHEDRON, options.length,
                                  options.width, options.height),
      "thermo_block", 0, 0);

  mesh->addDomainOfBoundaryElements("left", smith::by_attr<example_tm::dim>(3));
  mesh->addDomainOfBoundaryElements("right", smith::by_attr<example_tm::dim>(5));

  auto mechanics_solver = smith::buildNonlinearBlockSolver(example_tm::makeMechanicsNonlinearOptions(),
                                                           example_tm::makeMechanicsLinearOptions(), *mesh);
  auto thermal_solver = smith::buildNonlinearBlockSolver(example_tm::makeThermalNonlinearOptions(),
                                                         example_tm::makeThermalLinearOptions(), *mesh);

  auto staggered_solver = std::make_shared<smith::CoupledSystemSolver>(12);
  staggered_solver->addSubsystemSolver({0}, mechanics_solver, {.relative_tols = {1.0e-6}, .absolute_tols = {1.0e-7}});
  staggered_solver->addSubsystemSolver({1}, thermal_solver, {.relative_tols = {1.0e-6}, .absolute_tols = {1.0e-7}});

  auto system =
      smith::buildThermoMechanicsSystem<example_tm::dim, example_tm::displacement_order, example_tm::temperature_order>(
          mesh, staggered_solver, smith::QuasiStaticSecondOrderTimeIntegrationRule{},
          smith::QuasiStaticFirstOrderTimeIntegrationRule{}, "green_saint_venant_staggered");

  example_tm::LocalGreenSaintVenantThermoelasticMaterial material;
  system.setMaterial(material, mesh->entireBodyName());

  system.disp_bc->setFixedVectorBCs<example_tm::dim>(mesh->domain("left"));
  system.disp_bc->setVectorBCs<example_tm::dim>(mesh->domain("right"), {0},
                                                [pull_rate = options.pull_rate](double t, auto X) {
                                                  auto displacement = 0.0 * X;
                                                  displacement[0] = pull_rate * t;
                                                  return displacement;
                                                });
  system.temperature_bc->setScalarBCs<example_tm::dim>(mesh->domain("left"), [](double, auto) { return 0.0; });
  system.temperature_bc->setScalarBCs<example_tm::dim>(mesh->domain("right"), [](double, auto) { return 0.0; });

  example_tm::initializeStates(system, 0.0);

  auto physics = system.createDifferentiablePhysics("green_saint_venant_staggered_thermomechanics");
  physics->resetStates();

  auto pv_writer = smith::createParaviewWriter(*mesh, system.getStateFields(), options.output_name);
  pv_writer.write(0, physics->time(), system.getStateFields());

  if (mfem::Mpi::Root()) {
    std::cout << "Running staggered quasistatic thermo-mechanics example with " << options.steps
              << " load steps and dt=" << options.dt << "\n";
  }

  for (int step = 0; step < options.steps; ++step) {
    physics->advanceTimestep(options.dt);
    pv_writer.write(static_cast<size_t>(step + 1), physics->time(), system.getStateFields());

    double dispnorm = physics->state(system.prefix("displacement")).Normlinf();
    double tempnorm = physics->state(system.prefix("temperature")).Normlinf();
    if (mfem::Mpi::Root()) {
      const double prescribed_right_displacement = options.pull_rate * physics->time();
      std::cout << "step " << (step + 1) << "/" << options.steps << "  time=" << physics->time()
                << "  right_disp=" << prescribed_right_displacement
                << "  |u|_inf=" << dispnorm
                << "  |T|_inf=" << tempnorm << "\n";
    }
  }

  if (mfem::Mpi::Root()) {
    std::cout << "Wrote ParaView output to '" << options.output_name << "'\n";
  }

  return 0;
}
