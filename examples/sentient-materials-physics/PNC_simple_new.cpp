// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/*
Created 4/8-4/9 2026
This is an updated version of my older "PNC_double_network_simple.cpp" driver that I made for Thomas
It is a 50x50x1 block, faces fixed in +/-z and undergoing fully reversed sinusoidal loading from +y
The material model (for now) is included in this file. It is a simple version of PNC thermal stiffening with no ISVs
(so no rate effects or path-dependence). The moduli in the functions Gm0 and Ge0 are hard-coded from the Senses et al 2015 data,
and the units are Pa*modscale. Modscale is a material parameter that converts the modulus units to modscale*Pa, so set this
to 1. for Pa, 1000 for kPa, 1e6 for MPa, etc
*/

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
#include "helpers/cl_parser.hpp"
#include "smith/differentiable_numerics/reaction.hpp"
#include "helpers/calculate_reactions.hpp"

namespace example_tm {

static constexpr int dim = 3;
static constexpr int displacement_order = 1;
static constexpr int temperature_order = 1;

struct Options {
  int scale = 1;
  int nx = 4*scale;
  int ny = 4*scale;
  int nz = 1;

  int steps = 500;
  double dt = 1.;
  double length = 50.;    // X-dim
  double width = 50.;    // Y-dim
  double height = 1.;   // Z-dim
  double pull_rate = 0.2;
  std::string output_name = "paraview_PNC_simple_new";
};

Options parseOptions(int argc, char* argv[])
{
  using namespace smith::cl_parser;
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

struct LocalSimplePNCMaterial {
  using State = smith::Empty;

  template<typename scalar>
  SMITH_HOST_DEVICE auto equilibrium_xi(scalar temp) const{
    using std::pow, std::exp;
    auto Tt = 443.0;
    auto k = 36.0;
    return exp(-(pow(temp/Tt,k)));
  }

  template<typename scalar>
  SMITH_HOST_DEVICE auto Gm0(scalar g) const{
    using std::pow;
    // matrix shear modulus at reference temperature as a function of particle wt% g
    auto Gr = 0.017;      //GPa, rigid modulus
    auto Gs0 = 1.7e-6;    //GPa, soft modulus
    auto Xc = 0.21;       //critical percolation volume fraction
    auto rhof = 2.65;     //g/cc, filler density
    auto rhom = 1.06;     //g/cc, matrix density
    auto n = 0.4;         //percolation exponent
    auto gv = g*rhom/(rhof+g*(rhom-rhof));       //convert weight fraction gw to volume fraction gv
    auto peff = gv*pow(150.,3)/pow(100.,3);      //for 100nm particle, 50nm interphase
    auto Gs = Gs0*(1.+2.5*peff+14.1*peff*peff);  //Guth-Gold correction
    auto X = gv*rhom/(rhof+gv*(rhom-rhof));
    auto psi = 0.;
    if (X>Xc) {
      psi = X*pow((X-Xc)/(1.-Xc),n);
    }
    auto Gnum = (1.-2.*psi+psi*X)*Gr*Gs+(1.-X)*psi*Gr*Gr;
    auto Gdenom = (1.-X)*Gr+(X-psi)*Gs;
    auto G = Gnum/Gdenom; // this is in GPa
    
    return (G*1.e9)/modscale;        // convert to Pa*modscale
  }
  
  template<typename scalar>
  SMITH_HOST_DEVICE auto f1(scalar T) const{
    using std::exp;
    // thermal softening function for low-T modulus
    auto N = 0.02;
    return exp(-N * (T - ambient_temperature));
  }

  template<typename scalar>
  SMITH_HOST_DEVICE auto df1(scalar T) const{
    using std::exp;
    // thermal softening function for low-T modulus
    auto N = 0.02;
    return -N*exp(-N * (T - ambient_temperature));
  }

  template<typename scalar>
  SMITH_HOST_DEVICE auto Ge0(scalar g) const{
    using std::pow;
    // entanglement shear modulus at reference temperature as a function of particle wt% g
    auto Gr = 0.12;      //GPa, rigid modulus
    auto Gs0 = 6.5e-7;   //GPa, soft modulus
    auto Xc = 0.05;      //critical percolation volume fraction
    auto rhof = 2.65;    //g/cc, filler density
    auto rhom = 1.06;    //g/cc, matrix density
    auto n = 1.2;        // percolation exponent
    auto gv = g*rhom/(rhof+g*(rhom-rhof));       //convert weight fraction gw to volume fraction gv
    auto peff = gv*pow(150.,3)/pow(100.,3);      //effective volume fraction
    auto Gs = Gs0*(1.+2.5*peff+14.1*peff*peff);  //Guth-Gold correction
    auto X = gv*rhom/(rhof+gv*(rhom-rhof));
    auto psi = 0.;
    if (X>Xc) {
      psi = X*pow((X-Xc)/(1.-Xc),n);
    }
    auto Gnum = (1.-2.*psi+psi*X)*Gr*Gs+(1.-X)*psi*Gr*Gr;
    auto Gdenom = (1.-X)*Gr+(X-psi)*Gs;
    auto G = Gnum/Gdenom; // this is in GPa
    return (G*1.e9)/modscale;        //this is Pa*modscale
  }

  template <typename T1, typename T2, typename T3, typename T4>
  auto operator()(double /*dt*/, State&, const smith::tensor<T1, dim, dim>& grad_u,
                  const smith::tensor<T2, dim, dim>& grad_v, T3 theta, const smith::tensor<T4, dim>& grad_theta) const
  {
    using std::pow, std::exp;

    theta=theta+ambient_temperature;

    // get equilibrium we=1-xi
    auto we = 1. - equilibrium_xi(theta);

    // get kinematics
    constexpr auto I = smith::Identity<dim>();

    auto F = grad_u + I;

    auto B = dot(F, transpose(F));
    auto trB = tr(B);
    auto B_bar = B - (trB / 3.0) * I;
    auto J = det(F);

    // get moduli
    auto Gm_eff = Gm0(gw)*f1(theta);
    auto Ge_eff = Ge0(gw);

    // calculate B_bar, J based on Fh
    auto Be = dot(F, transpose(F));
    auto trBe = tr(Be);
    auto Be_bar = Be - (trBe / 3.0) * I;

    // calculate kirchoff stress
    auto Tm = Gm_eff * pow(J, -2./3.) * B_bar + J * Km * (J - 1. - betam*(theta-ambient_temperature)) * I;
    auto Te = Ge_eff * pow(J, -2./3.) * Be_bar + J * Ke * (J - 1.) * I;

    auto TK = Tm + we * Te;
  
    // 1st Piola from Kirchhoff
    const auto Piola = dot(TK, inv(transpose(F)));

    // heat flux
    const auto q0 = -thermal_conductivity * grad_theta;

    // internal heat power
    auto greenStrainRate =
        0.5 * (grad_v + transpose(grad_v) + dot(transpose(grad_v), grad_u) + dot(transpose(grad_u), grad_v));
    // derivative of elastic S with respect to T
    const auto dtmdT = Gm0(gw)*df1(theta)*pow(J,-2./3)*B_bar-Km*J*betam*I;
    const auto dSedT = dot(inv(F),dot(dtmdT,transpose(inv(F))));
    const auto s0 = tr(dot(theta*dSedT,greenStrainRate))*0.0;
    
    return smith::tuple{Piola, C_v, s0, q0};
  }
  double Km = 1;             ///< matrix bulk modulus, MPa
  double betam = 1.e-4;      ///< matrix volumetric thermal expansion coefficient
  double Ke = 1;             ///< entanglement bulk modulus, MPa
  double C_v = 25.;          ///< net volumetric heat capacity (must account for matrix+chain+particle)
  double thermal_conductivity = 0.5;    ///< net thermal conductivity (must account for matrix+chain+particle)
  double ambient_temperature = 353.;    ///< reference temperature, K, set to 353
  double gw = 0.3;           ///< particle weight fraction
  double density = 1.0;      ///< net density, not really needed
  double modscale = 1.;      ///< modulus scale factor. The modulus units will be Pa*modscale, so 1=Pa, 1000=kPa, eta.
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
  return smith::NonlinearSolverOptions{.nonlin_solver = smith::NonlinearSolver::TrustRegion,
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
  smith::StateManager::initialize(datastore, "PNC_simple_new_thermomech");

  auto mesh = std::make_shared<smith::Mesh>(
      mfem::Mesh::MakeCartesian3D(options.nx, options.ny, options.nz, mfem::Element::HEXAHEDRON, options.length,
                                  options.width, options.height),
      "thermo_block", 0, 0);

  mesh->addDomainOfBoundaryElements("back", smith::by_attr<example_tm::dim>(1));   //-z
  mesh->addDomainOfBoundaryElements("bottom", smith::by_attr<example_tm::dim>(2));  //-y
  mesh->addDomainOfBoundaryElements("front", smith::by_attr<example_tm::dim>(6));  //+z
  mesh->addDomainOfBoundaryElements("left", smith::by_attr<example_tm::dim>(5));   //-x
  mesh->addDomainOfBoundaryElements("right", smith::by_attr<example_tm::dim>(3)); //+x
  mesh->addDomainOfBoundaryElements("top", smith::by_attr<example_tm::dim>(4));    //+y
  
  auto mechanics_solver = smith::buildNonlinearBlockSolver(example_tm::makeMechanicsNonlinearOptions(),
                                                           example_tm::makeMechanicsLinearOptions(), *mesh);
  auto thermal_solver = smith::buildNonlinearBlockSolver(example_tm::makeThermalNonlinearOptions(),
                                                         example_tm::makeThermalLinearOptions(), *mesh);

  auto staggered_solver = std::make_shared<smith::CoupledSystemSolver>(12);
  staggered_solver->addSubsystemSolver({0}, mechanics_solver);
  staggered_solver->addSubsystemSolver({1}, thermal_solver);

  auto system =
      ::smith::buildThermoMechanicsSystem<example_tm::dim, example_tm::displacement_order, example_tm::temperature_order>(
          mesh, staggered_solver, ::smith::QuasiStaticSecondOrderTimeIntegrationRule{},
          ::smith::QuasiStaticFirstOrderTimeIntegrationRule{}, "bz_staggered",
          std::shared_ptr<::smith::CoupledSystemSolver>{});

  example_tm::LocalSimplePNCMaterial material;
  system.setMaterial(material, mesh->entireBodyName());

  system.disp_bc->setFixedVectorBCs<example_tm::dim>(mesh->domain("bottom"));
  system.disp_bc->setVectorBCs<example_tm::dim>(mesh->domain("top"), std::vector<int>{1},
                                                [](double t, auto X) {
                                                  auto output = 0.0 * X;
                                                  output[1] = -0.50 * sin(M_PI * t / 2.0);
                                                  return output;
                                                });
  system.disp_bc->setVectorBCs<example_tm::dim>(mesh->domain("front"), std::vector<int>{2},
                                                [](double t, auto X) {
                                                  auto output = 0.0 * X;
                                                  output[2] = 0.0 * t;
                                                  return output;
                                                });
  system.disp_bc->setVectorBCs<example_tm::dim>(mesh->domain("back"), std::vector<int>{2},
                                                [](double t, auto X) {
                                                  auto output = 0.0 * X;
                                                  output[2] = 0.0 * t;
                                                  return output;
                                                });
                                                
  system.temperature_bc->setScalarBCs<example_tm::dim>(mesh->domain("front"), [](double t, auto) { 
    if (t < 360) {
      return 120.0*(t/360.); //only the temperature differential here
    }
    else {
      return 120.0;
    } });
  system.temperature_bc->setScalarBCs<example_tm::dim>(mesh->domain("back"), [](double t, auto) {
    if (t < 360) {
      return 120.0*(t/360.); //only the temperature differential here
    }
    else {
      return 120.0;
    }
   });

   // function for the BC - need for reaction forces
  std::function<double(double, tensor<double, example_tm::dim>)> boundary_condition;
  boundary_condition = [](double t, tensor<double, example_tm::dim> /*X*/) -> double {
    return -0.50 * sin(M_PI * t / 2.0);
  };


  example_tm::initializeStates(system, 0.0);

  auto physics = system.createDifferentiablePhysics("bz_staggered_thermomechanics");
  physics->resetStates();

  auto pv_writer = smith::createParaviewWriter(*mesh, system.getStateFields(), options.output_name);
  pv_writer.write(0, physics->time(), system.getStateFields());

  if (mfem::Mpi::Root()) {
    std::cout << "Running staggered quasistatic thermo-mechanics example with " << options.steps
              << " load steps and dt=" << options.dt << "\n";
  }

  // added from kevin
  std::ofstream file;

  std::string stress_strain_output = "paraview_PNC_simple_new/_stress_strain_curve.csv";

  if (mfem::Mpi::Root()) {
    file = std::ofstream(stress_strain_output);

    if (!file.is_open()) {
      MFEM_ABORT("Could Not Open File");
    }
    file << std::setprecision(16) << std::scientific;

    file << "time,strain,force\n";
  }

  double time_increment = options.dt;
  // end added from kevin

  for (int step = 0; step < options.steps; ++step) {
    physics->advanceTimestep(options.dt);

    //-----------ADD FROM PNC_double_network_simple------------
    auto reactions = physics->getReactionStates();
    TimeInfo time_info(physics->time() - time_increment, time_increment);
    double reaction = CalculateReaction(*reactions[0].get(), mesh, "top", 1);
    if (mfem::Mpi::Root()) {
      std::cout << "Reaction: " << reaction << std::endl;
      file << time_info.time() << ","
           << boundary_condition(time_info.time(), {0, 0, 0})/50. << ","
           << reaction / (50*1) << "\n";
      file.flush();
      // 50 is sample height, dividing boundary_condition line by this was getting strain
      // 50*1 is surface area, getting stress instead of rxn force
    }
   // SLIC_INFO_ROOT_FLUSH(axom::fmt::format("    Reaction = {}", reaction));

    // Compute reactions

    pv_writer.write(step + 1, physics->time(), physics->getFieldStatesAndParamStates());
    //-----------------------END ADD---------------------------

    pv_writer.write(static_cast<size_t>(step + 1), physics->time(), system.getStateFields());

    double dispnorm = physics->state(system.prefix("displacement")).Normlinf();
    double tempnorm = physics->state(system.prefix("temperature")).Normlinf();

    // note this outputs junk, I am not using pull_rate
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
