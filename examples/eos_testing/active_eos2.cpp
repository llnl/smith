// Copyright (c) Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file cylinder.cpp
 *
 * @brief A buckling cylinder under compression, run with or without contact
 *
 * @note Run with mortar contact and PETSc preconditioners:
 * @code{.sh}
 * ./build/examples/buckling_cylinder --contact --contact-type 1 --preconditioner 6 \
 *    -options_file examples/buckling/cylinder_petsc_options.yml
 * @endcode
 * @note Run with penalty contact and HYPRE BoomerAMG preconditioner
 * @code{.sh}
 * ./build/examples/buckling_cylinder
 * @endcode
 * @note Run without contact:
 * @code{.sh}
 * ./build/examples/buckling_cylinder --no-contact
 * @endcode
 */

#include <set>
#include <string>

#include "axom/slic.hpp"
#include "axom/inlet.hpp"
#include "axom/CLI11.hpp"

#include "mfem.hpp"

#include "serac/serac.hpp"

using namespace serac;

/**
 * @brief Run buckling cylinder example
 *
 * @note Based on doi:10.1016/j.cma.2014.08.012
 */
namespace Zimmerman_EOS{
struct NeoHookean {
  // using State = Empty;  ///< this material has no internal variables
  struct State {
    double internal_variable;
  };

  /**
   * @brief stress calculation for a NeoHookean material model
   *
   * When applied to 2D displacement gradients, the stress is computed in plane strain,
   * returning only the in-plane components.
   *
   * @tparam T Number-like type for the displacement gradient components
   * @tparam dim Dimensionality of space
   * @param du_dX displacement gradient with respect to the reference configuration (displacement_grad)
   * @return The first Piola stress
   */
  template <typename T, int dim>
  SERAC_HOST_DEVICE auto operator()(State& state, const tensor<T, dim, dim>& du_dX) const
  {

    using std::log1p;

    double new_value = state.internal_variable;

    constexpr auto I = Identity<dim>();
    auto lambda = K - (2.0 / 3.0) * G + new_value;
    auto B_minus_I = dot(du_dX, transpose(du_dX)) + transpose(du_dX) + du_dX;

    auto logJ = log1p(detApIm1(du_dX));
    // Kirchoff stress, in form that avoids cancellation error when F is near I
    auto TK = lambda * logJ * I + G * B_minus_I;

    // Pull back to Piola
    auto F = du_dX + I;

    state.internal_variable = get_value(new_value);
    return dot(TK, inv(transpose(F)));
  }

  double density;  ///< mass density
  double K;        ///< bulk modulus
  double G;        ///< shear modulus
};

struct ThermalStiffening {
  // using State = Empty;  ///< this material has no internal variables
  struct State {
    double w_H = 0.0;
    double time = 0.0;
    tensor<double,3,3> Cp{{{1.0, 0.0, 0.0},
                           {0.0, 1.0, 0.0},
                           {0.0, 0.0, 1.0}}}; // maybe could've done Identity<3>()
  };

  /**
   * @brief stress calculation for a NeoHookean material model
   *
   * When applied to 2D displacement gradients, the stress is computed in plane strain,
   * returning only the in-plane components.
   *
   * @tparam T Number-like type for the displacement gradient components
   * @tparam dim Dimensionality of space
   * @param du_dX displacement gradient with respect to the reference configuration (displacement_grad)
   * @return The first Piola stress
   */

   SERAC_HOST_DEVICE auto temperature_function(double time) const{
      // return 120. * time + 353.;
      /*
      if (time <= 800.) {
        return 80. * (time/800.) + 353.;
      }
      else {
        return 433.;
      }
        */
      return 120. * (time/1440.) + 353.;
   }

  template <typename T, int dim>
  SERAC_HOST_DEVICE auto operator()(State& state, const tensor<T, dim, dim>& du_dX) const
  {


    auto whp = state.w_H;

    auto current_time = state.time;

    auto Cp = state.Cp;

    auto temperature = temperature_function(current_time);

    auto value = A * exp(-E_a / (R*temperature)) * dt;

    auto wh = whp + (1. - whp) * value / (1. + value);
    //std::cout << "wh: " << wh << "\n";

    constexpr auto I = Identity<dim>();

    auto F = du_dX + I;

    auto C = dot(transpose(F), F);
    auto Cdot = (C - Cp)/dt;
    auto CdFi = dot(Cdot, inv(F));
    auto D = dot(inv(transpose(F)),CdFi)*0.5;

    auto B = dot(F, transpose(F));
    auto trB = tr(B);
    auto B_bar = B - (trB / 3.0) * I;
    auto J = det(F);

    auto T0 = 353.;
    auto N = 0.02;
    auto Gl_eff = Gl * exp(-N * (temperature - T0));
    auto Tl = Gl_eff * pow(J, -2./3.) * B_bar + J * Kl * (J - 1.) * I + etal * D;
    auto Th = Gh * pow(J, -2./3.) * B_bar + J * Kh * (J - 1.) * I + etah * D;

    auto TK = wh * Th + (1. - wh) * Tl;
    


    // Pull back to Piola

    state.w_H = get_value(wh);
    state.time = get_value(current_time + dt);
    state.Cp = get_value(dot(transpose(F),F));
    return dot(TK, inv(transpose(F)));
  }

  double density;  ///< mass density
  double Kl;        ///< bulk modulus
  double Gl;        ///< shear modulus
  double etal;      ///< viscosity

  double Kh;        ///< bulk modulus
  double Gh;        ///< shear modulus
  double etah;      ///< viscosity

  double dt;        ///< fixed dt

  double A;         ///< exponential prefactor
  double E_a;       ///< activation energy

  double R;         ///< universal gas constant
};
};

int main(int argc, char* argv[])
{
  constexpr int dim = 3;
  constexpr int p = 1;

  // Command line arguments
  // Mesh options
  int serial_refinement = 0;
  int parallel_refinement = 0;
  double dt = 2./3.;

  // Solver options
  NonlinearSolverOptions nonlinear_options = solid_mechanics::default_nonlinear_options;
  LinearSolverOptions linear_options = solid_mechanics::default_linear_options;
  nonlinear_options.nonlin_solver = serac::NonlinearSolver::TrustRegion;
  nonlinear_options.relative_tol = 1e-8;
  nonlinear_options.absolute_tol = 1e-14;
  nonlinear_options.min_iterations = 1;
  nonlinear_options.max_iterations = 500;
  nonlinear_options.max_line_search_iterations = 20;
  nonlinear_options.print_level = 1;
#ifdef SERAC_USE_PETSC
  linear_options.linear_solver = serac::LinearSolver::GMRES;
  linear_options.preconditioner = serac::Preconditioner::HypreAMG;
  linear_options.relative_tol = 1e-8;
  linear_options.absolute_tol = 1e-16;
  linear_options.max_iterations = 2000;
#endif

  // Contact specific options

  // Initialize and automatically finalize MPI and other libraries
  serac::ApplicationManager applicationManager(argc, argv);


  // Handle command line arguments
  axom::CLI::App app{"Hollow cylinder buckling example"};
  // Mesh options
  app.add_option("--serial-refinement", serial_refinement, "Serial refinement steps")->check(axom::CLI::PositiveNumber);
  app.add_option("--parallel-refinement", parallel_refinement, "Parallel refinement steps")
      ->check(axom::CLI::PositiveNumber);

  nonlinear_options.force_monolithic = linear_options.preconditioner != Preconditioner::Petsc;


  // Create DataStore
  std::string name = "active_eos";
  std::string mesh_tag = "mesh";
  axom::sidre::DataStore datastore;
  serac::StateManager::initialize(datastore, name + "_data");

  // Create and refine mesh
  // std::string filename = SERAC_REPO_DIR "/data/meshes/hollow-cylinder.mesh";
  std::string filename = SERAC_REPO_DIR "/data/meshes/full_cylinder.g";
  auto mesh = serac::buildMeshFromFile(filename);
  auto refined_mesh = mesh::refineAndDistribute(std::move(mesh), serial_refinement, parallel_refinement);
  auto& pmesh = serac::StateManager::setMesh(std::move(refined_mesh), mesh_tag);

  // Surfaces for boundary conditions
  // constexpr int xneg_attr{2};
  // constexpr int xpos_attr{3};
  // auto xneg = serac::Domain::ofBoundaryElements(pmesh, serac::by_attr<dim>(xneg_attr));
  // auto xpos = serac::Domain::ofBoundaryElements(pmesh, serac::by_attr<dim>(xpos_attr));
  auto bottom = serac::Domain::ofBoundaryElements(pmesh, serac::by_attr<dim>(1));
  auto top = serac::Domain::ofBoundaryElements(pmesh, serac::by_attr<dim>(2));

  // Create solver, either with or without contact


  std::unique_ptr<SolidMechanics<p, dim>> solid_solver;
    solid_solver = std::make_unique<serac::SolidMechanics<p, dim>>(
        nonlinear_options, linear_options, serac::solid_mechanics::default_quasistatic_options, name, mesh_tag);
    // solid_solver->setPressure([&](auto&, double t) { return 0.01 * t; }, xpos);


  


  using Material = Zimmerman_EOS::ThermalStiffening;
  // Units are standard FEBio: Mpa-mm-s
  double Kl = 0.5;       ///< low-T bulk modulus, MPa
  double Gl = 0.001759;    ///< low-T shear modulus, MPa
  double etal = 0.005;   ///< low-T viscosity, MPa-s

  double Kh = 0.5;       ///< high-T bulk modulus, MPa
  double Gh = 0.0006408;     ///< high-T shear modulus, MPa
  double etah = 0.0;     ///< high-T viscosity, MPa-s

  // E_a and R can be SI units since they cancel out in the exponent
  double A = 2.5e15;      ///< forward (low-high) exponential prefactor, 1/s
  double E_a = 1.5e5;    ///< forward (low-high) activation energy, J/mol
  double R = 8.314;      ///< universal gas constant, J/mol/K

  Material mat{.density=1.0,.Kl=Kl, .Gl=Gl, .etal=etal, .Kh=Kh, .Gh=Gh, .etah=etah, .dt=dt,.A=A,.E_a=E_a,.R=R};

  // Set tfinal
  double tfinal = 1440.; // I want 120 degree rise at 5 degrees per minute

  // Material mat{.density = 1.0, .K = (3 * lambda + 2 * G) / 3, .G = G};
  Domain whole_mesh = EntireDomain(pmesh);
  auto internal_states = solid_solver->createQuadratureDataBuffer(Material::State{},whole_mesh); 
  solid_solver->setMaterial(mat, whole_mesh, internal_states);

  // Set up essential boundary conditions
  // Bottom of cylinder is fixed

  solid_solver->setFixedBCs(bottom); 

  // Top of cylinder has prescribed displacement of magnitude in x-z direction
  auto compress = [&](const serac::tensor<double, dim>, double t) {
    serac::tensor<double, dim> u{};
    // u[2] = 1.5 / std::sqrt(2.0) * t;
    double val = 0.0;
    double tstar = 0.; // was 30
    double ramp = 0.; //was 0.1
    double sin_mag = 0.01; //0.05
    double freq = 0.125; //0.0166666; // was 10
    if (t < tstar){
      val = ramp * (t / tstar);
    } else {
      val = ramp + sin_mag * sin(2.*M_PI * freq * (t-tstar));
    }
    u[1] = val;
    return u;
  };
  // solid_solver->setDisplacementBCs(compress, top, Component::X + Component::Z);
  // solid_solver->setDisplacementBCs(compress, top,
  //                                  Component::Y);  // BT: Would it be better to leave this component free?
  solid_solver->setDisplacementBCs(compress, top, Component::ALL);

  // Finalize the data structures
  solid_solver->completeSetup();

  // Save initial state
  std::string paraview_name = name + "_paraview";
  solid_solver->outputStateToDisk(paraview_name);

  // Perform the quasi-static solve
  SLIC_INFO_ROOT(axom::fmt::format("Running hollow cylinder bucking example with {} displacement dofs",
                                   solid_solver->displacement().GlobalSize()));
  SLIC_INFO_ROOT("Starting pseudo-timestepping.");
  serac::logger::flush();
  while (solid_solver->time() < tfinal && std::abs(solid_solver->time() - tfinal) > DBL_EPSILON) {
    SLIC_INFO_ROOT(axom::fmt::format("time = {} (out of 1.0)", solid_solver->time()));
    serac::logger::flush();

    // Refine dt as contact starts
    // auto next_dt = solid_solver->time() < 0.65 ? dt : dt * 0.1;
    
    solid_solver->advanceTimestep(dt);

    // Output the sidre-based plot files
    solid_solver->outputStateToDisk(paraview_name);
  }
  SLIC_INFO_ROOT(axom::fmt::format("final time = {}", solid_solver->time()));

  return 0;
}
