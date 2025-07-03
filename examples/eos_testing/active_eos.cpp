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

#include <mpi.h>
#include <mfem/fem/coefficient.hpp>
#include <mfem/fem/pfespace.hpp>
#include <mfem/fem/pgridfunc.hpp>
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
double CalculateReaction(FiniteElementDual & reactions, const int face, const int direction) { 

  auto fespace = reactions.space();

  mfem::ParGridFunction mask_gf(&fespace);
  mask_gf = 0.0;
  auto fhan = [direction](const mfem::Vector &, mfem::Vector & y){
    y = 0.0;
    y[direction] = 1.0; 
  };

  mfem::VectorFunctionCoefficient boundary_coeff(3, fhan);
  mfem::Array<int> ess_bdr(fespace.GetParMesh()->bdr_attributes.Max());
  ess_bdr = 0;
  ess_bdr[face] = 1;
  mask_gf.ProjectBdrCoefficient(boundary_coeff, ess_bdr);

  mfem::Vector projections = reactions;
  mfem::Vector mask_t(fespace.GetTrueVSize());
  mask_t = 0.0;
  fespace.GetRestrictionMatrix()->Mult(mask_gf, mask_t);
  projections *= mask_t;


  double local_sum = projections.Sum();
  MPI_Comm mycomm = fespace.GetParMesh()->GetComm();
  double global_sum = 0.0;
  MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, mycomm);

  return global_sum; 
}


double CalculateReaction2(FiniteElementDual & reactions, const int face, const int direction) { 

  FiniteElementState mask(reactions.space(), "reaction_mask");

  // auto & pmesh = reactions.mesh();
 
  mfem::VectorFunctionCoefficient func(3, [direction](const mfem::Vector& /*x*/, mfem::Vector& u) {
    u = 0.0;
    u[direction] = 1.0;
  });
 
  mfem::Array<int> ess_bdr(reactions.space().GetParMesh()->bdr_attributes.Max());
  ess_bdr = 0;
  ess_bdr[face] = 1;
  mfem::Array<int> ess_bdr_tdof;
  reactions.space().GetEssentialTrueDofs(ess_bdr, ess_bdr_tdof);
  std::cout << "gothere" << std::endl;
  mask.project(func,ess_bdr_tdof);
  std::cout << "gothereafter" << std::endl;
 
 
  mask *= reactions;


  double local_sum = mask.Sum();
  MPI_Comm mycomm = reactions.space().GetParMesh()->GetComm();
  double global_sum = 0.0;
  MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, mycomm);

  // double totalForce = serac::innerProduct(mask, reactions);

  return global_sum; 
}

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
      return 120. * time + 353.;
   }

  template <typename T, int dim>
  SERAC_HOST_DEVICE auto operator()(State& state, const tensor<T, dim, dim>& du_dX) const
  {


    auto whp = state.w_H;

    auto current_time = state.time;

    auto temperature = temperature_function(current_time);

    auto value = A * exp(-E_a / (R*temperature)) * dt;

    auto wh = whp + (1. - whp) * value / (1. + value);
    // std::cout << "wh: " << wh << "\n";

    constexpr auto I = Identity<dim>();

    auto F = du_dX + I;

    auto B = dot(F, transpose(F));
    auto trB = tr(B);
    auto B_bar = B - (trB / 3.0) * I;
    auto J = det(F);

    auto T0 = 353.;
    auto N = 0.02;
    auto Gl_eff = Gl * exp(-N * (temperature - T0));
    auto Tl = Gl_eff * pow(J, -2./3.) * B_bar + J * Kl * (J - 1.) * I;
    auto Th = Gh * pow(J, -2./3.) * B_bar + J * Kh * (J - 1.) * I;

    auto TK = wh * Th + (1. - wh) * Tl;
    


    // Pull back to Piola

    state.w_H = get_value(wh);
    state.time = get_value(current_time + dt);
    return dot(TK, inv(transpose(F)));
  }

  double density;  ///< mass density
  double Kl;        ///< bulk modulus
  double Gl;        ///< shear modulus

  double Kh;        ///< bulk modulus
  double Gh;        ///< shear modulus

  double dt;        ///< fixed dt

  double A;
  double E_a;

  double R;
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
  double dt = 0.005;

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
  double Kl = 0.5;        ///< bulk modulus
  double Gl = 0.0074;        ///< shear modulus

  double Kh = 0.5;        ///< bulk modulus
  double Gh = 0.225;        ///< shear modulus


  double A = 1.5e18;
  double E_a = 1.5e5;

  double R = 8.314;
  Material mat{.density=1.0,.Kl=Kl, .Gl=Gl,.Kh=Kh,.Gh=Gh,.dt=dt,.A=A,.E_a=E_a,.R=R};
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
    double tstar = 0.2;
    double ramp = 0.1;
    double sin_mag = 0.05;
    double freq = 10.;
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


  std::ofstream reaction_log;
  if (mfem::Mpi::Root()){
    reaction_log.open("reaction_log.csv");
  }


  double T_Max = 2.0;
  while (solid_solver->time() < T_Max && std::abs(solid_solver->time() - T_Max) > DBL_EPSILON) {
    SLIC_INFO_ROOT(axom::fmt::format("time = {}", solid_solver->time()));
    serac::logger::flush();

    // Refine dt as contact starts
    // auto next_dt = solid_solver->time() < 0.65 ? dt : dt * 0.1;
    
    solid_solver->advanceTimestep(dt);

    // Output the sidre-based plot files
    solid_solver->outputStateToDisk(paraview_name);
    auto reactions = solid_solver->reactions();

    
    
    
    double val = CalculateReaction(reactions, 1, 1);
    if (mfem::Mpi::Root()){
      std::cout << "---------------------------------" << std::endl;
      std::cout << "val: " << val << std::endl;
      std::cout << "---------------------------------" << std::endl;
      reaction_log << solid_solver->time() << "," << val << "\n";
    }
  }
  SLIC_INFO_ROOT(axom::fmt::format("final time = {}", solid_solver->time()));
  if (mfem::Mpi::Root()){
    reaction_log.close();
  }

  return 0;
}
