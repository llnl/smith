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
 //#include <deque>
 
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
   // using State = Empty;  ///< if this material has no internal variables
   struct State {
     double w_e = 0.0;   //high-T mass fraction
     double time = 0.0;  // time for scaling temperature ramp
     tensor<double,3,3> Cp{{{1.0, 0.0, 0.0},
                            {0.0, 1.0, 0.0},
                            {0.0, 0.0, 1.0}}}; // previous value of right Cauchy-Green
     tensor<double,3,3> Fesi{{{1.0, 0.0, 0.0},
                              {0.0, 1.0, 0.0},
                              {0.0, 0.0, 1.0}}}; 
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
       //for ramp-and-hold
       /*
       if (time <= 242.) {
         return 120. * (time/288.) + 353.;
       }
       else {
         return 453.83;
       }
       */  

       // for simple ramp
       //return 120. * (time/1440.) + 353.;

       // for reversible steady ramp
       double T_INC = 1440;//2880;
       double T_DEC = 1440;//2880;
       if (time <= T_INC) {
        return 120. * (time/T_INC) + 353.;
       }
       else {
        return 473. - 120.*((time-T_INC)/T_DEC);
       }
       
    }

    // this function calculates the equilibrium low-T mass fraction as a function of temperature
    SERAC_HOST_DEVICE auto equilibrium_xi(double temp) const{
        double Tt = 443.0;
        double k = 36.0;
        return exp(-(pow(temp/Tt,k)));
     }

     SERAC_HOST_DEVICE auto Gm0(double g) const{
      // low-T shear modulus at reference temperature as a function of particle wt% g
      double junk = g;
      return Gm*junk/g;
     }

     SERAC_HOST_DEVICE auto f1(double T) const{
      // thermal softening function for low-T modulus
      auto N = 0.02;
      return exp(-N * (T - Tr));
     }

     SERAC_HOST_DEVICE auto Ge0(double g) const{
      // high-T shear modulus at reference temperature as a function of particle wt% g
      double junk = g;
      return Ge*junk/g;
     }
 
   template <typename T, int dim>
   SERAC_HOST_DEVICE auto operator()(State& state, const tensor<T, dim, dim>& du_dX) const
   {
 
 
     auto wep = state.w_e; // previous wh
     auto wfp = 1.0-wep;
     auto current_time = state.time;
     auto Cp = state.Cp;
     auto Fesip = state.Fesi;
 
     auto Temp = temperature_function(current_time);

     // get equilibrium wl=xi
     auto xi = equilibrium_xi(Temp);
     //std::cout << "wh: " << wh << "\n";

     // get kinematics
 
     constexpr auto I = Identity<dim>();
 
     auto F = du_dX + I;
     auto Fe = dot(F,Fesip); // Fh for the extant high-T material, called Fh1 in my notes
     auto Je = det(Fe);
     //auto Ce = dot(transpose(Fe),Fe);
 
     auto C = dot(transpose(F), F);
     auto Cdot = (C - Cp)/dt;
     auto CdFi = dot(Cdot, inv(F));
     auto D = dot(inv(transpose(F)),CdFi)*0.5;
 
     auto B = dot(F, transpose(F));
     auto trB = tr(B);
     auto B_bar = B - (trB / 3.0) * I;
     auto J = det(F);

     // get moduli
     auto Gm_eff = Gm0(gw)*f1(Temp);
     auto Ge_eff = Ge0(gw);

     // calculate forward and reverse reaction rate
     auto kf = Af * exp(-E_af / (R*Temp));
     auto kr = Ar * exp(-E_ar / (R*Temp));

     // get mass fraction supplies, forward and reverse
     auto dwff = (xi-wfp)*kf*dt/(1.+kf*dt);
     auto dwer = (1.-xi-wep)*kr*dt/(1.+kr*dt);
     // get net mass fraction supply
     auto dwe = -dwff + dwer;

     // if dwh>0, I need to get the new equivalent Fhsi
     if (dwe>0 && wep==0) {
      auto Fesi = inv(F); // initialize Fhsi as the inverse of F at the current time
      Fe = dot(F,Fesi);
      state.Fesi = get_value(Fesi);
     }
     else if (dwe>0) {
      auto Fesi = (wep/(wep+dwe))*Fesip; // update the effective value of Fhsi
      Fe = dot(F,Fesi); // calculate the current elastic deformation of the high-T material
      state.Fesi = get_value(Fesi);
     }
     else {
      auto Fesi = Fesip;
      Fe = dot(F,Fesi);
      state.Fesi = get_value(Fesi);
     }

     // update mass fractions
     auto we = wep + dwe;

     std::cout << "we: " << we << "\n";

    // calculate B_bar, J based on Fh
     auto Be = dot(Fe, transpose(Fe));
     auto trBe = tr(Be);
     auto Be_bar = Be - (trBe / 3.0) * I;

     // calculate kirchoff stress
     auto Tm = Gm_eff * pow(J, -2./3.) * B_bar + J * Km * (J - 1. - betam*(Temp-Tr)) * I; // + etal * D;
     auto Te = Ge_eff * pow(Je, -2./3.) * Be_bar + Je * Ke * (Je - 1.) * I; // + etah * D;
 
     auto TK = wm * Tm + (1. - wm) * we * Te + (1.-we)*eta*D;
   
     // Pull back to Piola
 
     state.w_e = get_value(we);
     state.time = get_value(current_time + dt);
     state.Cp = get_value(dot(transpose(F),F));
     return dot(TK, inv(transpose(F)));
   }
 
   double density;   ///< mass density
   double Km;        ///< bulk modulus
   double Gm;        ///< shear modulus
   double betam;     ///< volumetric CTE
   double cvm;       ///< specific heat
   double rhom0;     ///< referential density
   double eta;      ///< viscosity
 
   double Ke;        ///< bulk modulus
   double Ge;        ///< shear modulus
   double cvc;       ///< specific heat
   double rhoc0;     ///< referential density
 
   double dt;        ///< fixed dt
 
   double Af;        ///< forward (l->h) exponential prefactor
   double E_af;      ///< forward activation energy

   double Ar;        ///< reverse (h->l) exponential prefactor
   double E_ar;      ///< reverse activation energy
 
   double R;         ///< universal gas constant
   double Tr;        ///< reference temperature

   double gw;        ///< particle weight fraction

   double wm;        ///< matrix mass fraction
 };
 };
 
 int main(int argc, char* argv[])
 {
   constexpr int dim = 3;
   constexpr int p = 1;

   //============= time and temp params ================

   // I always want 120 degree temperature rise - specify deg/min gives tfinal
   double degPerMin = 5;
   double tfinal = 2*(120/degPerMin)*60.;     // twice the amount of time required to heat up to account for cool down
   double Npts = 24;                          // number of time points per cycle
   double Ncycle = 240;                       // number of sinusoidal cycles up and back
   //=========== end time and temp params ==============
 
   // Command line arguments
   // Mesh options
   int serial_refinement = 0;
   int parallel_refinement = 0;
   double dt = tfinal/(Ncycle*Npts);
 
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
   std::string filename = SERAC_REPO_DIR "/data/meshes/2x2_cube.g";//full_cylinder.g";
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
   double Km = 0.5;         ///< matrix bulk modulus, MPa
   double Gm = 0.0073976;//0.001759;    ///< matrix shear modulus, MPa
   double betam = 0;
   double cvm = 1.0;
   double rhom0 = 1.0;
   double eta = 0.;//0.002;//0.005;     ///< viscosity, MPa-s
 
   double Ke = 0.5;       ///< entanglement bulk modulus, MPa
   double Ge = 0.225075;//0.0006408;     ///< entanglement shear modulus, MPa
   double cvc = 4.0;
   double rhoc0 = 1.0;
 
   // E_a and R can be SI units since they cancel out in the exponent
   double Af = 2.5e15;      ///< forward (low-high) exponential prefactor, 1/s
   double E_af = 1.5e5;     ///< forward (low-high) activation energy, J/mol
   double Ar = 1.0e-21;//4.2e-24;      ///< reverse exponential prefactor, 1/s
   double E_ar = -1.55e5;//-1.5e5;    ///< reverse activation energy, J/mol
   double R = 8.314;        ///< universal gas constant, J/mol/K
   double Tr = 353;         ///< reference temperature, K

   double gw = 0.2;         ///< particle weight fraction

   double wm = 0.5;         ///< matrix mass fraction
 

   Material mat{.density=1.0,.Km=Km, .Gm=Gm, .betam=betam, .cvm=cvm, .rhom0=rhom0, .eta=eta, .Ke=Ke, .Ge=Ge, .cvc=cvc, .rhoc0=rhoc0, .dt=dt,.Af=Af,.E_af=E_af, .Ar=Ar, .E_ar=E_ar, .R=R, .Tr=Tr, .gw=gw, .wm=wm};

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
     /*
     // this section is a ramp and hold
     double val = 0.0;
     double tstar = 100.0;
     double ramp = 0.05;
     if (t < tstar){
      val = ramp * (t / tstar);
     } else {
      val = ramp;
     }
     u[1] = val;
     */
      //this section applies sinusoidal loading
     double val = 0.0;
     double tstar = 0.; // was 30
     double ramp = 0.; //was 0.1
     double sin_mag = 0.05; //0.05
     //double freq = 0.125; //0.0166666; // was 10
     double freq = Ncycle/tfinal; // frequency is set to enforce the number of cycles specified earlier
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