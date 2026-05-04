// Copyright (c) Lawrence Livermore National Security, LLC and
// other SMITH Project Developers. See the top-level LICENSE file for
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
 
#include "smith/numerics/functional/domain.hpp"
 #include "smith/smith.hpp"
 
 #define USING_NOTCHED_CUTOUT
//  #undef USING_NOTCHED_CUTOUT
 using namespace smith;
 
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
   SMITH_HOST_DEVICE auto operator()(State& state, const tensor<T, dim, dim>& du_dX) const
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
 
 struct NeoHookeanVisco {
   // using State = Empty;  ///< if this material has no internal variables
   struct State {
     double time = 0.0;  // time
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
   SMITH_HOST_DEVICE auto operator()(State& state, const tensor<T, dim, dim>& du_dX) const
   {
    using std::exp;
    using std::pow;
   
     // update time
     auto time = state.time;
     //auto t = time + dt;

     // get kinematics
     constexpr auto I = Identity<dim>();
     auto F = du_dX + I;
     auto J = det(F);

     auto B = dot(F, transpose(F));

     auto Btilde = pow(J,-2./3)*B;
     auto I1tilde = tr(Btilde);
     auto Bbar = Btilde - (I1tilde/3.)*I;

     // set up viscoelastic function fn
     auto fn = 1.0;

     // calculate Kirchhoff stress
     if (tau1 != 0.0) {
      fn = 1.-g1*(1.-exp(-time/tau1));
     }
     auto TK = 2.*(C10*Bbar*fn+ p*J*J*I);
   
     // Pull back to Piola and store the current H1 as the state H1n
     state.time = get_value(time+dt);
     return dot(TK, inv(transpose(F)));
   }
 
   double density;
   double p;         ///< bulk modulus-like term for near-incompressibility
   double C10;       ///< shear modulus
   double g1;        ///< Prony series coefficient
   double tau1;      ///< viscoelastic time constant
   double dt;        ///< fixed dt

 };

 struct NeoHookeanViscoFEBio {
   // using State = Empty;  ///< if this material has no internal variables
   struct State {
     double time = 0.0;  // time
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
   SMITH_HOST_DEVICE auto operator()(State& state, const tensor<T, dim, dim>& du_dX) const
   {
    using std::exp;
    using std::pow;
   
     // update time
     auto time = state.time;
     //auto t = time + dt;

     auto mu = G; // second Lame parameter
     auto lam = K-2.*G/3.; // first Lame parameter

     // get kinematics
     constexpr auto I = Identity<dim>();
     auto F = du_dX + I;
     auto J = det(F);
     auto B = dot(F, transpose(F));

     // set up viscoelastic function fn with one Prony term
     auto fn = 1.0;
     if (tau1 != 0.0) {
      fn = 1.-g1*(1.-exp(-time/tau1));
     }
     // calculate Kirchhoff stress
     auto TK = mu*fn*(B-I)+lam*log(J)*I;
   
     // Pull back to Piola and store the current H1 as the state H1n
     state.time = get_value(time+dt);
     return dot(TK, inv(transpose(F)));
   }
 
   double density;
   double K;         ///< bulk modulus
   double G;       ///< shear modulus
   double g1;        ///< Prony series coefficient
   double tau1;      ///< viscoelastic time constant
   double dt;        ///< fixed dt

 };
 };
 
 int main(int argc, char* argv[])
 {
   constexpr int dim = 3;
   constexpr int p = 2;

   //============= time stepping params ================
   double tfinal = 100; // 35;     // final time
   double Npts = 300;       // number of time steps
   //=========== end time stepping params ==============
 
   // Command line arguments
   // Mesh options
   int serial_refinement = 0;
   int parallel_refinement = 0;
   double dt = tfinal/Npts;
 
   // Solver options
  smith::LinearSolverOptions linear_options{.linear_solver = LinearSolver::GMRES,
                                            .preconditioner = Preconditioner::HypreAMG,
                                            .relative_tol = 1.0e-8,
                                            .absolute_tol = 1.0e-16,
                                            .max_iterations = 2000,
                                            .print_level = 0};
  smith::NonlinearSolverOptions nonlinear_options{.nonlin_solver = NonlinearSolver::TrustRegion,
                                                     .relative_tol = 1.0e-8,
                                                     .absolute_tol = 1.0e-11,
                                                     .min_iterations = 2,
                                                     .max_iterations = 500,
                                                     .max_line_search_iterations = 30,
                                                     .print_level = 1};
 
   // Contact specific options
 
   // Initialize and automatically finalize MPI and other libraries
   smith::ApplicationManager applicationManager(argc, argv);
 
 
   // Handle command line arguments
   axom::CLI::App app{"Hollow cylinder buckling example"};
   // Mesh options
   app.add_option("--serial-refinement", serial_refinement, "Serial refinement steps")->check(axom::CLI::PositiveNumber);
   app.add_option("--parallel-refinement", parallel_refinement, "Parallel refinement steps")
       ->check(axom::CLI::PositiveNumber);
 
   // Create DataStore
#ifdef USING_NOTCHED_CUTOUT
   std::string name = "bibeam_test_notched";
#else
   std::string name = "bibeam_test";
#endif
   std::string mesh_tag = "mesh";
   axom::sidre::DataStore datastore;
   smith::StateManager::initialize(datastore, name + "_data");
 
   // Create and refine mesh
#ifdef USING_NOTCHED_CUTOUT
  //  std::string filename = SMITH_REPO_DIR "/data/meshes/bibeam_notched_120x21x1mm.g";//full_cylinder.g";
  std::string filename = SMITH_REPO_DIR "/data/meshes/bibeam_notched_120x21x1mm_3D.g";//full_cylinder.g";
#else
   std::string filename = SMITH_REPO_DIR "/data/meshes/bibeam_120x21x1mm.g";//full_cylinder.g";
#endif
   auto mesh = std::make_shared<smith::Mesh>(filename, mesh_tag, serial_refinement, parallel_refinement);

   // Surfaces for boundary conditions
   mesh->addDomainOfBoundaryElements("bottom_face", smith::by_attr<dim>(1));
   mesh->addDomainOfBoundaryElements("top_face", smith::by_attr<dim>(2));
   mesh->addDomainOfBoundaryElements("front_face", smith::by_attr<dim>(3));
   mesh->addDomainOfBoundaryElements("back_face", smith::by_attr<dim>(4));

  // define all my domains. Left and Right defined by spatial position
  auto everything = [](std::vector<tensor<double, dim>>, int /* attr */) { return true; };
  auto on_left = [](std::vector<tensor<double, dim>> X, int /* attr */) { return average(X)[0] < 0.0; };
  auto on_right = [](std::vector<tensor<double, dim>> X, int /* attr */) { return average(X)[0] >= 0.0; };

  mesh->addDomainOfBodyElements("left_mesh", on_left);
  mesh->addDomainOfBodyElements("right_mesh", on_right);
  mesh->addDomainOfBodyElements("whole_mesh", everything);

   // Create solver
   std::unique_ptr<SolidMechanics<p, dim>> solid_solver;
     solid_solver = std::make_unique<smith::SolidMechanics<p, dim>>(
         nonlinear_options, linear_options, smith::solid_mechanics::default_quasistatic_options, name, mesh);

   using Material = Zimmerman_EOS::NeoHookeanViscoFEBio;
   // Units are standard FEBio: Mpa-mm-s
     double h_density = 1.0;
     double h_K = 10.0;      // bulk modulus, MPa
     double h_G = 1.0;       // shear modulus, MPa
     double h_g1 = 0.0;      // Prony series coefficient
     double h_tau1 = 0.0;    // viscoelastic time constant

   // Units are standard FEBio: Mpa-mm-s
     double v_density = 1.0;
     double v_K = 10.0;      // bulk modulus, MPa
     double v_G = 1.4;       // shear modulus, MPa
     double v_g1 = 0.5;      // Prony series coefficient
     double v_tau1 = 10.0;   // viscoelastic time constant

   Material mat_hyper{.density=h_density, .K=h_K, .G=h_G, .g1=h_g1, .tau1=h_tau1, .dt=dt};
   Material mat_visco{.density=v_density, .K=v_K, .G=v_G, .g1=v_g1, .tau1=v_tau1, .dt=dt};
  
   //Domain whole_mesh = EntireDomain(mesh);
   auto internal_states = solid_solver->createQuadratureDataBuffer(Material::State{}, mesh->domain("whole_mesh")); 

   solid_solver->setMaterial(mat_hyper, mesh->domain("left_mesh"), internal_states);
   solid_solver->setMaterial(mat_visco, mesh->domain("right_mesh"), internal_states);
 
   // Set up essential boundary conditions
   // Bottom of cylinder is fixed
   solid_solver->setFixedBCs(mesh->domain("bottom_face")); 
 
   // Top of cylinder has prescribed displacement of magnitude in x-z direction
   auto compress = [&](const smith::tensor<double, dim>, double t) {
     smith::tensor<double, dim> u{};
     // this section is a fast ramp
     //simple strain rate is velocity/gauge length, so v/120
     //normalized strain rate as in the paper is v_tau1*velocity/120
     //I need edot=0.1, =>velocity=1.2?
     double velocity = 0.60; //mm/s
     double val = velocity*t;
     u[1] = -val;
     return u;
   };
   // solid_solver->setDisplacementBCs(compress, top, Component::X + Component::Z);
   // solid_solver->setDisplacementBCs(compress, top,
   //                                  Component::Y);  // BT: Would it be better to leave this component free?
   solid_solver->setDisplacementBCs(compress, mesh->domain("top_face"), Component::ALL);

   // global constraint to enforce 2D conditions
   solid_solver->setFixedBCs(mesh->domain("whole_mesh"),Component::Z);
 
   // Finalize the data structures
   solid_solver->completeSetup();
 
   // Save initial state
  //  std::string paraview_name = "sol_" + name + "_12mm_s_paraview";
  //  std::string paraview_name = "sol_" + name + "_1p2mm_s_paraview";
   std::string paraview_name = "sol_" + name + "_0p6mm_s_paraview";
   solid_solver->outputStateToDisk(paraview_name);
 
   // Perform the quasi-static solve
   SLIC_INFO_ROOT(axom::fmt::format("Running hollow cylinder bucking example with {} displacement dofs",
                                    solid_solver->displacement().GlobalSize()));
   SLIC_INFO_ROOT("Starting pseudo-timestepping.");
   smith::logger::flush();
   while (solid_solver->time() < tfinal && std::abs(solid_solver->time() - tfinal) > DBL_EPSILON) {
     SLIC_INFO_ROOT(axom::fmt::format("time = {} (out of 1.0)", solid_solver->time()));
     smith::logger::flush();
     
     solid_solver->advanceTimestep(dt);
 
     // Output the sidre-based plot files
     solid_solver->outputStateToDisk(paraview_name);
   }
   SLIC_INFO_ROOT(axom::fmt::format("final time = {}", solid_solver->time()));
 
   return 0;
 }