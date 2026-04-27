
//************DISPLACEMENT TEST********************* */
 
// #include <cfenv>
// #include <functional>
// #include <set>
// #include <string>
 
// #include "axom/slic.hpp"
// #include "axom/slic/core/SimpleLogger.hpp"
// #include "mfem.hpp"
 
// #include <mesh/vtk.hpp>
 
// #include "smith/infrastructure/application_manager.hpp"
// #include "smith/mesh_utils/mesh_utils.hpp"
// #include "smith/numerics/solver_config.hpp"
// #include "smith/physics/boundary_conditions/components.hpp"
// #include "smith/physics/contact/contact_config.hpp"
// #include "smith/physics/materials/parameterized_solid_material.hpp"
// #include "smith/physics/mesh.hpp"
// #include "smith/physics/solid_mechanics.hpp"
// #include "smith/physics/solid_mechanics_contact.hpp"
// #include "smith/physics/state/state_manager.hpp"
// #include "smith/physics/materials/solid_material.hpp"
// #include "smith/smith.hpp"
// #include "smith/smith_config.hpp"
 
// int main(int argc, char* argv[])
// {
//   smith::ApplicationManager applicationManager(argc, argv);
 
//   // NOTE: polynomial degree p = 1 required for Tribol mortar method
//   constexpr int p   = 1;
//   // NOTE: dim = 2 (plane-strain Hertzian contact)
//   constexpr int dim = 2;
 
//   const std::string name = "contact_hertzian_2D";
//   axom::sidre::DataStore datastore;
//   smith::StateManager::initialize(datastore, name + "_data");

//   const std::string mesh_file = (argc > 1)
//       ? std::string(argv[1])
//       : std::string("../../data/meshes/hertzian_contact.msh");
 
//   auto mesh = std::make_shared<smith::Mesh>(mesh_file, "hertzian_2d_mesh", 0, 0);
//   mesh->mfemParMesh().CheckElementOrientation(true);
 
//   // ── Solver options ────────────────────────────────────────────────────
//   smith::LinearSolverOptions linear_options{
//       .linear_solver  = smith::LinearSolver::CG,
//       .preconditioner = smith::Preconditioner::HypreAMG,
//       .print_level    = 0};
 
//   smith::NonlinearSolverOptions nonlinear_options{
//       .nonlin_solver             = smith::NonlinearSolver::TrustRegion,
//       .relative_tol              = 1.0e-8,
//       .absolute_tol              = 1.0e-8,
//       .max_iterations            = 5000,
//       .max_line_search_iterations = 10,
//       .print_level               = 1};
 
//   smith::ContactOptions contact_options{
//       .method      = smith::ContactMethod::EnergyMortar,
//       .enforcement = smith::ContactEnforcement::Penalty,
//       .type        = smith::ContactType::Frictionless,
//       .penalty     = 30000.0,
//       .penalty2    = 0,
//       .jacobian    = smith::ContactJacobian::Exact};
 
// #ifndef MFEM_USE_STRUMPACK
//   SLIC_INFO_ROOT("Contact requires MFEM built with strumpack.");
//   return 1;
// #endif
 
//   // ── Solid mechanics solver ────────────────────────────────────────────
//   smith::SolidMechanicsContact<p, dim, smith::Parameters<smith::L2<0>, smith::L2<0>>>
//       solid_solver(nonlinear_options, linear_options,
//                    smith::solid_mechanics::default_quasistatic_options,
//                    name, mesh, {"bulk_mod", "shear_mod"}, 0, 0.0,
//                    /*is_dynamic=*/false, /*geometric_nonlinearity=*/false);
 
//   smith::FiniteElementState K_field(
//       smith::StateManager::newState(smith::L2<0>{}, "bulk_mod", mesh->tag()));
//   mfem::Vector K_values({1000.0, 1.0});   // [attr_1=block, attr_2=indenter]
//   mfem::PWConstCoefficient K_coeff(K_values);
//   K_field.project(K_coeff);
//   solid_solver.setParameter(0, K_field);
 
//   smith::FiniteElementState G_field(
//       smith::StateManager::newState(smith::L2<0>{}, "shear_mod", mesh->tag()));
//   mfem::Vector G_values({250.0, 0.25});   // [attr_1=block, attr_2=indenter]
//   mfem::PWConstCoefficient G_coeff(G_values);
//   G_field.project(G_coeff);
//   solid_solver.setParameter(1, G_field);
 
//   smith::solid_mechanics::ParameterizedNeoHookeanSolid mat{1.0, 100.0, 1.0};
//   solid_solver.setMaterial(smith::DependsOn<0, 1>{}, mat, mesh->entireBody());
 
//   // ── Boundary conditions ────────────────────────────────────────────────
 
//   // 1. Fixed bottom of block (attr 4) — fully clamped
//   mesh->addDomainOfBoundaryElements("block_bottom", smith::by_attr<dim>(4));
//   solid_solver.setFixedBCs(mesh->domain("block_bottom"));
 
//   // 2. Roller BCs on block sides (attr 5) — fix x only, free y.
//   //    Implemented as a zero-x displacement BC so the block can compress
//   //    vertically without lateral drift.
//   auto zero_x_displacement = [](smith::tensor<double, dim> /*x*/, double /*t*/)
//   {
//     smith::tensor<double, dim> u{};
//     u[0] = 0.0;
//     return u;
//   };
//   mesh->addDomainOfBoundaryElements("block_sides", smith::by_attr<dim>(5));
//   solid_solver.setDisplacementBCs(zero_x_displacement,
//                                   mesh->domain("block_sides"));
 

//   constexpr double total_steps     = 300.0;
//   constexpr double max_indentation = 0.35;   
 
//   auto applied_displacement = [](smith::tensor<double, dim> /*x*/, double t)
//   {
//     smith::tensor<double, dim> u{};
//     u[1] = -t * max_indentation / total_steps;   // pure vertical ramp
//     return u;
//   };
 
//   mesh->addDomainOfBoundaryElements("indenter_top", smith::by_attr<dim>(1));
//   solid_solver.setDisplacementBCs(applied_displacement,
//                                   mesh->domain("indenter_top"));
 

//   const auto contact_id = 0;
//   std::set<int> master_attrs({3});   // block_top
//   std::set<int> slave_attrs ({2});   // indenter_arc
 
//   solid_solver.addContactInteraction(contact_id, master_attrs, slave_attrs,
//                                      contact_options);
 
//   // ── Output setup ──────────────────────────────────────────────────────
//   const std::string visit_name = name + "_visit";
//   solid_solver.outputStateToDisk(visit_name);
 
//   // ── Complete setup and time-march ─────────────────────────────────────
//   solid_solver.completeSetup();
 
//   if (std::isnan(solid_solver.displacement().Norml2()))
//   {
//     SLIC_ERROR_ROOT("NaN in displacement before first timestep!");
//     return 1;
//   }
 
//   const double dt = 1.0;
//   const int    n_steps = static_cast<int>(total_steps);
 
//   for (int i = 0; i < n_steps; ++i)
//   {
//     solid_solver.advanceTimestep(dt);
 
//     solid_solver.outputStateToDisk(visit_name);
//   }
 
//   return 0;
// }








//*******************TRACTION TEST ******************** */

// Copyright (c) Lawrence Livermore National Security, LLC and
// other smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
 
/**
 * @file contact_hertzian_2d.cpp
 *
 * 2-D Hertzian contact test for the smoothed mortar contact formulation.
 *
 * Geometry
 * ------------------------------------
 *   Block    : rectangle [-1, 1] × [0, 0.5]  (domain attr 1)
 *              top  face  (bdr attr 3) — contact MASTER surface
 *              bottom     (bdr attr 4) — fully fixed (Dirichlet)
 *              sides      (bdr attr 5) — fixed in x (roller)
 *   Indenter : solid half-disk, radius R = 0.5  (domain attr 2)
 *              arc tip at y = 0.5, flat face at y = 1.0
 *              flat face  (bdr attr 1) — displacement BC (load)
 *              arc face   (bdr attr 2) — contact SLAVE surface
 *
 * Loading
 * --------
 *   The flat top of the indenter is driven straight down in displacement
 *   control over `total_steps` pseudo-time steps.  No lateral motion is
 *   applied, so the problem remains symmetric and the Hertz pressure
 *   distribution can be extracted for validation.
 *
 * Expected validation
 * --------------------
 *   For two identical neo-Hookean bodies (small strain regime), the
 *   contact half-width a ≈ sqrt(2*R*δ) and peak pressure
 *   p0 = (2/π) * E_eff * a / R, where E_eff = E / (2*(1-ν²)) for
 *   plane-strain.  Compare the nodal contact traction distribution against
 *   this analytic profile to verify the smoothed mortar implementation.
 */
 
#include <cfenv>
#include <functional>
#include <set>
#include <string>
 
#include "axom/slic.hpp"
#include "axom/slic/core/SimpleLogger.hpp"
#include "mfem.hpp"
 
#include <mesh/vtk.hpp>
 
#include "smith/infrastructure/application_manager.hpp"
#include "smith/mesh_utils/mesh_utils.hpp"
#include "smith/numerics/solver_config.hpp"
#include "smith/physics/boundary_conditions/components.hpp"
#include "smith/physics/contact/contact_config.hpp"
#include "smith/physics/materials/parameterized_solid_material.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/solid_mechanics.hpp"
#include "smith/physics/solid_mechanics_contact.hpp"
#include "smith/physics/state/state_manager.hpp"
#include "smith/physics/materials/solid_material.hpp"
#include "smith/smith.hpp"
#include "smith/smith_config.hpp"
 
int main(int argc, char* argv[])
{
  smith::ApplicationManager applicationManager(argc, argv);
 
  // NOTE: polynomial degree p = 1 required for Tribol mortar method
  constexpr int p   = 1;
  // NOTE: dim = 2 (plane-strain Hertzian contact)
  constexpr int dim = 2;
 
  // ── Mesh ──────────────────────────────────────────────────────────────
  // ── Data store — must be initialized before smith::Mesh is constructed ──
  const std::string name = "contact_hertzian_2D";
  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, name + "_data");
 
  // hertzian_contact.msh lives in data/meshes/ at the repo root.
  // The binary runs from the build/examples/ directory, so the relative
  // path is two levels up.  Override with an absolute path as argv[1]:
  //   ./contact_contact_hertzian_2d /abs/path/to/hertzian_contact.msh
  //
  // Boundary attributes (set in the gmsh script)
  //   1 -> indenter_top  (flat face,   load BC)
  //   2 -> indenter_arc  (curved face, contact SLAVE)
  //   3 -> block_top     (flat face,   contact MASTER)
  //   4 -> block_bottom  (fully fixed)
  //   5 -> block_sides   (roller: fixed in x)
  //
  // Domain attributes
  //   1 -> block body
  //   2 -> indenter body
  const std::string mesh_file = (argc > 1)
      ? std::string(argv[1])
      : std::string("../../data/meshes/hertzian_contact.msh");
 
  auto mesh = std::make_shared<smith::Mesh>(mesh_file, "hertzian_2d_mesh", 0, 0);
  mesh->mfemParMesh().CheckElementOrientation(true);
 
  // ── Solver options ────────────────────────────────────────────────────
  smith::LinearSolverOptions linear_options{
      .linear_solver  = smith::LinearSolver::CG,
      .preconditioner = smith::Preconditioner::HypreAMG,
      .print_level    = 0};
 
  smith::NonlinearSolverOptions nonlinear_options{
      .nonlin_solver             = smith::NonlinearSolver::TrustRegion,
      .relative_tol              = 1.0e-8,
      .absolute_tol              = 1.0e-8,
      .max_iterations            = 5000,
      .max_line_search_iterations = 10,
      .print_level               = 1};
 
  // ── Contact options ───────────────────────────────────────────────────
  // Mirror the ironing test exactly; tune penalty after first run.
  // Rule of thumb: penalty ≈ E * h_elem^{-1} * factor, where factor ~ 1–10.
  // With E ~ 100 and h_elem ~ 0.025 in the contact zone, penalty ~ 30 000 is
  // a reasonable starting point.
  smith::ContactOptions contact_options{
      .method      = smith::ContactMethod::EnergyMortar,
      .enforcement = smith::ContactEnforcement::Penalty,
      .type        = smith::ContactType::Frictionless,
      .penalty     = 30000.0,
      .penalty2    = 0,
      .jacobian    = smith::ContactJacobian::Exact};
 
#ifndef MFEM_USE_STRUMPACK
  SLIC_INFO_ROOT("Contact requires MFEM built with strumpack.");
  return 1;
#endif
 
  // ── Solid mechanics solver ────────────────────────────────────────────
  smith::SolidMechanicsContact<p, dim, smith::Parameters<smith::L2<0>, smith::L2<0>>>
      solid_solver(nonlinear_options, linear_options,
                   smith::solid_mechanics::default_quasistatic_options,
                   name, mesh, {"bulk_mod", "shear_mod"}, 0, 0.0,
                   /*is_dynamic=*/false, /*geometric_nonlinearity=*/false);
 
  // ── Material parameters ───────────────────────────────────────────────
  // Domain attr 1 = block  (substrate, softer)
  // Domain attr 2 = indenter (stiffer)
  //
  // NeoHookean parameters:  K = bulk modulus,  G = shear modulus
  // Plane-strain Young's modulus:  E = 9KG / (3K + G)
  // Poisson's ratio:               ν = (3K - 2G) / (2*(3K + G))
  //
  // Block    : K=1.0,  G=0.25  → E ≈  0.82,  ν ≈ 0.33  (compliant)
  // Indenter : K=100,  G=25    → E ≈ 82,    ν ≈ 0.33  (rigid-ish)
  // For a "same material" Hertz test set both to the same values.
  smith::FiniteElementState K_field(
      smith::StateManager::newState(smith::L2<0>{}, "bulk_mod", mesh->tag()));
  mfem::Vector K_values({100.0, 1.0});   // [attr_1=block (stiff), attr_2=indenter (soft)]
  mfem::PWConstCoefficient K_coeff(K_values);
  K_field.project(K_coeff);
  solid_solver.setParameter(0, K_field);
 
  smith::FiniteElementState G_field(
      smith::StateManager::newState(smith::L2<0>{}, "shear_mod", mesh->tag()));
  mfem::Vector G_values({25.0, 0.25});   // [attr_1=block (stiff), attr_2=indenter (soft)]
  mfem::PWConstCoefficient G_coeff(G_values);
  G_field.project(G_coeff);
  solid_solver.setParameter(1, G_field);
 
  smith::solid_mechanics::ParameterizedNeoHookeanSolid mat{1.0, 100.0, 1.0};
  solid_solver.setMaterial(smith::DependsOn<0, 1>{}, mat, mesh->entireBody());
 
  // ── Boundary conditions ────────────────────────────────────────────────
 
  // 1. Fixed bottom of block (attr 4) — fully clamped
  mesh->addDomainOfBoundaryElements("block_bottom", smith::by_attr<dim>(4));
  solid_solver.setFixedBCs(mesh->domain("block_bottom"));
 
  // 2. Roller BCs on block sides (attr 5) — fix x only, free y.
  //    Implemented as a zero-x displacement BC so the block can compress
  //    vertically without lateral drift.
  auto zero_x_displacement = [](smith::tensor<double, dim> /*x*/, double /*t*/)
  {
    smith::tensor<double, dim> u{};
    u[0] = 0.0;
    return u;
  };
  mesh->addDomainOfBoundaryElements("block_sides", smith::by_attr<dim>(5));
  solid_solver.setDisplacementBCs(zero_x_displacement,
                                  mesh->domain("block_sides"));
 
  // 3. Displacement-controlled loading on indenter flat top (attr 1)
  //    Ramp straight down; no lateral motion (symmetric Hertz).
  //
  //    total_steps pseudo-time steps, each of size dt = 1.
  //    Traction ramps from 0 to max_pressure over total_steps steps.
  constexpr double total_steps = 300.0;
  // constexpr double max_pressure = 50.0;  // peak pressure magnitude — tune as needed
 
  // Applied traction on the flat top of the indenter (bdr attr 1).
  // addBoundaryFlux lambda signature: (x, n, t) → traction vector.
  // The outward normal on this face points in +y, so -P*n gives a
  // downward compressive load that ramps linearly in pseudo-time.
  mesh->addDomainOfBoundaryElements("indenter_top", smith::by_attr<dim>(1));
  solid_solver.setTraction(
      [](auto /*x*/, auto n, double t)
      {
        constexpr double max_p = 50.0;
        constexpr double steps = 300.0;
        return -(t * max_p / steps) * n;
      },
      mesh->domain("indenter_top"));

 
  // ── Contact interaction ────────────────────────────────────────────────
  // surface_1 = MASTER  → block top    (boundary attr 3, flat surface)
  // surface_2 = SLAVE   → indenter arc (boundary attr 2, smoothed curved surface)
  //
  // The slave surface is the one where the smoothed mortar integration bounds
  // are parameterically ramped — the whole point of this test.
  const auto contact_id = 0;
  std::set<int> master_attrs({3});   // block_top
  std::set<int> slave_attrs ({2});   // indenter_arc
 
  solid_solver.addContactInteraction(contact_id, master_attrs, slave_attrs,
                                     contact_options);
 
  // ── Output setup ──────────────────────────────────────────────────────
  const std::string visit_name = name + "_visit";
  solid_solver.outputStateToDisk(visit_name);
 
  // ── Complete setup and time-march ─────────────────────────────────────
  solid_solver.completeSetup();
 
  if (std::isnan(solid_solver.displacement().Norml2()))
  {
    SLIC_ERROR_ROOT("NaN in displacement before first timestep!");
    return 1;
  }
 
  const double dt = 1.0;
  const int    n_steps = static_cast<int>(total_steps);
 
  for (int i = 0; i < n_steps; ++i)
  {
    solid_solver.advanceTimestep(dt);
 
    solid_solver.outputStateToDisk(visit_name);
  }
 
  return 0;
}