// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <cmath>
#include <memory>
#include <set>
#include <string>

#include "axom/CLI11.hpp"
#include "axom/slic.hpp"
#include "mfem.hpp"

#include "shared/mesh/MeshBuilder.hpp"
#include "smith/smith.hpp"
#include "tribol/interface/tribol.hpp"

namespace {

constexpr int ORDER = 1;
constexpr int DIM = 2;

constexpr int FIXED_ATTR = 1;
constexpr int INNER_TOP_ATTR = 2;
constexpr int INNER_LEFT_ATTR = 3;
constexpr int INNER_BOTTOM_ATTR = 4;
constexpr int OUTER_TOP_ATTR = 6;
constexpr int OUTER_BOTTOM_ATTR = 7;
constexpr bool CHECKPOINT_TO_DISK = false;
constexpr bool USE_WARM_START = false;

}  // namespace

int main(int argc, char* argv[])
{
  smith::ApplicationManager application_manager(argc, argv);

  int num_contact_interactions = 3;
  int num_x_elements = 16;
  int num_y_elements = 16;
  int arm_thickness_elements = 3;
  int num_steps = 50;
  double residual_gap = 0.0;
  double penalty = 1.0e3;
  double traction = 0.05;
  double binning_proximity_scale = 10.0;
  double normal_smoothing_start_angle_degrees = 90.0;

  axom::CLI::App app{"C-shaped self-contact example"};
  app.add_option("--contact-interactions", num_contact_interactions,
                 "Number of contact interactions: three pairwise interactions or one all-to-all interaction.")
      ->check(axom::CLI::IsMember({1, 3}));
  app.add_option("--residual-gap", residual_gap,
                 "Residual gap applied to every contact interaction; positive values offset penetration from penalty "
                 "enforcement.")
      ->check(axom::CLI::NonNegativeNumber);
  app.add_option("--num-x-elements", num_x_elements, "Number of elements across the C-shape width.")
      ->check(axom::CLI::PositiveNumber);
  app.add_option("--num-y-elements", num_y_elements, "Number of elements across the C-shape height.")
      ->check(axom::CLI::PositiveNumber);
  app.add_option("--arm-thickness-elements", arm_thickness_elements, "Number of elements across each C-shape arm.")
      ->check(axom::CLI::PositiveNumber);
  app.add_option("--num-steps", num_steps, "Number of quasistatic steps.")->check(axom::CLI::PositiveNumber);
  app.add_option("--penalty", penalty, "EnergyMortar penalty parameter.")->check(axom::CLI::PositiveNumber);
  app.add_option("--traction", traction, "Final inward traction magnitude on the outer top and bottom surfaces.")
      ->check(axom::CLI::NonNegativeNumber);
  app.add_option("--binning-proximity-scale", binning_proximity_scale,
                 "Element-length multiplier used for the Tribol contact search radius.")
      ->check(axom::CLI::Range(2.0, 100.0));
  app.add_option("--normal-smoothing-start-angle", normal_smoothing_start_angle_degrees,
                 "EnergyMortar normal smoothing start angle in degrees; 90 disables normal attenuation.")
      ->check(axom::CLI::Range(0.0, 90.0));
  app.set_help_flag("--help");
  CLI11_PARSE(app, argc, argv);

#ifndef MFEM_USE_STRUMPACK
  SLIC_INFO_ROOT("Contact requires MFEM built with STRUMPACK.");
  return 1;
#endif

  SLIC_ERROR_ROOT_IF(num_x_elements <= arm_thickness_elements,
                     "--num-x-elements must be greater than --arm-thickness-elements.");
  SLIC_ERROR_ROOT_IF(num_y_elements <= 2 * arm_thickness_elements,
                     "--num-y-elements must be greater than twice --arm-thickness-elements.");

  const std::string interaction_name = num_contact_interactions == 1 ? "one_interaction" : "three_interactions";
  const std::string name = "contact_c_shape_" + interaction_name;
  const double normal_smoothing_start_angle = normal_smoothing_start_angle_degrees * std::acos(-1.0) / 180.0;

  SLIC_INFO_ROOT("Running the C-shape example with "
                 << num_contact_interactions << " contact interaction(s), a " << residual_gap
                 << " residual gap, a binning proximity scale of " << binning_proximity_scale
                 << ", and normal smoothing beginning at " << normal_smoothing_start_angle_degrees << " degrees.");

  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, name + "_data");

  auto mesh = std::make_shared<smith::Mesh>(
      shared::MeshBuilder::CShapeMesh(num_x_elements, num_y_elements, arm_thickness_elements), "c_shape_mesh", 0, 0);
  mesh->addDomainOfBoundaryElements("fixed_left", smith::by_attr<DIM>(FIXED_ATTR));
  mesh->addDomainOfBoundaryElements("outer_top", smith::by_attr<DIM>(OUTER_TOP_ATTR));
  mesh->addDomainOfBoundaryElements("outer_bottom", smith::by_attr<DIM>(OUTER_BOTTOM_ATTR));

  smith::LinearSolverOptions linear_options{.linear_solver = smith::LinearSolver::Strumpack, .print_level = 0};
  smith::NonlinearSolverOptions nonlinear_options{.nonlin_solver = smith::NonlinearSolver::TrustRegion,
                                                  .relative_tol = 1.0e-8,
                                                  .absolute_tol = 1.0e-6,
                                                  .max_iterations = 100,
                                                  .max_line_search_iterations = 10,
                                                  .print_level = 1};
  smith::ContactOptions contact_options{.method = smith::ContactMethod::EnergyMortar,
                                        .enforcement = smith::ContactEnforcement::Penalty,
                                        .type = smith::ContactType::Frictionless,
                                        .penalty = penalty,
                                        .jacobian = smith::ContactJacobian::Exact};

  smith::SolidMechanicsContact<ORDER, DIM> solid_solver(
      nonlinear_options, linear_options, smith::solid_mechanics::default_quasistatic_options, name, mesh, {}, 0, 0.0,
      CHECKPOINT_TO_DISK, USE_WARM_START);
  mfem::VisItDataCollection visit_dc(name + "_visit", &mesh->mfemParMesh());
  visit_dc.SetPrefixPath("visit_out");
  visit_dc.RegisterField("displacement", &solid_solver.displacement().gridFunction());

  smith::solid_mechanics::NeoHookean material{1.0, 10.0, 0.25};
  solid_solver.setMaterial(material, mesh->entireBody());
  solid_solver.setFixedBCs(mesh->domain("fixed_left"));
  solid_solver.setTraction(
      [traction, num_steps](auto, auto, double time) {
        return smith::tensor<double, DIM>{0.0, -traction * time / num_steps};
      },
      mesh->domain("outer_top"));
  solid_solver.setTraction(
      [traction, num_steps](auto, auto, double time) {
        return smith::tensor<double, DIM>{0.0, traction * time / num_steps};
      },
      mesh->domain("outer_bottom"));

  auto add_contact_interaction = [&](int interaction_id, const std::set<int>& surface_1,
                                     const std::set<int>& surface_2) {
    solid_solver.addContactInteraction(interaction_id, surface_1, surface_2, contact_options);
    tribol::setBinningProximityScale(interaction_id, binning_proximity_scale);
    tribol::setResidualGap(interaction_id, residual_gap);
    tribol::setEnforcementLocation(interaction_id, tribol::EnforcementLocation::QuadraturePoint);
    tribol::setEnergyMortarNormalSmoothingStartAngle(interaction_id, normal_smoothing_start_angle);
  };

  if (num_contact_interactions == 3) {
    add_contact_interaction(0, {INNER_TOP_ATTR}, {INNER_BOTTOM_ATTR});
    add_contact_interaction(1, {INNER_TOP_ATTR}, {INNER_LEFT_ATTR});
    add_contact_interaction(2, {INNER_BOTTOM_ATTR}, {INNER_LEFT_ATTR});
  } else {
    const std::set<int> inner_surface_attributes{INNER_TOP_ATTR, INNER_LEFT_ATTR, INNER_BOTTOM_ATTR};
    add_contact_interaction(0, inner_surface_attributes, inner_surface_attributes);
  }

  solid_solver.completeSetup();

  const std::string paraview_name = name + "_paraview";
  solid_solver.outputStateToDisk(paraview_name);
  visit_dc.SetCycle(0);
  visit_dc.SetTime(0.0);
  visit_dc.Save();

  constexpr double timestep = 1.0;
  for (int step = 0; step < num_steps; ++step) {
    solid_solver.advanceTimestep(timestep);
    solid_solver.outputStateToDisk(paraview_name);
    visit_dc.SetCycle(step + 1);
    visit_dc.SetTime((step + 1) * timestep);
    visit_dc.Save();
  }

  return 0;
}
