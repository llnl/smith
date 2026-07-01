// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <cmath>
#include <functional>
#include <memory>
#include <set>
#include <sstream>
#include <string>

#include "axom/CLI11.hpp"
#include "axom/sidre/core/MFEMSidreDataCollection.hpp"
#include "axom/slic.hpp"
#include "mfem.hpp"
#include "shared/mesh/MeshBuilder.hpp"

#include "smith/infrastructure/application_manager.hpp"
#include "smith/mesh_utils/mesh_utils.hpp"
#include "smith/numerics/solver_config.hpp"
#include "smith/physics/boundary_conditions/components.hpp"
#include "smith/physics/materials/parameterized_solid_material.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/solid_mechanics_contact.hpp"
#include "smith/physics/state/state_manager.hpp"
#include "smith/smith.hpp"
#include "smith/smith_config.hpp"
#include "tribol/interface/tribol.hpp"
#include "tribol/interface/mfem_tribol.hpp"

namespace {

constexpr int P = 1;
constexpr int DIM = 2;

using MeshPtr = std::shared_ptr<smith::Mesh>;
using DisplacementFunction = std::function<smith::tensor<double, DIM>(smith::tensor<double, DIM>, double)>;

enum class IroningCase
{
  Square,
  Circle,
  Twisted
};

IroningCase parseCase(const std::string& value)
{
  if (value == "square") {
    return IroningCase::Square;
  }
  if (value == "circle") {
    return IroningCase::Circle;
  }
  if (value == "twisted") {
    return IroningCase::Twisted;
  }

  SLIC_ERROR_ROOT("Unknown ironing case '" << value << "'. Expected one of: square, circle, twisted.");
  return IroningCase::Square;
}

std::string caseName(IroningCase ironing_case)
{
  switch (ironing_case) {
    case IroningCase::Square:
      return "square";
    case IroningCase::Circle:
      return "circle";
    case IroningCase::Twisted:
      return "twisted";
  }

  SLIC_ERROR_ROOT("Unsupported ironing case.");
  return "square";
}

std::string formatNameValue(double value)
{
  std::ostringstream os;
  os << value;
  std::string out = os.str();
  for (auto& c : out) {
    if (c == '.') {
      c = 'p';
    } else if (c == '-') {
      c = 'm';
    } else if (c == '+') {
      c = 'p';
    }
  }
  return out;
}

std::string makeRunName(const std::string& case_name, const std::string& selected_normal,
                        const std::string& penalty_mode, bool projection_smoothing, double active_set_smoothing_gap,
                        const std::string& nodal_energy_basis, bool nodal_energy_angle_smoothing)
{
  return "contact_ironing_2D_" + case_name + "_normal_" + selected_normal + "_penalty_" + penalty_mode +
         "_proj_smooth_" + (projection_smoothing ? "on" : "off") +
         "_active_gap_" + formatNameValue(active_set_smoothing_gap) + "_nodal_basis_" + nodal_energy_basis +
         "_angle_smooth_" + (nodal_energy_angle_smoothing ? "on" : "off") + "_example";
}

MeshPtr buildSquareMesh(const std::string& mesh_tag)
{
  constexpr auto mesh_factor = 8;

  auto mesh = shared::MeshBuilder::Unify({shared::MeshBuilder::SquareMesh(8 * mesh_factor, 2 * mesh_factor)
                                              .updateBdrAttrib(1, 6)
                                              .updateBdrAttrib(3, 9)
                                              .updateBdrAttrib(2, 1)
                                              .scale({1.0, 0.25}),
                                          shared::MeshBuilder::SquareMesh(2 * mesh_factor, 1 * mesh_factor)
                                              .scale({0.25, 0.125})
                                              .translate({0.0, 0.25})
                                              .updateBdrAttrib(3, 5)
                                              .updateBdrAttrib(1, 8)
                                              .updateBdrAttrib(4, 2)
                                              .updateAttrib(1, 2)});
  return std::make_shared<smith::Mesh>(mesh, mesh_tag, 0, 0);
}

MeshPtr buildCircleMesh(const std::string& mesh_tag)
{
  constexpr auto mesh_factor = 4;

  auto mesh = shared::MeshBuilder::Unify(
      {shared::MeshBuilder::SquareMesh(32 * mesh_factor, 8 * mesh_factor)
           .updateBdrAttrib(1, 6)
           .updateBdrAttrib(3, 9)
           .scale({1.0, 0.25}),
       shared::MeshBuilder::SemiCircularShell(mesh_factor * 3 / 2, 10 * mesh_factor, 0.075, 0.125)
           .translate({0.125, 0.375})
           .updateBdrAttrib(1, 5)
           .updateBdrAttrib(2, 8)
           .updateBdrAttrib(3, 5)
           .updateBdrAttrib(4, 5)
           .updateAttrib(1, 2)});

  auto out = std::make_shared<smith::Mesh>(mesh, mesh_tag, 0, 0);
  out->mfemParMesh().CheckElementOrientation(true);
  return out;
}

struct CaseConfig {
  std::string name;
  std::string mesh_tag;
  int num_steps;
  smith::NonlinearSolverOptions nonlinear_options;
  MeshPtr mesh;
  DisplacementFunction displacement;
  std::set<int> substrate_contact_attrs;
  std::set<int> indenter_contact_attrs;
  bool add_secondary_contact = false;
  std::set<int> secondary_indenter_contact_attrs;
};

DisplacementFunction slidingDisplacement(double depth)
{
  return [depth](smith::tensor<double, DIM>, double t) {
    constexpr double init_steps = 20.0;
    smith::tensor<double, DIM> u{};
    if (t <= init_steps + 1.0e-12) {
      u[1] = -t * depth / init_steps;
    } else {
      u[0] = (t - init_steps) * 0.005;
      u[1] = -depth;
    }
    return u;
  };
}

DisplacementFunction twistedDisplacement()
{
  const smith::tensor<double, DIM> r0{{0.125, 0.625}};
  return [r0](smith::tensor<double, DIM> x, double t) {
    constexpr double init_steps = 10.0;
    constexpr double theta_max = 80.0 * M_PI / 180.0;
    smith::tensor<double, DIM> u{};
    if (t <= init_steps + 1.0e-12) {
      u[1] = -t * 0.05 / init_steps;
    } else {
      double hm = (t - init_steps) * 0.01;
      double theta = theta_max * hm;
      double cos_theta = std::cos(theta);
      double sin_theta = std::sin(theta);

      smith::tensor<double, DIM> y{{x[0] - r0[0], x[1] - r0[1]}};
      smith::tensor<double, DIM> y_rot{{cos_theta * y[0] - sin_theta * y[1], sin_theta * y[0] + cos_theta * y[1]}};

      u[0] = (y_rot[0] - y[0]) + 0.01 * (t - init_steps);
      u[1] = (y_rot[1] - y[1]) - 0.05;
    }
    return u;
  };
}

CaseConfig makeCaseConfig(IroningCase ironing_case)
{
  auto base_nonlinear_options = smith::NonlinearSolverOptions{.nonlin_solver = smith::NonlinearSolver::TrustRegion,
                                                              .relative_tol = 1.0e-8,
                                                              .absolute_tol = 1.0e-10,
                                                              .max_iterations = 500,
                                                              .max_line_search_iterations = 10,
                                                              .print_level = 1};

  const auto case_name = caseName(ironing_case);
  CaseConfig config;
  config.name = "contact_ironing_2D_" + case_name + "_example";
  config.mesh_tag = "ironing_2D_" + case_name + "_mesh";
  config.num_steps = 175;
  config.nonlinear_options = base_nonlinear_options;

  switch (ironing_case) {
    case IroningCase::Square:
      config.mesh = buildSquareMesh(config.mesh_tag);
      config.displacement = slidingDisplacement(0.1);
      config.substrate_contact_attrs = {9};
      config.indenter_contact_attrs = {8, 2};
      break;
    case IroningCase::Circle:
      config.mesh = buildCircleMesh(config.mesh_tag);
      config.displacement = slidingDisplacement(0.101);
      config.substrate_contact_attrs = {9};
      config.indenter_contact_attrs = {8};
      config.nonlinear_options.absolute_tol = 1.0e-8;
      config.nonlinear_options.max_iterations = 5000;
      break;
    case IroningCase::Twisted:
      config.mesh = buildSquareMesh(config.mesh_tag);
      config.displacement = twistedDisplacement();
      config.substrate_contact_attrs = {9};
      config.indenter_contact_attrs = {8};
      config.num_steps = 110;
      break;
  }

  return config;
}

}  // namespace

int main(int argc, char* argv[])
{
  smith::ApplicationManager applicationManager(argc, argv);

  std::string selected_case = "square";
  std::string selected_normal = "element";
  std::string penalty_mode = "nodal";
  std::string nodal_energy_basis = "cubic-spline";
  bool projection_smoothing = true;
  bool nodal_energy_angle_smoothing = true;
  double active_set_smoothing_gap = 0.001;
  int num_steps_override = -1;
  int nonlinear_print_level = -1;
  axom::CLI::App app{"2D contact ironing example"};
  app.add_option("--case", selected_case, "Ironing case: square, circle, or twisted")
      ->check(axom::CLI::IsMember({"square", "circle", "twisted"}));
  app.add_option("--energy-mortar-normal", selected_normal, "Energy mortar normal field: element or averaged")
      ->check(axom::CLI::IsMember({"element", "averaged"}));
  app.add_option("--energy-mortar-penalty-mode", penalty_mode,
                 "Energy mortar penalty enforcement mode: nodal, quadrature-point, or nodal-energy")
      ->check(axom::CLI::IsMember({"nodal", "quadrature-point", "nodal-energy"}));
  app.add_option("--energy-mortar-nodal-energy-basis", nodal_energy_basis,
                 "Nodal energy basis: fe or cubic-spline")
      ->check(axom::CLI::IsMember({"fe", "cubic-spline"}));
  app.add_flag("--energy-mortar-projection-smoothing,!--no-energy-mortar-projection-smoothing", projection_smoothing,
               "Use projection-bound smoothing in the energy mortar contact calculation");
  app.add_flag("--energy-mortar-nodal-energy-angle-smoothing,!--no-energy-mortar-nodal-energy-angle-smoothing",
               nodal_energy_angle_smoothing, "Use 80-to-90 degree angle smoothing in nodal energy mode");
  app.add_option("--energy-mortar-active-set-smoothing-gap", active_set_smoothing_gap,
                 "Energy mortar active-set smoothing transition gap; disabled when <= 0")
      ->check(axom::CLI::NonNegativeNumber);
  app.add_option("--num-steps", num_steps_override, "Override the number of time steps")
      ->check(axom::CLI::NonNegativeNumber);
  app.add_option("--nonlinear-print-level", nonlinear_print_level, "Override the nonlinear solver print level")
      ->check(axom::CLI::Range(0, 2));
  app.set_help_flag("--help");
  CLI11_PARSE(app, argc, argv);

#ifndef MFEM_USE_STRUMPACK
  SLIC_INFO_ROOT("Contact requires MFEM built with strumpack.");
  return 1;
#endif

  const auto ironing_case = parseCase(selected_case);
  const std::string case_name = caseName(ironing_case);
  const std::string run_name =
      makeRunName(case_name, selected_normal, penalty_mode, projection_smoothing, active_set_smoothing_gap,
                  nodal_energy_basis, nodal_energy_angle_smoothing);
  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, run_name + "_data");

  CaseConfig config = makeCaseConfig(ironing_case);
  if (num_steps_override >= 0) {
    config.num_steps = num_steps_override;
  }
  if (nonlinear_print_level >= 0) {
    config.nonlinear_options.print_level = nonlinear_print_level;
  }
  config.name = run_name;
  auto& mesh = config.mesh;

  smith::LinearSolverOptions linear_options{
      .linear_solver = smith::LinearSolver::CG, .preconditioner = smith::Preconditioner::HypreAMG, .print_level = 0};

  mfem::VisItDataCollection visit_dc(config.name + "_visit", &mesh->mfemParMesh());
  visit_dc.SetPrefixPath("visit_out");
  visit_dc.Save();

  smith::ContactOptions contact_options{.method = smith::ContactMethod::EnergyMortar,
                                        .enforcement = smith::ContactEnforcement::Penalty,
                                        .type = smith::ContactType::Frictionless,
                                        .penalty = 30000.0,
                                        .penalty2 = 0.0,
                                        .jacobian = smith::ContactJacobian::Exact};

  smith::SolidMechanicsContact<P, DIM, smith::Parameters<smith::L2<0>, smith::L2<0>>> solid_solver(
      config.nonlinear_options, linear_options, smith::solid_mechanics::default_quasistatic_options, config.name, mesh,
      {"bulk_mod", "shear_mod"}, 0, 0.0, false, false);

  smith::FiniteElementState K_field(smith::StateManager::newState(smith::L2<0>{}, "bulk_mod", mesh->tag()));

  mfem::Vector K_values({1.0, 100.0});
  mfem::PWConstCoefficient K_coeff(K_values);
  K_field.project(K_coeff);
  solid_solver.setParameter(0, K_field);

  smith::FiniteElementState G_field(smith::StateManager::newState(smith::L2<0>{}, "shear_mod", mesh->tag()));

  mfem::Vector G_values({0.25, 25.0});
  mfem::PWConstCoefficient G_coeff(G_values);
  G_field.project(G_coeff);
  solid_solver.setParameter(1, G_field);

  smith::solid_mechanics::ParameterizedNeoHookeanSolid mat{1.0, 100.0, 1.0};
  solid_solver.setMaterial(smith::DependsOn<0, 1>{}, mat, mesh->entireBody());

  mesh->addDomainOfBoundaryElements("bottom_of_substrate", smith::by_attr<DIM>(6));
  solid_solver.setFixedBCs(mesh->domain("bottom_of_substrate"));

  mesh->addDomainOfBoundaryElements("top_of_indenter", smith::by_attr<DIM>(5));
  solid_solver.setDisplacementBCs(config.displacement, mesh->domain("top_of_indenter"));

  solid_solver.addContactInteraction(0, config.substrate_contact_attrs, config.indenter_contact_attrs, contact_options);
  tribol::setEnergyMortarProjectionSmoothing(0, projection_smoothing);
  tribol::setEnergyMortarH1ActiveSetSmoothing(0, active_set_smoothing_gap);
  tribol::setEnergyMortarPenaltyMode(0, penalty_mode == "nodal-energy"
                                            ? tribol::EnergyMortarPenaltyMode::NODAL_ENERGY
                                            : penalty_mode == "quadrature-point"
                                                  ? tribol::EnergyMortarPenaltyMode::QUADRATURE_POINT_GAP
                                                  : tribol::EnergyMortarPenaltyMode::NODAL_GAP);
  tribol::setEnergyMortarNodalEnergyBasis(0, nodal_energy_basis == "fe"
                                                 ? tribol::EnergyMortarNodalEnergyBasis::FE
                                                 : tribol::EnergyMortarNodalEnergyBasis::CUBIC_SPLINE);
  tribol::setEnergyMortarNodalEnergyAngleSmoothing(0, nodal_energy_angle_smoothing);
  if (selected_normal == "averaged") {
    tribol::setEnergyMortarNormalMode(0, tribol::EnergyMortarNormalMode::H1_NODAL_NORMAL);
  }
  if (config.add_secondary_contact) {
    solid_solver.addContactInteraction(1, config.substrate_contact_attrs, config.secondary_indenter_contact_attrs,
                                       contact_options);
    tribol::setEnergyMortarProjectionSmoothing(1, projection_smoothing);
    tribol::setEnergyMortarH1ActiveSetSmoothing(1, active_set_smoothing_gap);
    tribol::setEnergyMortarPenaltyMode(1, penalty_mode == "nodal-energy"
                                              ? tribol::EnergyMortarPenaltyMode::NODAL_ENERGY
                                              : penalty_mode == "quadrature-point"
                                                    ? tribol::EnergyMortarPenaltyMode::QUADRATURE_POINT_GAP
                                                    : tribol::EnergyMortarPenaltyMode::NODAL_GAP);
    tribol::setEnergyMortarNodalEnergyBasis(1, nodal_energy_basis == "fe"
                                                   ? tribol::EnergyMortarNodalEnergyBasis::FE
                                                   : tribol::EnergyMortarNodalEnergyBasis::CUBIC_SPLINE);
    tribol::setEnergyMortarNodalEnergyAngleSmoothing(1, nodal_energy_angle_smoothing);
    if (selected_normal == "averaged") {
      tribol::setEnergyMortarNormalMode(1, tribol::EnergyMortarNormalMode::H1_NODAL_NORMAL);
    }
  }

  const std::string paraview_name = config.name + "_paraview";
  solid_solver.outputStateToDisk(paraview_name);

  solid_solver.completeSetup();

  std::unique_ptr<axom::sidre::DataStore> submesh_normal_datastore;
  std::unique_ptr<axom::sidre::MFEMSidreDataCollection> submesh_normal_dc;
  auto save_submesh_normals = [&](int cycle, double time) {
    if (selected_normal != "averaged") {
      return;
    }

    const std::string coll_name = config.name + "_submesh_normals";
    auto& nodal_normal = tribol::getMfemEnergyMortarNodalNormal(0);
    if (!submesh_normal_dc) {
      submesh_normal_datastore = std::make_unique<axom::sidre::DataStore>();
      auto* global_group = submesh_normal_datastore->getRoot()->createGroup(coll_name + "_global");
      auto* bp_index_group = global_group->createGroup("blueprint_index/" + coll_name);
      auto* domain_group = submesh_normal_datastore->getRoot()->createGroup(coll_name);

      submesh_normal_dc =
          std::make_unique<axom::sidre::MFEMSidreDataCollection>(coll_name, bp_index_group, domain_group, true);
      auto& contact_submesh = tribol::getMfemSubmesh(0);
      submesh_normal_dc->SetComm(contact_submesh.GetComm());
      submesh_normal_dc->SetPrefixPath(config.name + "_submesh_normals_data");
      submesh_normal_dc->SetMesh(&contact_submesh);
      submesh_normal_dc->SetOwnData(false);
      submesh_normal_dc->RegisterField("energy_mortar_nodal_normal", &nodal_normal);
    }
    nodal_normal.HostRead();
    submesh_normal_dc->SetCycle(cycle);
    submesh_normal_dc->SetTime(time);
    submesh_normal_dc->Save();
  };

  constexpr double dt = 1.0;
  for (int i{0}; i < config.num_steps; ++i) {
    solid_solver.advanceTimestep(dt);
    visit_dc.SetCycle(i);
    visit_dc.SetTime((i + 1) * dt);
    visit_dc.Save();

    solid_solver.outputStateToDisk(paraview_name);
    save_submesh_normals(i + 1, (i + 1) * dt);
  }

  return 0;
}
