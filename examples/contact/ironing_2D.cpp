// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <memory>
#include <set>
#include <string>
#include <vector>

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

std::string penaltyModeTag(const std::string& penalty_mode)
{
  if (penalty_mode == "quadrature-point") {
    return "qp";
  }
  if (penalty_mode == "nodal-energy") {
    return "ne";
  }
  return "ng";
}

std::string normalTag(const std::string& selected_normal)
{
  if (selected_normal == "averaged") {
    return "h1";
  }
  return "el";
}

std::string nonlinearSolverTag(const std::string& nonlinear_solver)
{
  if (nonlinear_solver == "newton-line-search") {
    return "nls";
  }
  return "tr";
}

std::string projectionSmoothingCurveTag(const std::string& projection_smoothing_curve)
{
  if (projection_smoothing_curve == "quadratic") {
    return "q2";
  }
  return "q5";
}

smith::NonlinearSolver parseNonlinearSolver(const std::string& value)
{
  if (value == "trust-region") {
    return smith::NonlinearSolver::TrustRegion;
  }
  if (value == "newton-line-search") {
    return smith::NonlinearSolver::NewtonLineSearch;
  }

  SLIC_ERROR_ROOT("Unknown nonlinear solver '" << value << "'. Expected one of: trust-region, newton-line-search.");
  return smith::NonlinearSolver::TrustRegion;
}

tribol::EnergyMortarProjectionSmoothingCurve parseProjectionSmoothingCurve(const std::string& value)
{
  if (value == "quadratic") {
    return tribol::EnergyMortarProjectionSmoothingCurve::QUADRATIC;
  }
  if (value == "quintic") {
    return tribol::EnergyMortarProjectionSmoothingCurve::QUINTIC;
  }

  SLIC_ERROR_ROOT("Unknown projection smoothing curve '" << value << "'. Expected one of: quadratic, quintic.");
  return tribol::EnergyMortarProjectionSmoothingCurve::QUINTIC;
}

double smoothstep(double t)
{
  if (t <= 0.0) {
    return 0.0;
  }
  if (t >= 1.0) {
    return 1.0;
  }
  return t * t * (3.0 - 2.0 * t);
}

double fullPathWeightFromResidual(double residual_norm, double min_residual, double max_residual)
{
  const double t = (residual_norm - min_residual) / (max_residual - min_residual);
  return 1.0 - smoothstep(t);
}

std::string makeRunName(const std::string& case_name, const std::string& selected_normal,
                        const std::string& penalty_mode, bool enzyme_quadrature, bool projection_smoothing,
                        const std::string& projection_smoothing_curve, bool fixed_integration_jacobian,
                        bool qp_frozen_integration, bool qp_derivative_blend_adaptive_gap,
                        const std::string& nonlinear_solver)
{
  return "ci2d_" + case_name + "_p" + penaltyModeTag(penalty_mode) + "_n" + normalTag(selected_normal) + "_s" +
         nonlinearSolverTag(nonlinear_solver) + "_eq" + (enzyme_quadrature ? "1" : "0") + "_ps" +
         (projection_smoothing ? "1" : "0") + projectionSmoothingCurveTag(projection_smoothing_curve) + "_fj" +
         (fixed_integration_jacobian ? "1" : "0") + "_fq" + (qp_frozen_integration ? "1" : "0") + "_ag" +
         (qp_derivative_blend_adaptive_gap ? "1" : "0");
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
                                              .updateBdrAttrib(4, 10)
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

struct CycleRunStats {
  int cycle = 0;
  double time = 0.0;
  smith::NonlinearSolveStats nonlinear;
  tribol::EnergyMortarQpDiagnostics contact0;
  tribol::EnergyMortarQpDiagnostics contact1;
  bool has_secondary_contact = false;
};

std::string jsonString(const std::string& value)
{
  std::string out = "\"";
  for (const char c : value) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        out += c;
        break;
    }
  }
  out += "\"";
  return out;
}

void writeNonlinearSolveStatsJson(std::ostream& os, const smith::NonlinearSolveStats& stats, const std::string& indent)
{
  os << indent << "{\n";
  os << indent << "  \"solver\": " << jsonString(smith::nonlinearName(stats.nonlin_solver)) << ",\n";
  os << indent << "  \"converged\": " << (stats.converged ? "true" : "false") << ",\n";
  os << indent << "  \"iterations\": " << stats.iterations << ",\n";
  os << indent << "  \"initial_residual_norm\": " << stats.initial_residual_norm << ",\n";
  os << indent << "  \"final_residual_norm\": " << stats.final_residual_norm << ",\n";
  os << indent << "  \"final_residual_goal\": " << stats.final_residual_goal << ",\n";
  os << indent << "  \"linear_solves\": " << stats.linear_solves << ",\n";
  os << indent << "  \"line_search_cutbacks\": " << stats.line_search_cutbacks << ",\n";
  os << indent << "  \"accepted_steps\": " << stats.accepted_steps << ",\n";
  os << indent << "  \"rejected_steps\": " << stats.rejected_steps << ",\n";
  os << indent << "  \"min_step_scale\": " << stats.min_step_scale << ",\n";
  os << indent << "  \"final_step_scale\": " << stats.final_step_scale << ",\n";
  os << indent << "  \"trust_region_cg_iterations\": " << stats.trust_region_cg_iterations << ",\n";
  os << indent << "  \"max_trust_region_cg_iterations\": " << stats.max_trust_region_cg_iterations << ",\n";
  os << indent << "  \"trust_region_trial_steps\": " << stats.trust_region_trial_steps << ",\n";
  os << indent << "  \"initial_trust_region_size\": " << stats.initial_trust_region_size << ",\n";
  os << indent << "  \"min_trust_region_size\": " << stats.min_trust_region_size << ",\n";
  os << indent << "  \"final_trust_region_size\": " << stats.final_trust_region_size << ",\n";
  os << indent << "  \"residual_evaluations\": " << stats.residual_evaluations << ",\n";
  os << indent << "  \"jacobian_assemblies\": " << stats.jacobian_assemblies << ",\n";
  os << indent << "  \"hessian_vector_products\": " << stats.hessian_vector_products << ",\n";
  os << indent << "  \"preconditioner_applications\": " << stats.preconditioner_applications << ",\n";
  os << indent << "  \"subspace_solves\": " << stats.subspace_solves << "\n";
  os << indent << "}";
}

void writeQpDiagnosticsJson(std::ostream& os, const tribol::EnergyMortarQpDiagnostics& diagnostics,
                            const std::string& indent)
{
  os << indent << "{\n";
  os << indent << "  \"energy\": " << diagnostics.energy << ",\n";
  os << indent << "  \"residual_gap_average\": " << diagnostics.residual_gap_average << ",\n";
  os << indent << "  \"residual_gap_min\": " << diagnostics.residual_gap_min << ",\n";
  os << indent << "  \"residual_gap_max\": " << diagnostics.residual_gap_max << ",\n";
  os << indent << "  \"blend_weight_average\": " << diagnostics.blend_weight_average << ",\n";
  os << indent << "  \"blend_weight_min\": " << diagnostics.blend_weight_min << ",\n";
  os << indent << "  \"blend_weight_max\": " << diagnostics.blend_weight_max << ",\n";
  os << indent
     << "  \"full_simplified_energy_difference_average\": " << diagnostics.full_simplified_energy_difference_average
     << ",\n";
  os << indent << "  \"full_simplified_energy_difference_max\": " << diagnostics.full_simplified_energy_difference_max
     << ",\n";
  os << indent << "  \"active_pair_count\": " << diagnostics.active_pair_count << ",\n";
  os << indent << "  \"contributing_pair_count\": " << diagnostics.contributing_pair_count << ",\n";
  os << indent << "  \"full_weight_pair_count\": " << diagnostics.full_weight_pair_count << ",\n";
  os << indent << "  \"blended_pair_count\": " << diagnostics.blended_pair_count << ",\n";
  os << indent << "  \"simplified_pair_count\": " << diagnostics.simplified_pair_count << ",\n";
  os << indent << "  \"missing_frozen_integration_pair_count\": " << diagnostics.missing_frozen_integration_pair_count
     << "\n";
  os << indent << "}";
}

void writeCycleRunStatsJson(std::ostream& os, const CycleRunStats& cycle_stats, const std::string& indent)
{
  os << indent << "{\n";
  os << indent << "  \"cycle\": " << cycle_stats.cycle << ",\n";
  os << indent << "  \"time\": " << cycle_stats.time << ",\n";
  os << indent << "  \"nonlinear\": ";
  writeNonlinearSolveStatsJson(os, cycle_stats.nonlinear, indent + "  ");
  os << ",\n" << indent << "  \"contact\": [\n";
  writeQpDiagnosticsJson(os, cycle_stats.contact0, indent + "    ");
  if (cycle_stats.has_secondary_contact) {
    os << ",\n";
    writeQpDiagnosticsJson(os, cycle_stats.contact1, indent + "    ");
  }
  os << "\n" << indent << "  ]\n";
  os << indent << "}";
}

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
      config.indenter_contact_attrs = {8, 2, 10};
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
  bool enzyme_quadrature = true;
  bool projection_smoothing = true;
  std::string projection_smoothing_curve = "quintic";
  bool fixed_integration_jacobian = false;
  bool qp_frozen_integration = false;
  bool eta_gap_scaling = true;
  bool eta_angle_smoothing = false;
  double eta_angle_smoothing_start_angle = 80.0;
  bool nodal_energy_angle_smoothing = true;
  double active_set_smoothing_gap = 0.001;
  double qp_derivative_blend_min_gap = 0.0;
  double qp_derivative_blend_max_gap = 0.0;
  bool qp_derivative_blend_enzyme_gap_weight = true;
  bool qp_derivative_blend_adaptive_gap = false;
  double qp_derivative_blend_adaptive_min_gap_scale = 1.5;
  double qp_derivative_blend_adaptive_max_gap_scale = 20.0;
  double qp_derivative_blend_force_residual_min = 0.0;
  double qp_derivative_blend_force_residual_max = 0.0;
  bool print_qp_diagnostics = false;
  std::string nonlinear_solver = "trust-region";
  std::string run_summary_json;
  int num_steps_override = -1;
  int nonlinear_print_level = -1;
  int nonlinear_max_line_search_iterations = -1;
  axom::CLI::App app{"2D contact ironing example"};
  app.add_option("--case", selected_case, "Ironing case: square, circle, or twisted")
      ->check(axom::CLI::IsMember({"square", "circle", "twisted"}));
  app.add_option("--energy-mortar-normal", selected_normal, "Energy mortar normal field: element or averaged")
      ->check(axom::CLI::IsMember({"element", "averaged"}));
  app.add_option("--energy-mortar-penalty-mode", penalty_mode,
                 "Energy mortar penalty enforcement mode: nodal, quadrature-point, or nodal-energy")
      ->check(axom::CLI::IsMember({"nodal", "quadrature-point", "nodal-energy"}));
  app.add_option("--energy-mortar-nodal-energy-basis", nodal_energy_basis, "Nodal energy basis: fe or cubic-spline")
      ->check(axom::CLI::IsMember({"fe", "cubic-spline"}));
  app.add_flag("--energy-mortar-enzyme-quadrature,!--no-energy-mortar-enzyme-quadrature", enzyme_quadrature,
               "Differentiate through geometry-dependent quadrature construction in EnergyMortar");
  app.add_flag("--energy-mortar-projection-smoothing,!--no-energy-mortar-projection-smoothing", projection_smoothing,
               "Use projection-bound smoothing in the energy mortar contact calculation");
  app.add_option("--energy-mortar-projection-smoothing-curve", projection_smoothing_curve,
                 "Energy mortar projection-bound smoothing curve: quadratic or quintic")
      ->check(axom::CLI::IsMember({"quadratic", "quintic"}));
  app.add_flag("--energy-mortar-fixed-integration-jacobian,!--no-energy-mortar-fixed-integration-jacobian",
               fixed_integration_jacobian,
               "Hold the physical integration measure fixed during EnergyMortar differentiation");
  app.add_flag("--energy-mortar-qp-frozen-integration,!--no-energy-mortar-qp-frozen-integration", qp_frozen_integration,
               "Use previous-timestep cached quadrature points, weights, and J for the simplified QP blend path");
  app.add_flag("--energy-mortar-eta-gap-scaling,!--no-energy-mortar-eta-gap-scaling", eta_gap_scaling,
               "Scale the energy mortar normal gap by eta, the surface-normal dot product");
  app.add_flag("--energy-mortar-eta-angle-smoothing,!--no-energy-mortar-eta-angle-smoothing", eta_angle_smoothing,
               "Smooth energy mortar eta to zero near 90 degrees when eta gap scaling is disabled");
  app.add_option("--energy-mortar-eta-angle-smoothing-start-angle", eta_angle_smoothing_start_angle,
                 "Energy mortar eta angle-smoothing start angle in degrees; smoothing ends at 90 degrees")
      ->check(axom::CLI::NonNegativeNumber);
  app.add_flag("--energy-mortar-nodal-energy-angle-smoothing,!--no-energy-mortar-nodal-energy-angle-smoothing",
               nodal_energy_angle_smoothing, "Use 80-to-90 degree angle smoothing in nodal energy mode");
  app.add_option("--energy-mortar-active-set-smoothing-gap", active_set_smoothing_gap,
                 "Energy mortar active-set smoothing transition gap; disabled when <= 0")
      ->check(axom::CLI::NonNegativeNumber);
  app.add_option("--energy-mortar-qp-derivative-blend-min-gap", qp_derivative_blend_min_gap,
                 "Residual gap below which the full energy mortar QP derivative is used")
      ->check(axom::CLI::NonNegativeNumber);
  app.add_option("--energy-mortar-qp-derivative-blend-max-gap", qp_derivative_blend_max_gap,
                 "Residual gap above which the simplified energy mortar QP derivative is used")
      ->check(axom::CLI::NonNegativeNumber);
  app.add_flag(
      "--energy-mortar-qp-derivative-blend-enzyme-gap-weight,"
      "!--no-energy-mortar-qp-derivative-blend-enzyme-gap-weight",
      qp_derivative_blend_enzyme_gap_weight,
      "Differentiate the gap-based energy mortar QP derivative blend weight with Enzyme");
  app.add_flag("--energy-mortar-qp-derivative-blend-adaptive-gap", qp_derivative_blend_adaptive_gap,
               "Update the energy mortar QP derivative gap blend range from the previous timestep average gap");
  app.add_option("--energy-mortar-qp-derivative-blend-adaptive-min-gap-scale",
                 qp_derivative_blend_adaptive_min_gap_scale,
                 "Adaptive QP derivative blend min gap scale applied to the previous timestep average gap")
      ->check(axom::CLI::PositiveNumber);
  app.add_option("--energy-mortar-qp-derivative-blend-adaptive-max-gap-scale",
                 qp_derivative_blend_adaptive_max_gap_scale,
                 "Adaptive QP derivative blend max gap scale applied to the adaptive min gap")
      ->check(axom::CLI::PositiveNumber);
  app.add_option("--energy-mortar-qp-derivative-blend-force-residual-min", qp_derivative_blend_force_residual_min,
                 "Force residual norm below which the full energy mortar QP derivative is used")
      ->check(axom::CLI::NonNegativeNumber);
  app.add_option("--energy-mortar-qp-derivative-blend-force-residual-max", qp_derivative_blend_force_residual_max,
                 "Force residual norm above which the simplified energy mortar QP derivative is used")
      ->check(axom::CLI::NonNegativeNumber);
  app.add_flag("--energy-mortar-qp-diagnostics", print_qp_diagnostics,
               "Print EnergyMortar QP diagnostics during nonlinear iterations");
  app.add_option("--nonlinear-solver", nonlinear_solver, "Nonlinear solver: trust-region or newton-line-search")
      ->check(axom::CLI::IsMember({"trust-region", "newton-line-search"}));
  app.add_option("--run-summary-json", run_summary_json, "Write a structured run summary JSON file");
  app.add_option("--num-steps", num_steps_override, "Override the number of time steps")
      ->check(axom::CLI::NonNegativeNumber);
  app.add_option("--nonlinear-print-level", nonlinear_print_level, "Override the nonlinear solver print level")
      ->check(axom::CLI::Range(0, 2));
  app.add_option("--nonlinear-max-line-search-iterations", nonlinear_max_line_search_iterations,
                 "Override the maximum Newton line-search cutbacks")
      ->check(axom::CLI::NonNegativeNumber);
  app.set_help_flag("--help");
  CLI11_PARSE(app, argc, argv);

  SLIC_ERROR_ROOT_IF((qp_derivative_blend_min_gap > 0.0 || qp_derivative_blend_max_gap > 0.0) &&
                         qp_derivative_blend_max_gap <= qp_derivative_blend_min_gap,
                     "The energy mortar QP derivative blend max gap must be greater than the min gap.");
  SLIC_ERROR_ROOT_IF(qp_derivative_blend_adaptive_gap && (qp_derivative_blend_min_gap <= 0.0 ||
                                                          qp_derivative_blend_max_gap <= qp_derivative_blend_min_gap),
                     "Adaptive energy mortar QP derivative gap blending requires initial min/max gaps with max > min.");
  SLIC_ERROR_ROOT_IF((qp_derivative_blend_force_residual_min > 0.0 || qp_derivative_blend_force_residual_max > 0.0) &&
                         qp_derivative_blend_force_residual_max <= qp_derivative_blend_force_residual_min,
                     "The energy mortar QP derivative blend force residual max must be greater than the min.");
  SLIC_ERROR_ROOT_IF(qp_derivative_blend_adaptive_gap && qp_derivative_blend_adaptive_max_gap_scale <= 1.0,
                     "The adaptive energy mortar QP derivative blend max gap scale must be greater than 1.");
  SLIC_ERROR_ROOT_IF(eta_angle_smoothing_start_angle >= 90.0,
                     "The energy mortar eta angle-smoothing start angle must be in [0, 90) degrees.");
  SLIC_WARNING_ROOT_IF(
      (qp_derivative_blend_min_gap > 0.0 || qp_derivative_blend_max_gap > 0.0 || qp_derivative_blend_adaptive_gap) &&
          (qp_derivative_blend_force_residual_min > 0.0 || qp_derivative_blend_force_residual_max > 0.0),
      "Force-residual derivative blending overrides gap-residual derivative blending.");
  SLIC_WARNING_ROOT_IF(qp_derivative_blend_adaptive_gap && penalty_mode != "quadrature-point",
                       "Energy mortar QP adaptive derivative gap blending is only used with quadrature-point penalty "
                       "mode.");
  SLIC_WARNING_ROOT_IF(qp_frozen_integration && penalty_mode != "quadrature-point",
                       "Energy mortar QP frozen integration is only used with quadrature-point penalty mode.");

#ifndef MFEM_USE_STRUMPACK
  SLIC_INFO_ROOT("Contact requires MFEM built with strumpack.");
  return 1;
#endif

  const auto ironing_case = parseCase(selected_case);
  const std::string case_name = caseName(ironing_case);
  const std::string run_name = makeRunName(case_name, selected_normal, penalty_mode, enzyme_quadrature,
                                           projection_smoothing, projection_smoothing_curve, fixed_integration_jacobian,
                                           qp_frozen_integration, qp_derivative_blend_adaptive_gap, nonlinear_solver);
  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, run_name + "_data");

  CaseConfig config = makeCaseConfig(ironing_case);
  if (num_steps_override >= 0) {
    config.num_steps = num_steps_override;
  }
  if (nonlinear_print_level >= 0) {
    config.nonlinear_options.print_level = nonlinear_print_level;
  }
  config.nonlinear_options.nonlin_solver = parseNonlinearSolver(nonlinear_solver);
  if (nonlinear_max_line_search_iterations >= 0) {
    config.nonlinear_options.max_line_search_iterations = nonlinear_max_line_search_iterations;
  }
  const bool force_residual_blend = qp_derivative_blend_force_residual_max > qp_derivative_blend_force_residual_min;
  if (force_residual_blend || print_qp_diagnostics) {
    const bool add_secondary_contact = config.add_secondary_contact;
    config.nonlinear_options.residual_norm_callback =
        [qp_derivative_blend_force_residual_min, qp_derivative_blend_force_residual_max, force_residual_blend,
         print_qp_diagnostics, add_secondary_contact, callback_count = 0](double residual_norm) mutable {
          if (force_residual_blend) {
            const double weight = fullPathWeightFromResidual(residual_norm, qp_derivative_blend_force_residual_min,
                                                             qp_derivative_blend_force_residual_max);
            tribol::setEnergyMortarQpDerivativeBlendWeight(0, weight);
            if (add_secondary_contact) {
              tribol::setEnergyMortarQpDerivativeBlendWeight(1, weight);
            }
          }
          if (print_qp_diagnostics) {
            auto print_diagnostics = [residual_norm, callback_count](int interaction) {
              const auto diagnostics = tribol::getMfemEnergyMortarQpDiagnostics(interaction);
              SLIC_INFO_ROOT("Energy mortar QP diagnostics callback "
                             << callback_count << ", interaction " << interaction << ": residual_norm=" << residual_norm
                             << ", energy=" << diagnostics.energy << ", active_pairs=" << diagnostics.active_pair_count
                             << ", contributing_pairs=" << diagnostics.contributing_pair_count
                             << ", gap[min/avg/max]=" << diagnostics.residual_gap_min << "/"
                             << diagnostics.residual_gap_average << "/" << diagnostics.residual_gap_max
                             << ", full_weight[min/avg/max]=" << diagnostics.blend_weight_min << "/"
                             << diagnostics.blend_weight_average << "/" << diagnostics.blend_weight_max
                             << ", weight_counts(full/blend/simplified)=" << diagnostics.full_weight_pair_count << "/"
                             << diagnostics.blended_pair_count << "/" << diagnostics.simplified_pair_count
                             << ", |Efull-Esimpl|[avg/max]=" << diagnostics.full_simplified_energy_difference_average
                             << "/" << diagnostics.full_simplified_energy_difference_max
                             << ", missing_frozen_pairs=" << diagnostics.missing_frozen_integration_pair_count);
            };
            print_diagnostics(0);
            if (add_secondary_contact) {
              print_diagnostics(1);
            }
          }
          ++callback_count;
        };
  }
  std::vector<smith::NonlinearSolveStats> nonlinear_solve_stats;
  config.nonlinear_options.solve_stats_callback = [&nonlinear_solve_stats](const smith::NonlinearSolveStats& stats) {
    nonlinear_solve_stats.push_back(stats);
  };
  config.name = run_name;
  auto& mesh = config.mesh;

  smith::LinearSolverOptions linear_options =
      nonlinear_solver == "newton-line-search"
          ? smith::LinearSolverOptions{.linear_solver = smith::LinearSolver::Strumpack,
                                       .preconditioner = smith::Preconditioner::None,
                                       .print_level = 0}
          : smith::LinearSolverOptions{.linear_solver = smith::LinearSolver::CG,
                                       .preconditioner = smith::Preconditioner::HypreAMG,
                                       .print_level = 0};

  smith::ContactOptions contact_options{.method = smith::ContactMethod::EnergyMortar,
                                        .enforcement = smith::ContactEnforcement::Penalty,
                                        .type = smith::ContactType::Frictionless,
                                        .penalty = 30000.0,
                                        .jacobian = smith::ContactJacobian::Exact};

  smith::SolidMechanicsContact<P, DIM, smith::Parameters<smith::L2<0>, smith::L2<0>>> solid_solver(
      config.nonlinear_options, linear_options, smith::solid_mechanics::default_quasistatic_options, config.name, mesh,
      {"bulk_mod", "shear_mod"}, 0, 0.0, false, false);

  mfem::VisItDataCollection visit_dc(config.name + "_visit", &mesh->mfemParMesh());
  visit_dc.SetPrefixPath("visit_out");
  visit_dc.RegisterField("displacement", &solid_solver.displacement().gridFunction());
  auto save_visit_output = [&](int cycle, double time) {
    solid_solver.displacement().gridFunction();
    visit_dc.SetCycle(cycle);
    visit_dc.SetTime(time);
    visit_dc.Save();
  };
  save_visit_output(0, 0.0);

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
  const double eta_angle_smoothing_start = eta_angle_smoothing_start_angle * M_PI / 180.0;

  solid_solver.addContactInteraction(0, config.substrate_contact_attrs, config.indenter_contact_attrs, contact_options);
  tribol::setEnergyMortarEnzymeQuadrature(0, enzyme_quadrature);
  tribol::setEnergyMortarFixedIntegrationJacobian(0, fixed_integration_jacobian);
  tribol::setEnergyMortarQpFrozenIntegration(0, qp_frozen_integration);
  tribol::setEnergyMortarProjectionSmoothing(0, projection_smoothing);
  tribol::setEnergyMortarProjectionSmoothingCurve(0, parseProjectionSmoothingCurve(projection_smoothing_curve));
  tribol::setEnergyMortarH1ActiveSetSmoothing(0, active_set_smoothing_gap);
  tribol::setEnergyMortarQpDerivativeBlendGapRange(0, qp_derivative_blend_min_gap, qp_derivative_blend_max_gap);
  tribol::setEnergyMortarQpDerivativeBlendEnzymeGapWeight(0, qp_derivative_blend_enzyme_gap_weight);
  tribol::setEnergyMortarEtaGapScaling(0, eta_gap_scaling);
  tribol::setEnergyMortarEtaAngleSmoothing(0, eta_angle_smoothing);
  tribol::setEnergyMortarEtaAngleSmoothingStart(0, eta_angle_smoothing_start);
  tribol::setEnergyMortarPenaltyMode(0, penalty_mode == "nodal-energy" ? tribol::EnergyMortarPenaltyMode::NODAL_ENERGY
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
    tribol::setEnergyMortarEnzymeQuadrature(1, enzyme_quadrature);
    tribol::setEnergyMortarFixedIntegrationJacobian(1, fixed_integration_jacobian);
    tribol::setEnergyMortarQpFrozenIntegration(1, qp_frozen_integration);
    tribol::setEnergyMortarProjectionSmoothing(1, projection_smoothing);
    tribol::setEnergyMortarProjectionSmoothingCurve(1, parseProjectionSmoothingCurve(projection_smoothing_curve));
    tribol::setEnergyMortarH1ActiveSetSmoothing(1, active_set_smoothing_gap);
    tribol::setEnergyMortarQpDerivativeBlendGapRange(1, qp_derivative_blend_min_gap, qp_derivative_blend_max_gap);
    tribol::setEnergyMortarQpDerivativeBlendEnzymeGapWeight(1, qp_derivative_blend_enzyme_gap_weight);
    tribol::setEnergyMortarEtaGapScaling(1, eta_gap_scaling);
    tribol::setEnergyMortarEtaAngleSmoothing(1, eta_angle_smoothing);
    tribol::setEnergyMortarEtaAngleSmoothingStart(1, eta_angle_smoothing_start);
    tribol::setEnergyMortarPenaltyMode(1, penalty_mode == "nodal-energy" ? tribol::EnergyMortarPenaltyMode::NODAL_ENERGY
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

  auto print_qp_residual_gap_average = [&](int cycle, double time) {
    SLIC_INFO_ROOT("Energy mortar QP residual gap average at cycle "
                   << cycle << ", time " << time
                   << ", interaction 0: " << tribol::getMfemEnergyMortarQpResidualGapAverage(0));
    if (config.add_secondary_contact) {
      SLIC_INFO_ROOT("Energy mortar QP residual gap average at cycle "
                     << cycle << ", time " << time
                     << ", interaction 1: " << tribol::getMfemEnergyMortarQpResidualGapAverage(1));
    }
  };

  auto update_adaptive_qp_derivative_blend_gap_range = [&](int cycle, double time) {
    if (!qp_derivative_blend_adaptive_gap) {
      return;
    }
    auto update_interaction = [&](int interaction) {
      const double average_gap = tribol::getMfemEnergyMortarQpResidualGapAverage(interaction);
      const double min_gap = qp_derivative_blend_adaptive_min_gap_scale * std::abs(average_gap);
      const double max_gap = qp_derivative_blend_adaptive_max_gap_scale * min_gap;
      tribol::setEnergyMortarQpDerivativeBlendGapRange(interaction, min_gap, max_gap);
      SLIC_INFO_ROOT("Energy mortar adaptive QP derivative gap blend range at cycle "
                     << cycle << ", time " << time << ", interaction " << interaction << ": average_gap=" << average_gap
                     << ", min_gap=" << min_gap << ", max_gap=" << max_gap);
    };
    update_interaction(0);
    if (config.add_secondary_contact) {
      update_interaction(1);
    }
  };

  auto update_qp_frozen_integration_data = [&]() {
    if (!qp_frozen_integration) {
      return;
    }
    tribol::updateEnergyMortarQpFrozenIntegrationData(0);
    if (config.add_secondary_contact) {
      tribol::updateEnergyMortarQpFrozenIntegrationData(1);
    }
  };

  update_qp_frozen_integration_data();

  std::vector<CycleRunStats> cycle_run_stats;
  constexpr double dt = 1.0;
  for (int i{0}; i < config.num_steps; ++i) {
    const auto stats_size_before_solve = nonlinear_solve_stats.size();
    solid_solver.advanceTimestep(dt);
    const int cycle = i + 1;
    const double time = cycle * dt;
    print_qp_residual_gap_average(cycle, time);
    CycleRunStats cycle_stats;
    cycle_stats.cycle = cycle;
    cycle_stats.time = time;
    if (nonlinear_solve_stats.size() > stats_size_before_solve) {
      cycle_stats.nonlinear = nonlinear_solve_stats.back();
    }
    cycle_stats.contact0 = tribol::getMfemEnergyMortarQpDiagnostics(0);
    cycle_stats.has_secondary_contact = config.add_secondary_contact;
    if (config.add_secondary_contact) {
      cycle_stats.contact1 = tribol::getMfemEnergyMortarQpDiagnostics(1);
    }
    cycle_run_stats.push_back(cycle_stats);
    update_adaptive_qp_derivative_blend_gap_range(cycle, time);
    update_qp_frozen_integration_data();
    save_visit_output(i, time);

    solid_solver.outputStateToDisk(paraview_name);
    save_submesh_normals(cycle, time);
  }

  if (!run_summary_json.empty()) {
    std::ofstream out(run_summary_json);
    SLIC_ERROR_ROOT_IF(!out, "Could not open run summary JSON file '" << run_summary_json << "'.");
    out << std::setprecision(16);

    int total_iterations = 0;
    int max_iterations = 0;
    int failed_cycles = 0;
    int total_line_search_cutbacks = 0;
    int total_rejected_steps = 0;
    int total_trust_region_cg_iterations = 0;
    int total_residual_evaluations = 0;
    int total_jacobian_assemblies = 0;
    int total_hessian_vector_products = 0;
    for (const auto& cycle_stats : cycle_run_stats) {
      const auto& nonlinear = cycle_stats.nonlinear;
      total_iterations += nonlinear.iterations;
      max_iterations = std::max(max_iterations, nonlinear.iterations);
      failed_cycles += nonlinear.converged ? 0 : 1;
      total_line_search_cutbacks += nonlinear.line_search_cutbacks;
      total_rejected_steps += nonlinear.rejected_steps;
      total_trust_region_cg_iterations += nonlinear.trust_region_cg_iterations;
      total_residual_evaluations += nonlinear.residual_evaluations;
      total_jacobian_assemblies += nonlinear.jacobian_assemblies;
      total_hessian_vector_products += nonlinear.hessian_vector_products;
    }
    const double average_iterations =
        cycle_run_stats.empty() ? 0.0
                                : static_cast<double>(total_iterations) / static_cast<double>(cycle_run_stats.size());

    out << "{\n";
    out << "  \"problem\": \"ironing_2D\",\n";
    out << "  \"run_name\": " << jsonString(config.name) << ",\n";
    out << "  \"options\": {\n";
    out << "    \"case\": " << jsonString(case_name) << ",\n";
    out << "    \"normal\": " << jsonString(selected_normal) << ",\n";
    out << "    \"penalty_mode\": " << jsonString(penalty_mode) << ",\n";
    out << "    \"nodal_energy_basis\": " << jsonString(nodal_energy_basis) << ",\n";
    out << "    \"nonlinear_solver\": " << jsonString(nonlinear_solver) << ",\n";
    out << "    \"linear_solver\": " << jsonString(smith::linearName(linear_options.linear_solver)) << ",\n";
    out << "    \"preconditioner\": " << jsonString(smith::preconditionerName(linear_options.preconditioner)) << ",\n";
    out << "    \"enzyme_quadrature\": " << (enzyme_quadrature ? "true" : "false") << ",\n";
    out << "    \"projection_smoothing\": " << (projection_smoothing ? "true" : "false") << ",\n";
    out << "    \"projection_smoothing_curve\": " << jsonString(projection_smoothing_curve) << ",\n";
    out << "    \"fixed_integration_jacobian\": " << (fixed_integration_jacobian ? "true" : "false") << ",\n";
    out << "    \"qp_frozen_integration\": " << (qp_frozen_integration ? "true" : "false") << ",\n";
    out << "    \"eta_gap_scaling\": " << (eta_gap_scaling ? "true" : "false") << ",\n";
    out << "    \"eta_angle_smoothing\": " << (eta_angle_smoothing ? "true" : "false") << ",\n";
    out << "    \"eta_angle_smoothing_start_angle\": " << eta_angle_smoothing_start_angle << ",\n";
    out << "    \"nodal_energy_angle_smoothing\": " << (nodal_energy_angle_smoothing ? "true" : "false") << ",\n";
    out << "    \"active_set_smoothing_gap\": " << active_set_smoothing_gap << ",\n";
    out << "    \"qp_derivative_blend_min_gap\": " << qp_derivative_blend_min_gap << ",\n";
    out << "    \"qp_derivative_blend_max_gap\": " << qp_derivative_blend_max_gap << ",\n";
    out << "    \"qp_derivative_blend_enzyme_gap_weight\": "
        << (qp_derivative_blend_enzyme_gap_weight ? "true" : "false") << ",\n";
    out << "    \"qp_derivative_blend_adaptive_gap\": " << (qp_derivative_blend_adaptive_gap ? "true" : "false")
        << ",\n";
    out << "    \"qp_derivative_blend_force_residual_min\": " << qp_derivative_blend_force_residual_min << ",\n";
    out << "    \"qp_derivative_blend_force_residual_max\": " << qp_derivative_blend_force_residual_max << "\n";
    out << "  },\n";
    out << "  \"summary\": {\n";
    out << "    \"cycles\": " << cycle_run_stats.size() << ",\n";
    out << "    \"converged\": " << (failed_cycles == 0 ? "true" : "false") << ",\n";
    out << "    \"failed_cycles\": " << failed_cycles << ",\n";
    out << "    \"total_nonlinear_iterations\": " << total_iterations << ",\n";
    out << "    \"average_nonlinear_iterations\": " << average_iterations << ",\n";
    out << "    \"max_nonlinear_iterations\": " << max_iterations << ",\n";
    out << "    \"total_line_search_cutbacks\": " << total_line_search_cutbacks << ",\n";
    out << "    \"total_rejected_steps\": " << total_rejected_steps << ",\n";
    out << "    \"total_trust_region_cg_iterations\": " << total_trust_region_cg_iterations << ",\n";
    out << "    \"total_residual_evaluations\": " << total_residual_evaluations << ",\n";
    out << "    \"total_jacobian_assemblies\": " << total_jacobian_assemblies << ",\n";
    out << "    \"total_hessian_vector_products\": " << total_hessian_vector_products << "\n";
    out << "  },\n";
    out << "  \"cycles\": [\n";
    for (std::size_t i = 0; i < cycle_run_stats.size(); ++i) {
      writeCycleRunStatsJson(out, cycle_run_stats[i], "    ");
      out << (i + 1 == cycle_run_stats.size() ? "\n" : ",\n");
    }
    out << "  ]\n";
    out << "}\n";
  }

  return 0;
}
