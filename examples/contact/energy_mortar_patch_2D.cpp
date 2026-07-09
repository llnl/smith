// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <cmath>
#include <iostream>
#include <memory>
#include <set>
#include <string>

#include "axom/CLI11.hpp"
#include "axom/sidre.hpp"
#include "axom/slic.hpp"
#include "mfem.hpp"
#include "mpi.h"
#include "shared/mesh/MeshBuilder.hpp"

#include "smith/infrastructure/application_manager.hpp"
#include "smith/numerics/functional/domain.hpp"
#include "smith/numerics/functional/tensor.hpp"
#include "smith/numerics/solver_config.hpp"
#include "smith/physics/boundary_conditions/components.hpp"
#include "smith/physics/contact/contact_config.hpp"
#include "smith/physics/materials/solid_material.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/solid_mechanics_contact.hpp"
#include "smith/physics/state/state_manager.hpp"
#include "smith/smith_config.hpp"
#include "tribol/interface/tribol.hpp"

namespace {

constexpr int P = 1;
constexpr int DIM = 2;

tribol::EnergyMortarProjectionSmoothingCurve parseProjectionSmoothingCurve(const std::string& value)
{
  if (value == "quadratic") {
    return tribol::EnergyMortarProjectionSmoothingCurve::QUADRATIC;
  }
  if (value == "quintic") {
    return tribol::EnergyMortarProjectionSmoothingCurve::QUINTIC;
  }

  SLIC_ERROR_ROOT("Unknown projection smoothing curve '" << value << "'. Expected one of: quadratic, quintic.");
  return tribol::EnergyMortarProjectionSmoothingCurve::QUADRATIC;
}

std::string etaModeName(bool eta_smoothing, const std::string& eta_smoothing_type)
{
  return eta_smoothing ? eta_smoothing_type : "none";
}

}  // namespace

int main(int argc, char* argv[])
{
  smith::ApplicationManager applicationManager(argc, argv);

  int nel_per_dir = 50;
  double prescribed_displacement = -0.01;
  double penalty = 10000.0;
  double bulk_modulus = 1000.0;
  double shear_modulus = 10.0;
  int nonlinear_print_level = 1;
  std::string name = "energy_mortar_patch_2D";
  std::string eta_smoothing_type = "angle";
  bool eta_smoothing = false;
  double eta_angle_smoothing_start_angle = 80.0;
  bool projection_smoothing = false;
  std::string projection_smoothing_curve = "quadratic";
  bool check_error = true;
  double error_tolerance = 1.0e-2;

  axom::CLI::App app{"2D EnergyMortar contact patch example"};
  app.add_option("--num-elements", nel_per_dir, "Elements per direction in each square block")
      ->check(axom::CLI::PositiveNumber);
  app.add_option("--prescribed-displacement", prescribed_displacement, "Top-surface vertical displacement");
  app.add_option("--penalty", penalty, "EnergyMortar penalty stiffness")->check(axom::CLI::PositiveNumber);
  app.add_option("--bulk-modulus", bulk_modulus, "Neo-Hookean bulk modulus")->check(axom::CLI::PositiveNumber);
  app.add_option("--shear-modulus", shear_modulus, "Neo-Hookean shear modulus")->check(axom::CLI::PositiveNumber);
  app.add_option("--nonlinear-print-level", nonlinear_print_level, "Nonlinear solver print level")
      ->check(axom::CLI::Range(0, 2));
  app.add_option("--name", name, "Output/state name prefix");
  app.add_flag("--eta-smoothing,!--no-eta-smoothing", eta_smoothing,
               "Enable eta smoothing/scaling for the EnergyMortar normal gap");
  app.add_option("--eta-smoothing-type", eta_smoothing_type,
                 "Eta smoothing/scaling type used when --eta-smoothing is enabled: angle or dot")
      ->check(axom::CLI::IsMember({"angle", "dot"}));
  app.add_option("--eta-angle-smoothing-start-angle", eta_angle_smoothing_start_angle,
                 "Angle-smoothing start angle in degrees; smoothing ends at 90 degrees")
      ->check(axom::CLI::NonNegativeNumber);
  app.add_flag("--projection-smoothing,!--no-projection-smoothing", projection_smoothing,
               "Enable EnergyMortar projection-bound smoothing");
  app.add_option("--projection-smoothing-curve", projection_smoothing_curve,
                 "Projection-bound smoothing curve: quadratic or quintic")
      ->check(axom::CLI::IsMember({"quadratic", "quintic"}));
  app.add_flag("--quadratic-projection-smoothing", projection_smoothing,
               "Enable projection smoothing with the quadratic smoothing curve");
  app.add_flag("--check,!--no-check", check_error, "Return nonzero if the L2 displacement error exceeds tolerance");
  app.add_option("--error-tolerance", error_tolerance, "L2 error tolerance used when --check is enabled")
      ->check(axom::CLI::PositiveNumber);
  app.set_help_flag("--help");
  CLI11_PARSE(app, argc, argv);

  SLIC_ERROR_ROOT_IF(eta_angle_smoothing_start_angle >= 90.0,
                     "The eta angle-smoothing start angle must be in [0, 90) degrees.");
  if (app.count("--quadratic-projection-smoothing") > 0) {
    projection_smoothing = true;
    projection_smoothing_curve = "quadratic";
  }

  MPI_Barrier(MPI_COMM_WORLD);

  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, name + "_data");

  auto mesh = std::make_shared<smith::Mesh>(
      shared::MeshBuilder::Unify({shared::MeshBuilder::SquareMesh(nel_per_dir, nel_per_dir)
                                      .translate({0.0, 1.0})
                                      .bdrAttribInfo()
                                      .updateBdrAttrib(4, 7)
                                      .updateBdrAttrib(3, 9)
                                      .updateBdrAttrib(1, 6),
                                  shared::MeshBuilder::SquareMesh(nel_per_dir, nel_per_dir)
                                      .bdrAttribInfo()
                                      .updateBdrAttrib(4, 7)
                                      .updateBdrAttrib(1, 8)
                                      .updateBdrAttrib(3, 5)}),
      "patch_mesh_2D", 0, 0);

  mfem::VisItDataCollection visit_dc(name + "_visit", &mesh->mfemParMesh());
  visit_dc.SetPrefixPath("visit_out");
  visit_dc.Save();

  mesh->addDomainOfBoundaryElements("x0_faces", smith::by_attr<DIM>(7));
  mesh->addDomainOfBoundaryElements("y0_faces", smith::by_attr<DIM>(8));
  mesh->addDomainOfBoundaryElements("Ymax_face", smith::by_attr<DIM>(9));

#ifdef MFEM_USE_STRUMPACK
  smith::LinearSolverOptions linear_options{.linear_solver = smith::LinearSolver::Strumpack, .print_level = 0};
#else
  smith::LinearSolverOptions linear_options{};
  SLIC_INFO_ROOT("Contact requires MFEM built with strumpack.");
  return 1;
#endif

  smith::NonlinearSolverOptions nonlinear_options{.nonlin_solver = smith::NonlinearSolver::TrustRegion,
                                                  .relative_tol = 1.0e-13,
                                                  .absolute_tol = 1.0e-13,
                                                  .max_iterations = 20,
                                                  .max_line_search_iterations = 12,
                                                  .print_level = nonlinear_print_level};

  smith::ContactOptions contact_options{.method = smith::ContactMethod::EnergyMortar,
                                        .enforcement = smith::ContactEnforcement::Penalty,
                                        .type = smith::ContactType::Frictionless,
                                        .penalty = penalty,
                                        .penalty2 = 0.0,
                                        .jacobian = smith::ContactJacobian::Exact};

  smith::SolidMechanicsContact<P, DIM> solid_solver(
      nonlinear_options, linear_options, smith::solid_mechanics::default_quasistatic_options, name, mesh);

  smith::solid_mechanics::NeoHookean mat{1.0, bulk_modulus, shear_modulus};
  solid_solver.setMaterial(mat, mesh->entireBody());

  auto applied_disp_function = [prescribed_displacement](smith::tensor<double, DIM>, auto) {
    return smith::tensor<double, DIM>{{0.0, prescribed_displacement}};
  };

  solid_solver.setFixedBCs(mesh->domain("x0_faces"), smith::Component::X);
  solid_solver.setFixedBCs(mesh->domain("y0_faces"), smith::Component::Y);
  solid_solver.setDisplacementBCs(applied_disp_function, mesh->domain("Ymax_face"), smith::Component::Y);

  solid_solver.addContactInteraction(0, {6}, {5}, contact_options);
  tribol::setEnergyMortarPenaltyMode(0, tribol::EnergyMortarPenaltyMode::QUADRATURE_POINT_GAP);
  tribol::setEnergyMortarNormalMode(0, tribol::EnergyMortarNormalMode::ELEMENT_NORMAL);
  tribol::setEnergyMortarProjectionSmoothing(0, projection_smoothing);
  tribol::setEnergyMortarProjectionSmoothingCurve(0, parseProjectionSmoothingCurve(projection_smoothing_curve));
  tribol::setEnergyMortarEtaGapScaling(0, eta_smoothing && eta_smoothing_type == "dot");
  tribol::setEnergyMortarEtaAngleSmoothing(0, eta_smoothing && eta_smoothing_type == "angle");
  tribol::setEnergyMortarEtaAngleSmoothingStart(0, eta_angle_smoothing_start_angle * M_PI / 180.0);

  solid_solver.completeSetup();

  const std::string paraview_name = name + "_paraview";
  solid_solver.outputStateToDisk(paraview_name);
  solid_solver.advanceTimestep(1.0);
  solid_solver.outputStateToDisk(paraview_name);

  const double c = (3.0 * bulk_modulus - 2.0 * shear_modulus) / ((3.0 * bulk_modulus + 2.0 * shear_modulus));
  mfem::VectorFunctionCoefficient elasticity_sol_coeff(DIM, [c, prescribed_displacement](const mfem::Vector& x,
                                                                                          mfem::Vector& u) {
    u[0] = -0.5 * prescribed_displacement * c * x[0];
    u[1] = 0.5 * prescribed_displacement * x[1];
  });

  const mfem::ParFiniteElementSpace& u_space_const = solid_solver.displacement().space();
  auto& u_space = const_cast<mfem::ParFiniteElementSpace&>(u_space_const);
  mfem::ParGridFunction U_exact(&u_space);
  U_exact.ProjectCoefficient(elasticity_sol_coeff);

  const mfem::ParGridFunction& U_num = solid_solver.displacement().gridFunction();
  mfem::ParGridFunction U_err(U_exact);
  U_err -= U_num;
  const double L2_err_vec = mfem::ParNormlp(U_err, 2, MPI_COMM_WORLD);

  const mfem::FiniteElementCollection* fec = u_space.FEColl();
  mfem::ParFiniteElementSpace scalar_fes(&mesh->mfemParMesh(), fec, /*vdim=*/1, u_space.GetOrdering());
  mfem::ParGridFunction ux_ex(&scalar_fes), ux_num(&scalar_fes), uy_ex(&scalar_fes), uy_num(&scalar_fes);
  const int ndofs = scalar_fes.GetNDofs();
  for (int i = 0; i < ndofs; ++i) {
    ux_ex(i) = U_exact(i);
    ux_num(i) = U_num(i);
    uy_ex(i) = U_exact(ndofs + i);
    uy_num(i) = U_num(ndofs + i);
  }

  mfem::ParGridFunction ux_err(ux_ex);
  mfem::ParGridFunction uy_err(uy_ex);
  ux_err -= ux_num;
  uy_err -= uy_num;
  const double L2_err_x = mfem::ParNormlp(ux_err, 2, MPI_COMM_WORLD);
  const double L2_err_y = mfem::ParNormlp(uy_err, 2, MPI_COMM_WORLD);

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    std::cout << "EnergyMortar patch options:\n"
              << "  penalty_mode: quadrature-point\n"
              << "  normal_mode: element\n"
              << "  eta_smoothing: " << etaModeName(eta_smoothing, eta_smoothing_type) << "\n"
              << "  eta_angle_smoothing_start_angle: " << eta_angle_smoothing_start_angle << "\n"
              << "  projection_smoothing: " << (projection_smoothing ? projection_smoothing_curve : "off") << "\n"
              << "Patch error:\n"
              << "  L2_err_vec = " << L2_err_vec << "\n"
              << "  L2_err_x   = " << L2_err_x << "\n"
              << "  L2_err_y   = " << L2_err_y << "\n";
  }

  if (check_error && (L2_err_vec > error_tolerance || L2_err_x > error_tolerance || L2_err_y > error_tolerance)) {
    SLIC_ERROR_ROOT("EnergyMortar patch error exceeded tolerance " << error_tolerance << ".");
    return 1;
  }

  return 0;
}
