// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include "mfem.hpp"
#include <string>
#include <variant>
#include <vector>
#include "axom/slic.hpp"

#include "helpers/extended_thermomechanics.hpp"

#include "smith/infrastructure/application_manager.hpp"
#include "smith/infrastructure/logger.hpp"
#include "smith/numerics/solver_config.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/state/state_manager.hpp"

#include "smith/differentiable_numerics/nonlinear_block_solver.hpp"
#include "smith/differentiable_numerics/paraview_writer.hpp"

#include "helpers/extended_thermomechanics_materials.hpp"
namespace example_etm {

#define SLIC_INFO_ROOT_FLUSH(...) \
  do {                            \
    SLIC_INFO_ROOT(__VA_ARGS__);  \
    smith::logger::flush();       \
  } while (0)

static constexpr int dim = 3;
[[maybe_unused]] static constexpr int vdim = dim;
static constexpr int StateMatrixDim = (dim == 2) ? 3 : (dim == 3) ? 6 : -1;
static constexpr int Statedim = StateMatrixDim + 1;
static constexpr int displacement_order = 1;
static constexpr int temperature_order = 1;
// using StateSpace = smith::L2<1>;
using ExtendedStateSpace = smith::L2<1, Statedim>;

enum class CoupledLinearSolver
{
  Strumpack,
  GmresBlockAmg
};

enum class MaterialModelKind
{
  GreenSaintVenant,
  ThermalStiffening
};

enum class GmresBlockPreconditioner
{
  Diagonal,
  LowerTriangular,
  SchurDiagonal,
  SchurLower,
  SchurUpper,
  SchurFull
};

bool usesSchurPreconditioner(GmresBlockPreconditioner preconditioner)
{
  switch (preconditioner) {
    case GmresBlockPreconditioner::SchurDiagonal:
    case GmresBlockPreconditioner::SchurLower:
    case GmresBlockPreconditioner::SchurUpper:
    case GmresBlockPreconditioner::SchurFull:
      return true;
    case GmresBlockPreconditioner::Diagonal:
    case GmresBlockPreconditioner::LowerTriangular:
      return false;
  }
  return false;
}

std::string solverName(CoupledLinearSolver solver)
{
  switch (solver) {
    case CoupledLinearSolver::Strumpack:
      return "strumpack";
    case CoupledLinearSolver::GmresBlockAmg:
      return "gmres-block-amg";
  }
  return "unknown";
}

bool parseSolverArgument(const std::string& value, CoupledLinearSolver& solver)
{
  if (value == "strumpack") {
    solver = CoupledLinearSolver::Strumpack;
    return true;
  }
  if (value == "gmres-block-amg") {
    solver = CoupledLinearSolver::GmresBlockAmg;
    return true;
  }
  return false;
}

std::string materialModelName(MaterialModelKind material_model)
{
  switch (material_model) {
    case MaterialModelKind::GreenSaintVenant:
      return "green-saint-venant";
    case MaterialModelKind::ThermalStiffening:
      return "thermal-stiffening";
  }
  return "unknown";
}

bool parseMaterialModelArgument(const std::string& value, MaterialModelKind& material_model)
{
  if (value == "green-saint-venant") {
    material_model = MaterialModelKind::GreenSaintVenant;
    return true;
  }
  if (value == "thermal-stiffening") {
    material_model = MaterialModelKind::ThermalStiffening;
    return true;
  }
  return false;
}

std::string gmresBlockPreconditionerName(GmresBlockPreconditioner preconditioner)
{
  switch (preconditioner) {
    case GmresBlockPreconditioner::Diagonal:
      return "diagonal";
    case GmresBlockPreconditioner::LowerTriangular:
      return "lower-triangular";
    case GmresBlockPreconditioner::SchurDiagonal:
      return "schur-diagonal";
    case GmresBlockPreconditioner::SchurLower:
      return "schur-lower";
    case GmresBlockPreconditioner::SchurUpper:
      return "schur-upper";
    case GmresBlockPreconditioner::SchurFull:
      return "schur-full";
  }
  return "unknown";
}

bool parseGmresBlockPreconditionerArgument(const std::string& value, GmresBlockPreconditioner& preconditioner)
{
  if (value == "diagonal") {
    preconditioner = GmresBlockPreconditioner::Diagonal;
    return true;
  }
  if (value == "lower-triangular") {
    preconditioner = GmresBlockPreconditioner::LowerTriangular;
    return true;
  }
  if (value == "schur-diagonal") {
    preconditioner = GmresBlockPreconditioner::SchurDiagonal;
    return true;
  }
  if (value == "schur-lower") {
    preconditioner = GmresBlockPreconditioner::SchurLower;
    return true;
  }
  if (value == "schur-upper") {
    preconditioner = GmresBlockPreconditioner::SchurUpper;
    return true;
  }
  if (value == "schur-full") {
    preconditioner = GmresBlockPreconditioner::SchurFull;
    return true;
  }
  return false;
}

smith::LinearSolverOptions makeMechanicsStageLinearSolverOptions()
{
  return smith::LinearSolverOptions{.linear_solver = smith::LinearSolver::CG,
                                    .preconditioner = smith::Preconditioner::HypreAMG,
                                    .relative_tol = 1.0e-6,
                                    .absolute_tol = 1.0e-7,
                                    .max_iterations = 300,
                                    .print_level = 0,
                                    .preconditioner_print_level = 0};
}

smith::LinearSolverOptions makeThermalStateStageLinearSolverOptions(CoupledLinearSolver solver,
                                                                    GmresBlockPreconditioner gmres_block_preconditioner)
{
  smith::LinearSolverOptions options{.linear_solver = smith::LinearSolver::Strumpack,
                                     .preconditioner = smith::Preconditioner::HypreJacobi,
                                     .relative_tol = 1e-8,
                                     .absolute_tol = 1e-10,
                                     .max_iterations = 300,
                                     .print_level = 0};

  if (solver == CoupledLinearSolver::GmresBlockAmg) {
    options.linear_solver = smith::LinearSolver::GMRES;
    switch (gmres_block_preconditioner) {
      case GmresBlockPreconditioner::Diagonal:
        options.preconditioner = smith::Preconditioner::BlockDiagonal;
        break;
      case GmresBlockPreconditioner::LowerTriangular:
        options.preconditioner = smith::Preconditioner::BlockTriangularLower;
        break;
      case GmresBlockPreconditioner::SchurDiagonal:
        options.preconditioner = smith::Preconditioner::BlockSchurDiagonal;
        break;
      case GmresBlockPreconditioner::SchurLower:
        options.preconditioner = smith::Preconditioner::BlockSchurLower;
        break;
      case GmresBlockPreconditioner::SchurUpper:
        options.preconditioner = smith::Preconditioner::BlockSchurUpper;
        break;
      case GmresBlockPreconditioner::SchurFull:
        options.preconditioner = smith::Preconditioner::BlockSchurFull;
        break;
    }

    smith::LinearSolverOptions thermal_block_options{.linear_solver = smith::LinearSolver::Strumpack,
                                                     .preconditioner = smith::Preconditioner::HypreAMG,
                                                     .relative_tol = 1.0e-9,
                                                     .absolute_tol = 1.0e-12,
                                                     .max_iterations = 100,
                                                     .print_level = 0,
                                                     .preconditioner_print_level = 0};

    smith::LinearSolverOptions state_block_options{.linear_solver = smith::LinearSolver::Strumpack};
    options.sub_block_linear_solver_options = {thermal_block_options, state_block_options};
  }

  return options;
}

std::shared_ptr<smith::CoupledSystemSolver> makeCoupledSolver(const std::shared_ptr<smith::Mesh>& mesh,
                                                              CoupledLinearSolver linear_solver,
                                                              GmresBlockPreconditioner gmres_block_preconditioner)
{
  smith::NonlinearSolverOptions mechanics_nonlinear_opts{.nonlin_solver = smith::NonlinearSolver::TrustRegion,
                                                         .relative_tol = 1.0e-8,
                                                         .absolute_tol = 1.0e-8,
                                                         .max_iterations = 500,
                                                         .print_level = 2};

  smith::NonlinearSolverOptions thermal_nonlinear_opts{.nonlin_solver = smith::NonlinearSolver::TrustRegion,
                                                       .relative_tol = 1.0e-8,
                                                       .absolute_tol = 1.0e-10,
                                                       .max_iterations = 200,
                                                       .max_line_search_iterations = 30,
                                                       .print_level = 2};

  auto mechanics_solver =
      smith::buildNonlinearBlockSolver(mechanics_nonlinear_opts, makeMechanicsStageLinearSolverOptions(), *mesh);
  auto thermal_state_solver = smith::buildNonlinearBlockSolver(
      thermal_nonlinear_opts, makeThermalStateStageLinearSolverOptions(linear_solver, gmres_block_preconditioner),
      *mesh);

  auto coupled_solver = std::make_shared<smith::CoupledSystemSolver>(20);
  coupled_solver->addSubsystemSolver({0}, mechanics_solver);
  coupled_solver->addSubsystemSolver({1, 2}, thermal_state_solver);
  return coupled_solver;
}

using GreenSaintVenantMaterial =
    extended_thermomechanics_materials::GreenSaintVenantThermoelasticWithExtendedStateMaterial;
using ThermalStiffeningMaterial = extended_thermomechanics_materials::ThermalStiffeningMaterial;
using MaterialModel = std::variant<GreenSaintVenantMaterial, ThermalStiffeningMaterial>;

GreenSaintVenantMaterial makeGreenSaintVenantMaterial(double alpha_T)
{
  double rho = 1.0;
  double E0 = 100.0;
  double nu = 0.25;
  double specific_heat = 1.00e3;
  double kappa = 0.1;
  return GreenSaintVenantMaterial{rho, E0, nu, specific_heat, alpha_T, 1.0, kappa};
}

ThermalStiffeningMaterial makeThermalStiffeningMaterial()
{
  double Km = 0.5;
  double Gm = 0.0073976;
  double betam = 0.0;
  double rhom0 = 1.0;
  double etam = 0.0;
  double Ke = 0.5;
  double Ge = 0.225075;
  double betae = 0.0;
  double rhoe0 = 1.0;
  double etae = 0.0;
  double C_v = 1.5;
  double kappa = 30.0;
  double Af = 2.5e15;
  double E_af = 1.5e5;
  double Ar = 1.0e-21;
  double E_ar = -1.55e5;
  double R = 8.314;
  double Tr = 353.0;
  double gw = 0.2;
  double wm = 0.5;

  return ThermalStiffeningMaterial{Km,  Gm,    betam, rhom0, etam, Ke,   Ge, betae, rhoe0, etae,
                                   C_v, kappa, Af,    E_af,  Ar,   E_ar, R,  Tr,    gw,    wm};
}

MaterialModel makeMaterialModel(MaterialModelKind material_model_kind, double alpha_T)
{
  switch (material_model_kind) {
    case MaterialModelKind::GreenSaintVenant:
      return makeGreenSaintVenantMaterial(alpha_T);
    case MaterialModelKind::ThermalStiffening:
      return makeThermalStiffeningMaterial();
  }
  return makeGreenSaintVenantMaterial(alpha_T);
}

int runExtendedThermomechanics(const std::shared_ptr<smith::Mesh>& mesh, double dt, double T, double alpha_T,
                               MaterialModelKind material_model_kind, CoupledLinearSolver linear_solver,
                               GmresBlockPreconditioner gmres_block_preconditioner)
{
  double initial_temperature = 0.0;
  auto material = makeMaterialModel(material_model_kind, alpha_T);

  auto coupled_solver = makeCoupledSolver(mesh, linear_solver, gmres_block_preconditioner);

  auto system =
      smith::buildExtendedThermoMechanicsSystem<dim, displacement_order, temperature_order, ExtendedStateSpace>(
          mesh, coupled_solver, "");

  std::visit([&](const auto& selected_material) { system.setMaterial(selected_material, mesh->entireBodyName()); },
             material);

  // constexpr double left_face_traction_magnitude = 1.0e-3;
  // constexpr double min_traction_scale = 1.0e-2;
  constexpr double heat_source_magnitude = 0.0;
  constexpr double dt1 = 0.05;
  constexpr double dt2 = 0.2;
  system.disp_bc->setVectorBCs<dim>(mesh->domain("left"), [](double t, smith::tensor<double, dim> X) {
    auto bc = 0.0 * X;
    // Keep the loading modest so the first Newton solves are well-conditioned.
    bc[0] = .01 * t;
    return bc;
  });
  // system.addSolidTraction("left", [=](double t, auto X, auto, auto, auto, auto, auto, auto, auto) {
  //   auto traction = 0.0 * X;
  //   auto ramp_scale = (dt1 > 0.0) ? std::min(t / dt1, 1.0) : 1.0;
  //   auto traction_scale = min_traction_scale + (1.0 - min_traction_scale) * ramp_scale;
  //   traction[1] = -left_face_traction_magnitude * traction_scale;
  //   return traction;
  // });
  // system.addSolidBodyForce(mesh->entireBodyName(), [=](double t, auto X, auto, auto, auto, auto, auto, auto) {
  //   auto force = 0.0 * X;
  //   auto ramp_scale = (dt1 > 0.0) ? std::min(t / dt1, 1.0) : 1.0;
  //   auto force_scale = min_traction_scale + (1.0 - min_traction_scale) * ramp_scale;
  //   force[1] = -left_face_traction_magnitude * force_scale;
  //   return force;
  // });
  system.disp_bc->setFixedVectorBCs<dim, vdim>(mesh->domain("right"));
  // system.disp_bc->setFixedVectorBCs<dim, vdim>(mesh->domain("left"));

  system.temperature_bc->setFixedScalarBCs<dim>(mesh->domain("left"));
  // system.temperature_bc->setFixedScalarBCs<dim>(mesh->domain("right"));

  system.addThermalHeatSource(mesh->entireBodyName(), [=](double t, auto, auto, auto, auto, auto, auto, auto, auto...) {
    if (t <= dt1) {
      return 0.0;
    }
    if (t >= dt2) {
      return heat_source_magnitude;
    }
    auto heat_scale = (dt2 > dt1) ? (t - dt1) / (dt2 - dt1) : 1.0;
    return heat_source_magnitude * heat_scale;
  });

  // Initialize displacement fields (avoid solver starting from uninitialized/NaN values).
  auto& disp_pred =
      const_cast<smith::FiniteElementState&>(*system.field_store->getField("displacement_predicted").get());
  disp_pred.setFromFieldFunction([](smith::tensor<double, dim>) { return smith::tensor<double, dim>{}; });
  const_cast<smith::FiniteElementState&>(*system.field_store->getField("displacement").get()) = disp_pred;

  // Initialize temperature fields
  auto& temp_pred =
      const_cast<smith::FiniteElementState&>(*system.field_store->getField("temperature_predicted").get());
  temp_pred.setFromFieldFunction([=](smith::tensor<double, dim>) { return initial_temperature; });
  const_cast<smith::FiniteElementState&>(*system.field_store->getField("temperature").get()) = temp_pred;

  // Initialize extended state fields (w, Feinv in Mandel-like packing)
  auto& state_pred = const_cast<smith::FiniteElementState&>(*system.field_store->getField("state_predicted").get());
  state_pred.setFromFieldFunction([](smith::tensor<double, dim>) {
    smith::tensor<double, Statedim> alpha{};
    auto [w_0, F_0] = GreenSaintVenantMaterial::SymmetricStatePacking<dim>::unpack(alpha);
    w_0 = 0.0;
    extended_thermomechanics_materials::setIdentity(F_0);
    alpha = GreenSaintVenantMaterial::SymmetricStatePacking<dim>::pack<Statedim>(w_0, F_0);
    return alpha;
  });
  const_cast<smith::FiniteElementState&>(*system.field_store->getField("state").get()) = state_pred;

  std::string pv_dir = "paraview_extended_thermomechanics";
  auto pv_writer = smith::createParaviewWriter(*mesh, system.getStateFields(), pv_dir);

  double time = 0.0;
  int cycle = 0;

  auto shape_disp = system.field_store->getShapeDisp();
  auto states = system.getStateFields();
  auto params = system.getParameterFields();

  pv_writer.write(cycle, time, states);

  [[maybe_unused]] auto print_primal_field_magnitudes = [](size_t step_index, double current_time,
                                                           const std::vector<smith::FieldState>& field_states) {
    std::cout << "step " << step_index << " time=" << current_time;
    for (const auto& field_state : field_states) {
      std::cout << " | " << field_state.get()->name() << " l2=" << field_state.get()->Norml2();
    }
    std::cout << "\n";
  };

  size_t step = 0;

  while (time < T) {
    smith::TimeInfo t_info(time, dt, step);
    SLIC_INFO_ROOT_FLUSH(
        axom::fmt::format("Extended thermomechanics timestep {}: time = {} -> {} (dt = {})", step, time, time + dt,
                          dt));
    auto [new_states, reactions] = system.advancer->advanceState(t_info, shape_disp, states, params);
    states = std::move(new_states);

    // std::cout << "step " << step << " max reaction (disp)=" << reactions[0].get()->Normlinf()
    //           << " (temp)=" << reactions[1].get()->Normlinf() << " (state)=" << reactions[2].get()->Normlinf() <<
    //           "\n";

    time += dt;
    cycle++;
    // print_primal_field_magnitudes(step, time, states);
    pv_writer.write(cycle, time, states);
    // SLIC_INFO_ROOT_FLUSH(axom::fmt::format("Completed timestep {} at time = {}", step, time));

    step++;
  }
  std::cout << "Material model: " << materialModelName(material_model_kind) << "\n";
  std::cout << "Solver: " << solverName(linear_solver) << "\n";
  std::cout << "Mechanics stage: trust-region / cg / hypre-amg\n";
  std::cout << "Thermal+state stage: "
            << ((linear_solver == CoupledLinearSolver::Strumpack) ? "strumpack" : "gmres-block-amg") << "\n";
  if (linear_solver == CoupledLinearSolver::GmresBlockAmg) {
    std::cout << "Thermal+state block preconditioner: " << gmresBlockPreconditionerName(gmres_block_preconditioner)
              << "\n";
  }
  std::cout << "Wrote ParaView output to '" << pv_dir << "'\n";
  return 0;
}

}  // namespace example_etm

int main(int argc, char** argv)
{
  smith::ApplicationManager applicationManager(argc, argv);

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: extended_thermomechanics [--nx=<int>] [--ny=<int>] [--nz=<int>] [--dt=<real>] [--T=<real>] "
                   "[--alpha=<real>] [--material=green-saint-venant|thermal-stiffening] "
                   "[--solver=strumpack|gmres-block-amg] "
                   "[--gmres-block-preconditioner=diagonal|lower-triangular|schur-diagonal|schur-lower|schur-upper|"
                   "schur-full]\n";
      std::cout << "Defaults: nx=60 ny=10 nz=10 dt=0.01 T=1.0 alpha=0.0 material=green-saint-venant solver=strumpack "
                   "gmres-block-preconditioner=diagonal\n";
      return 0;
    }
  }

  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, "solid");

  double length = 1.0;
  double width = 0.04;
  int num_elements_x = 60;
  int num_elements_y = 10;
  int num_elements_z = 10;
  double dt = 0.001;
  double T = 1.0;
  double alpha_T = 0.0*1.0e-3;
  auto material_model = example_etm::MaterialModelKind::GreenSaintVenant;
  auto solver_type = example_etm::CoupledLinearSolver::Strumpack;
  auto gmres_block_preconditioner = example_etm::GmresBlockPreconditioner::Diagonal;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto parse_int = [&](const char* prefix, int& value) {
      const std::string p(prefix);
      if (arg.rfind(p, 0) == 0) value = std::stoi(arg.substr(p.size()));
    };
    auto parse_double = [&](const char* prefix, double& value) {
      const std::string p(prefix);
      if (arg.rfind(p, 0) == 0) value = std::stod(arg.substr(p.size()));
    };
    parse_int("--nx=", num_elements_x);
    parse_int("--ny=", num_elements_y);
    parse_int("--nz=", num_elements_z);
    parse_double("--dt=", dt);
    parse_double("--T=", T);
    parse_double("--alpha=", alpha_T);
    const std::string material_prefix = "--material=";
    if (arg.rfind(material_prefix, 0) == 0) {
      const auto material_name = arg.substr(material_prefix.size());
      if (!example_etm::parseMaterialModelArgument(material_name, material_model)) {
        std::cerr << "Unknown material option '" << material_name
                  << "'. Expected green-saint-venant or thermal-stiffening.\n";
        return 1;
      }
    }
    const std::string solver_prefix = "--solver=";
    if (arg.rfind(solver_prefix, 0) == 0) {
      const auto solver_name = arg.substr(solver_prefix.size());
      if (!example_etm::parseSolverArgument(solver_name, solver_type)) {
        std::cerr << "Unknown solver option '" << solver_name << "'. Expected strumpack or gmres-block-amg.\n";
        return 1;
      }
    }
    const std::string gmres_block_preconditioner_prefix = "--gmres-block-preconditioner=";
    if (arg.rfind(gmres_block_preconditioner_prefix, 0) == 0) {
      const auto preconditioner_name = arg.substr(gmres_block_preconditioner_prefix.size());
      if (!example_etm::parseGmresBlockPreconditionerArgument(preconditioner_name, gmres_block_preconditioner)) {
        std::cerr << "Unknown GMRES block preconditioner option '" << preconditioner_name
                  << "'. Expected diagonal, lower-triangular, schur-diagonal, schur-lower, schur-upper, or "
                     "schur-full.\n";
        return 1;
      }
    }
  }

  auto mfem_shape = mfem::Element::QUADRILATERAL;
  auto mesh = std::make_shared<smith::Mesh>(
      mfem::Mesh::MakeCartesian3D(num_elements_x, num_elements_y, num_elements_z, mfem_shape, length, width, width),
      "mesh", 0, 0);
  mesh->addDomainOfBoundaryElements("left", smith::by_attr<example_etm::dim>(3));
  mesh->addDomainOfBoundaryElements("right", smith::by_attr<example_etm::dim>(5));

  return example_etm::runExtendedThermomechanics(mesh, dt, T, alpha_T, material_model, solver_type,
                                                 gmres_block_preconditioner);
}
