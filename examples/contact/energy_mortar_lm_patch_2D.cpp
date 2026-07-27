// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <string>

#ifdef TRIBOL_USE_UMPIRE
#include "umpire/ResourceManager.hpp"
#endif

#include "axom/CLI11.hpp"
#include "axom/slic.hpp"
#include "mfem.hpp"
#include "mpi.h"
#include "shared/math/ParSparseMat.hpp"
#include "shared/mesh/MeshBuilder.hpp"

#include "smith/infrastructure/application_manager.hpp"
#include "tribol/config.hpp"
#include "tribol/interface/mfem_tribol.hpp"
#include "tribol/interface/tribol.hpp"

namespace {

tribol::EnergyMortarProjectionSmoothingCurve parseProjectionSmoothingCurve(const std::string& value)
{
  if (value == "quadratic") {
    return tribol::EnergyMortarProjectionSmoothingCurve::QUADRATIC;
  }
  if (value == "quintic") {
    return tribol::EnergyMortarProjectionSmoothingCurve::QUINTIC;
  }
  if (value == "cubic") {
    return tribol::EnergyMortarProjectionSmoothingCurve::CUBIC_IN_BOUNDS;
  }

  SLIC_ERROR_ROOT("Unknown projection smoothing curve '" << value << "'. Expected one of: quadratic, quintic, cubic.");
  return tribol::EnergyMortarProjectionSmoothingCurve::QUADRATIC;
}

struct Options {
  int num_elements = 10;
  int lower_num_elements = 0;
  int upper_num_elements = 0;
  int num_timesteps = 1;
  double prescribed_displacement = -0.01;
  double lambda = 50.0;
  double mu = 50.0;
  int max_newton_iterations = 10;
  double newton_relative_tol = 1.0e-10;
  double newton_absolute_tol = 1.0e-12;
  double linear_relative_tol = 1.0e-10;
  double linear_absolute_tol = 1.0e-14;
  int linear_max_iterations = 5000;
  int linear_print_level = 3;
  std::string linear_solver = "minres";
  double direct_pressure_regularization = 1.0e-14;
  std::string name = "energy_mortar_lm_patch_2D";
  bool eta_smoothing = false;
  std::string eta_smoothing_type = "angle";
  double eta_angle_smoothing_start_angle = 45.0;
  bool projection_smoothing = false;
  std::string projection_smoothing_curve = "quadratic";
  bool reference_geometry = false;
  bool check_error = true;
  double error_tolerance = 1.0e-2;
};

void applyEssentialBCs(const mfem::Array<int>& ess_tdof_list, mfem::Vector& residual, shared::ParSparseMat& jacobian,
                       shared::ParSparseMat& contact_gradient_transpose)
{
  for (int i = 0; i < ess_tdof_list.Size(); ++i) {
    residual(ess_tdof_list[i]) = 0.0;
  }
  jacobian.eliminateRowsCols(ess_tdof_list);
  contact_gradient_transpose.eliminateRows(ess_tdof_list);
}

}  // namespace

int main(int argc, char* argv[])
{
  smith::ApplicationManager applicationManager(argc, argv);

#ifdef TRIBOL_USE_UMPIRE
  umpire::ResourceManager::getInstance();
#endif

  Options opts;

  axom::CLI::App app{"2D EnergyMortar Lagrange-multiplier contact patch example"};
  app.add_option("--num-elements", opts.num_elements, "Elements per direction in each square block")
      ->check(axom::CLI::PositiveNumber);
  app.add_option("--lower-num-elements", opts.lower_num_elements,
                 "Elements per direction in the lower/nonmortar square block; defaults to --num-elements")
      ->check(axom::CLI::NonNegativeNumber);
  app.add_option("--upper-num-elements", opts.upper_num_elements,
                 "Elements per direction in the upper/mortar square block; defaults to --num-elements")
      ->check(axom::CLI::NonNegativeNumber);
  app.add_option("--num-steps", opts.num_timesteps, "Number of displacement load steps")
      ->check(axom::CLI::PositiveNumber);
  app.add_option("--prescribed-displacement", opts.prescribed_displacement, "Top-surface vertical displacement");
  app.add_option("--lambda", opts.lambda, "Lame parameter lambda")->check(axom::CLI::PositiveNumber);
  app.add_option("--mu", opts.mu, "Lame parameter mu")->check(axom::CLI::PositiveNumber);
  app.add_option("--max-newton-iterations", opts.max_newton_iterations, "Maximum Newton iterations per load step")
      ->check(axom::CLI::PositiveNumber);
  app.add_option("--newton-relative-tol", opts.newton_relative_tol, "Newton relative tolerance")
      ->check(axom::CLI::PositiveNumber);
  app.add_option("--newton-absolute-tol", opts.newton_absolute_tol, "Newton absolute tolerance")
      ->check(axom::CLI::PositiveNumber);
  app.add_option("--linear-relative-tol", opts.linear_relative_tol, "MINRES relative tolerance")
      ->check(axom::CLI::PositiveNumber);
  app.add_option("--linear-absolute-tol", opts.linear_absolute_tol, "MINRES absolute tolerance")
      ->check(axom::CLI::PositiveNumber);
  app.add_option("--linear-max-iterations", opts.linear_max_iterations, "Maximum MINRES iterations")
      ->check(axom::CLI::PositiveNumber);
  app.add_option("--linear-print-level", opts.linear_print_level, "MINRES print level")->check(axom::CLI::Range(0, 3));
  app.add_option("--linear-solver", opts.linear_solver, "Linear solver: direct or minres")
      ->check(axom::CLI::IsMember({"direct", "minres"}));
  app.add_option("--direct-pressure-regularization", opts.direct_pressure_regularization,
                 "Small pressure-block diagonal regularization used only by --linear-solver direct")
      ->check(axom::CLI::NonNegativeNumber);
  app.add_option("--name", opts.name, "Output/state name prefix");
  app.add_flag("--eta-smoothing,!--no-eta-smoothing", opts.eta_smoothing,
               "Enable eta smoothing/scaling for the EnergyMortar normal gap");
  app.add_option("--eta-smoothing-type", opts.eta_smoothing_type,
                 "Eta smoothing/scaling type used when --eta-smoothing is enabled: angle or dot")
      ->check(axom::CLI::IsMember({"angle", "dot"}));
  app.add_option("--eta-angle-smoothing-start-angle", opts.eta_angle_smoothing_start_angle,
                 "Angle-smoothing start angle in degrees; smoothing ends at 90 degrees")
      ->check(axom::CLI::NonNegativeNumber);
  app.add_flag("--projection-smoothing,!--no-projection-smoothing", opts.projection_smoothing,
               "Enable EnergyMortar projection-bound smoothing");
  app.add_option("--projection-smoothing-curve", opts.projection_smoothing_curve,
                 "Projection-bound smoothing curve: quadratic, quintic, or cubic")
      ->check(axom::CLI::IsMember({"quadratic", "quintic", "cubic"}));
  app.add_flag("--quadratic-projection-smoothing", opts.projection_smoothing,
               "Enable projection smoothing with the quadratic smoothing curve");
  app.add_flag("--reference-geometry,!--current-geometry", opts.reference_geometry,
               "Use reference geometry for EnergyMortar LM projections, normals, weights, and gap derivatives");
  app.add_flag("--check,!--no-check", opts.check_error, "Return nonzero if the L2 displacement error exceeds tolerance");
  app.add_option("--error-tolerance", opts.error_tolerance, "L2 error tolerance used when --check is enabled")
      ->check(axom::CLI::PositiveNumber);
  app.set_help_flag("--help");
  CLI11_PARSE(app, argc, argv);

  SLIC_ERROR_ROOT_IF(opts.eta_angle_smoothing_start_angle >= 90.0,
                     "The eta angle-smoothing start angle must be in [0, 90) degrees.");
  if (app.count("--quadratic-projection-smoothing") > 0) {
    opts.projection_smoothing = true;
    opts.projection_smoothing_curve = "quadratic";
  }
  if (opts.lower_num_elements == 0) {
    opts.lower_num_elements = opts.num_elements;
  }
  if (opts.upper_num_elements == 0) {
    opts.upper_num_elements = opts.num_elements;
  }

  auto mortar_attrs = std::set<int>({5});
  auto nonmortar_attrs = std::set<int>({3});
  auto xfixed_attrs = std::set<int>({4});
  auto yfixed_bottom_attrs = std::set<int>({1});
  auto prescribed_attrs = std::set<int>({6});

  mfem::ParMesh mesh = shared::ParMeshBuilder(
      MPI_COMM_WORLD, shared::MeshBuilder::Unify({shared::MeshBuilder::SquareMesh(opts.lower_num_elements,
                                                                                  opts.lower_num_elements)
                                                      .updateBdrAttrib(1, 1)
                                                      .updateBdrAttrib(2, 2)
                                                      .updateBdrAttrib(3, 3)
                                                      .updateBdrAttrib(4, 4),
                                                  shared::MeshBuilder::SquareMesh(opts.upper_num_elements,
                                                                                  opts.upper_num_elements)
                                                      .translate({0.0, 1.0})
                                                      .updateBdrAttrib(1, 5)
                                                      .updateBdrAttrib(2, 2)
                                                      .updateBdrAttrib(3, 6)
                                                      .updateBdrAttrib(4, 4)}));

  constexpr int order = 1;
  auto fe_coll = mfem::H1_FECollection(order, mesh.SpaceDimension());
  auto par_fe_space = mfem::ParFiniteElementSpace(&mesh, &fe_coll, mesh.SpaceDimension(), mfem::Ordering::byVDIM);
  auto coords = mfem::ParGridFunction(&par_fe_space);
  mesh.GetNodes(coords);

  mfem::ParGridFunction displacement(&par_fe_space);
  displacement = 0.0;
  mfem::ParGridFunction exact_disp(&par_fe_space);
  exact_disp = 0.0;
  mfem::ParGridFunction displacement_error(&par_fe_space);
  displacement_error = 0.0;
  mfem::ParGridFunction contact_force_field(&par_fe_space);
  contact_force_field = 0.0;
  mfem::ParGridFunction exact_contact_force_field(&par_fe_space);
  exact_contact_force_field = 0.0;
  mfem::ParGridFunction contact_force_error(&par_fe_space);
  contact_force_error = 0.0;
  mfem::ParGridFunction ref_coords(&par_fe_space);
  mesh.GetNodes(ref_coords);

  const mfem::FiniteElementCollection* fec = par_fe_space.FEColl();
  mfem::ParFiniteElementSpace scalar_fes(&mesh, fec, 1, par_fe_space.GetOrdering());
  mfem::ParGridFunction displacement_l2_error(&scalar_fes);
  displacement_l2_error = 0.0;
  mfem::ParGridFunction contact_force_l2_error(&scalar_fes);
  contact_force_l2_error = 0.0;

  mfem::Array<int> ess_vdof_marker(par_fe_space.GetVSize());
  ess_vdof_marker = 0;

  auto mark_boundary_component = [&](const std::set<int>& attrs, int component, mfem::Array<int>& marker) {
    mfem::Array<int> tmp;
    mfem::Array<int> bdr(mesh.bdr_attributes.Max());
    bdr = 0;
    for (auto attr : attrs) {
      bdr[attr - 1] = 1;
    }
    par_fe_space.GetEssentialVDofs(bdr, tmp, component);
    for (int i = 0; i < tmp.Size(); ++i) {
      marker[i] = marker[i] || tmp[i];
    }
  };

  mark_boundary_component(xfixed_attrs, 0, ess_vdof_marker);
  mark_boundary_component(yfixed_bottom_attrs, 1, ess_vdof_marker);

  mfem::Array<int> prescribed_vdof_marker(par_fe_space.GetVSize());
  prescribed_vdof_marker = 0;
  mark_boundary_component(prescribed_attrs, 1, prescribed_vdof_marker);
  for (int i = 0; i < prescribed_vdof_marker.Size(); ++i) {
    ess_vdof_marker[i] = ess_vdof_marker[i] || prescribed_vdof_marker[i];
  }

  mfem::Array<int> ess_tdof_list;
  {
    mfem::Array<int> ess_tdof_marker;
    par_fe_space.GetRestrictionMatrix()->BooleanMult(ess_vdof_marker, ess_tdof_marker);
    mfem::FiniteElementSpace::MarkerToList(ess_tdof_marker, ess_tdof_list);
  }

  mfem::Array<int> prescribed_tdof_list;
  {
    mfem::Array<int> prescribed_tdof_marker;
    par_fe_space.GetRestrictionMatrix()->BooleanMult(prescribed_vdof_marker, prescribed_tdof_marker);
    mfem::FiniteElementSpace::MarkerToList(prescribed_tdof_marker, prescribed_tdof_list);
  }

  mfem::ParBilinearForm elastic_form(&par_fe_space);
  mfem::ConstantCoefficient lambda_coeff(opts.lambda);
  mfem::ConstantCoefficient mu_coeff(opts.mu);
  elastic_form.AddDomainIntegrator(new mfem::ElasticityIntegrator(lambda_coeff, mu_coeff));
  elastic_form.Assemble();
  elastic_form.Finalize();
  shared::ParSparseMat elastic_jacobian(elastic_form.ParallelAssemble());

  mfem::VisItDataCollection visit_dc(opts.name + "_visit", &mesh);
  visit_dc.SetPrecision(8);
  visit_dc.RegisterField("displacement", &displacement);
  visit_dc.RegisterField("exact_displacement", &exact_disp);
  visit_dc.RegisterField("displacement_error", &displacement_error);
  visit_dc.RegisterField("displacement_l2_error", &displacement_l2_error);
  visit_dc.RegisterField("contact_force", &contact_force_field);
  visit_dc.RegisterField("exact_contact_force", &exact_contact_force_field);
  visit_dc.RegisterField("contact_force_error", &contact_force_error);
  visit_dc.RegisterField("contact_force_l2_error", &contact_force_l2_error);
  visit_dc.SetCycle(0);
  visit_dc.SetTime(0.0);
  visit_dc.Save();

  constexpr int cs_id = 0;
  constexpr int mesh1_id = 0;
  constexpr int mesh2_id = 1;

  coords.ReadWrite();
  tribol::registerMfemCouplingScheme(cs_id, mesh1_id, mesh2_id, mesh, coords, mortar_attrs, nonmortar_attrs,
                                     tribol::SURFACE_TO_SURFACE, tribol::NO_SLIDING, tribol::ENERGY_MORTAR,
                                     tribol::FRICTIONLESS, tribol::LAGRANGE_MULTIPLIER, tribol::BINNING_GRID);
  tribol::registerMfemReferenceCoords(cs_id, ref_coords);
  tribol::setLagrangeMultiplierOptions(cs_id, tribol::ImplicitEvalMode::MORTAR_RESIDUAL_JACOBIAN);
  tribol::setEnergyMortarReferenceGeometry(cs_id, opts.reference_geometry);
  tribol::setEnergyMortarProjectionSmoothing(cs_id, opts.projection_smoothing);
  tribol::setEnergyMortarProjectionSmoothingCurve(cs_id, parseProjectionSmoothingCurve(opts.projection_smoothing_curve));
  tribol::setEnergyMortarEtaGapScaling(cs_id, opts.eta_smoothing && opts.eta_smoothing_type == "dot");
  tribol::setEnergyMortarEtaAngleSmoothing(cs_id, opts.eta_smoothing && opts.eta_smoothing_type == "angle");
  tribol::setEnergyMortarEtaAngleSmoothingStart(cs_id, opts.eta_angle_smoothing_start_angle * M_PI / 180.0);

  auto& pressure = tribol::getMfemPressure(cs_id);
  auto& contact_fes = *pressure.ParFESpace();
  const int contact_size = contact_fes.GetTrueVSize();
  const int displacement_size = par_fe_space.GetTrueVSize();

  mfem::Vector displacement_true(displacement_size);
  displacement_true = 0.0;
  mfem::HypreParVector pressure_true(&contact_fes);
  pressure_true = 0.0;

  const double disp_increment = opts.prescribed_displacement / opts.num_timesteps;
  tribol::RealT dt = 1.0 / opts.num_timesteps;
  int total_newton_iterations = 0;
  int max_newton_iterations = 0;

  for (int step = 1; step <= opts.num_timesteps; ++step) {
    const double current_prescribed_disp = disp_increment * step;
    for (int i = 0; i < prescribed_tdof_list.Size(); ++i) {
      displacement_true(prescribed_tdof_list[i]) = current_prescribed_disp;
    }

    for (int newton_iter = 0; newton_iter < opts.max_newton_iterations; ++newton_iter) {
      auto& prolongation = *par_fe_space.GetProlongationMatrix();
      prolongation.Mult(displacement_true, displacement);
      coords = ref_coords;
      coords += displacement;

      tribol::updateMfemParallelDecomposition();

      auto& tribol_pressure = tribol::getMfemContactPressure(cs_id);
      tribol_pressure = 0.0;
      tribol_pressure.Add(1.0, pressure_true);

      tribol::update(step, step * dt, dt);

      mfem::Vector elastic_residual(displacement_size);
      elastic_jacobian.get().Mult(displacement_true, elastic_residual);

      auto contact_force = tribol::getMfemContactForce(cs_id);
      mfem::Vector residual_u(elastic_residual);
      residual_u += contact_force;

      auto residual_lambda = tribol::getMfemContactGap(cs_id);

      double residual_u_norm_squared = mfem::InnerProduct(MPI_COMM_WORLD, residual_u, residual_u);
      for (int i = 0; i < ess_tdof_list.Size(); ++i) {
        residual_u_norm_squared -= residual_u(ess_tdof_list[i]) * residual_u(ess_tdof_list[i]);
      }
      const double residual_lambda_norm_squared = mfem::InnerProduct(MPI_COMM_WORLD, residual_lambda, residual_lambda);
      const double residual_norm = std::sqrt(std::abs(residual_u_norm_squared) + residual_lambda_norm_squared);
      const double residual_goal = opts.newton_absolute_tol;
      SLIC_INFO_ROOT("Newton iteration " << newton_iter << " residual norm = " << residual_norm);
      if (newton_iter > 0 && residual_norm <= residual_goal) {
        max_newton_iterations = std::max(max_newton_iterations, newton_iter);
        break;
      }

      auto contact_hessian_ptr = tribol::getMfemDfDx(cs_id);
      auto contact_gradient_transpose_ptr = tribol::getMfemDfDp(cs_id);
      SLIC_ERROR_ROOT_IF(!contact_gradient_transpose_ptr, "Missing EnergyMortar LM contact gradient block.");

      shared::ParSparseMat jacobian_uu = shared::ParSparseMatView(&elastic_jacobian.get()) * 1.0;
      if (contact_hessian_ptr && contact_hessian_ptr->NumRows() > 0) {
        shared::ParSparseMat contact_hessian(std::move(contact_hessian_ptr));
        jacobian_uu = shared::ParSparseMatView(&elastic_jacobian.get()) + shared::ParSparseMatView(&contact_hessian.get());
      }
      shared::ParSparseMat contact_gradient_transpose(std::move(contact_gradient_transpose_ptr));
      applyEssentialBCs(ess_tdof_list, residual_u, jacobian_uu, contact_gradient_transpose);
      shared::ParSparseMat contact_gradient = contact_gradient_transpose.transpose();

      mfem::Array<int> block_offsets(3);
      block_offsets[0] = 0;
      block_offsets[1] = displacement_size;
      block_offsets[2] = displacement_size + contact_size;

      mfem::BlockOperator jacobian(block_offsets);
      jacobian.SetBlock(0, 0, &jacobian_uu.get());
      jacobian.SetBlock(0, 1, &contact_gradient_transpose.get());
      jacobian.SetBlock(1, 0, &contact_gradient.get());

      mfem::BlockVector rhs(block_offsets);
      rhs.GetBlock(0) = residual_u;
      rhs.GetBlock(0).Neg();
      rhs.GetBlock(1) = residual_lambda;
      rhs.GetBlock(1).Neg();

      mfem::BlockVector delta(block_offsets);
      delta = 0.0;

      if (opts.linear_solver == "direct") {
        mfem::Array2D<const mfem::HypreParMatrix*> hypre_blocks(2, 2);
        hypre_blocks(0, 0) = &jacobian_uu.get();
        hypre_blocks(0, 1) = &contact_gradient_transpose.get();
        hypre_blocks(1, 0) = &contact_gradient.get();
        auto pressure_regularization = shared::ParSparseMat::diagonalMatrix(
            MPI_COMM_WORLD, contact_size, contact_fes.GetTrueDofOffsets(), opts.direct_pressure_regularization);
        hypre_blocks(1, 1) = opts.direct_pressure_regularization > 0.0 ? &pressure_regularization.get() : nullptr;
        auto merged_jacobian = std::unique_ptr<mfem::HypreParMatrix>(mfem::HypreParMatrixFromBlocks(hypre_blocks));
#ifdef MFEM_USE_SUPERLU
        mfem::SuperLURowLocMatrix row_matrix(*merged_jacobian);
        mfem::SuperLUSolver solver(MPI_COMM_WORLD);
        solver.SetPrintStatistics(false);
        solver.SetSymmetricPattern(false);
        solver.SetColumnPermutation(mfem::superlu::PARMETIS);
        solver.SetOperator(row_matrix);
        solver.Mult(rhs, delta);
        SLIC_INFO_ROOT("  SuperLU direct solve complete");
#else
        SLIC_ERROR_ROOT("--linear-solver direct requires MFEM_USE_SUPERLU.");
#endif
      } else {
        mfem::MINRESSolver solver(MPI_COMM_WORLD);
        solver.SetRelTol(opts.linear_relative_tol);
        solver.SetAbsTol(opts.linear_absolute_tol);
        solver.SetMaxIter(opts.linear_max_iterations);
        solver.SetPrintLevel(opts.linear_print_level);
        solver.SetOperator(jacobian);
        solver.Mult(rhs, delta);

        SLIC_INFO_ROOT("  MINRES converged: " << solver.GetConverged() << " in " << solver.GetNumIterations()
                                               << " iterations");
      }

      displacement_true += delta.GetBlock(0);
      pressure_true.Add(1.0, delta.GetBlock(1));
      for (int i = 0; i < prescribed_tdof_list.Size(); ++i) {
        displacement_true(prescribed_tdof_list[i]) = current_prescribed_disp;
      }

      ++total_newton_iterations;
      if (newton_iter + 1 == opts.max_newton_iterations) {
        max_newton_iterations = std::max(max_newton_iterations, opts.max_newton_iterations);
      }
    }

    SLIC_INFO_ROOT("Timestep " << step << "/" << opts.num_timesteps
                               << " | prescribed disp = " << current_prescribed_disp);
  }

  auto& prolongation = *par_fe_space.GetProlongationMatrix();
  prolongation.Mult(displacement_true, displacement);
  coords = ref_coords;
  coords += displacement;

  tribol::updateMfemParallelDecomposition();
  auto& final_tribol_pressure = tribol::getMfemContactPressure(cs_id);
  final_tribol_pressure = 0.0;
  final_tribol_pressure.Add(1.0, pressure_true);
  tribol::update(opts.num_timesteps, 1.0, dt);

  const double total_height = 2.0;
  const double eps_yy = opts.prescribed_displacement / total_height;
  const double eps_xx = -opts.lambda / (opts.lambda + 2.0 * opts.mu) * eps_yy;

  mfem::VectorFunctionCoefficient exact_sol_coeff(2, [eps_xx, eps_yy](const mfem::Vector& x, mfem::Vector& u) {
    u[0] = eps_xx * x[0];
    u[1] = eps_yy * x[1];
  });
  exact_disp.ProjectCoefficient(exact_sol_coeff);

  mfem::ParGridFunction error_vec(exact_disp);
  error_vec -= displacement;
  displacement_error = error_vec;
  const double l2_err_vec = mfem::ParNormlp(error_vec, 2, MPI_COMM_WORLD);
  const int num_scalar_dofs = scalar_fes.GetNDofs();

  mfem::ParGridFunction ux_exact(&scalar_fes), ux_num(&scalar_fes);
  mfem::ParGridFunction uy_exact(&scalar_fes), uy_num(&scalar_fes);
  for (int i = 0; i < num_scalar_dofs; ++i) {
    const int ux_vdof = par_fe_space.DofToVDof(i, 0);
    const int uy_vdof = par_fe_space.DofToVDof(i, 1);
    ux_exact(i) = exact_disp(ux_vdof);
    ux_num(i) = displacement(ux_vdof);
    uy_exact(i) = exact_disp(uy_vdof);
    uy_num(i) = displacement(uy_vdof);
  }

  mfem::ParGridFunction ux_err(ux_exact);
  ux_err -= ux_num;
  const double l2_err_x = mfem::ParNormlp(ux_err, 2, MPI_COMM_WORLD);

  mfem::ParGridFunction uy_err(uy_exact);
  uy_err -= uy_num;
  const double l2_err_y = mfem::ParNormlp(uy_err, 2, MPI_COMM_WORLD);

  mfem::Vector exact_displacement_true(displacement_size);
  exact_disp.GetTrueDofs(exact_displacement_true);

  mfem::Vector exact_elastic_residual(displacement_size);
  elastic_jacobian.get().Mult(exact_displacement_true, exact_elastic_residual);
  mfem::Vector exact_contact_force(exact_elastic_residual);
  exact_contact_force.Neg();
  for (int i = 0; i < ess_tdof_list.Size(); ++i) {
    exact_contact_force(ess_tdof_list[i]) = 0.0;
  }

  auto contact_force = tribol::getMfemContactForce(cs_id);
  contact_force_field.SetFromTrueDofs(contact_force);
  exact_contact_force_field.SetFromTrueDofs(exact_contact_force);

  mfem::Vector contact_force_error_true(contact_force);
  contact_force_error_true -= exact_contact_force;
  contact_force_error.SetFromTrueDofs(contact_force_error_true);

  double max_displacement_error = 0.0;
  int max_displacement_error_tdof = -1;
  double max_contact_force_error = 0.0;
  int max_contact_force_error_tdof = -1;

  for (int i = 0; i < num_scalar_dofs; ++i) {
    const int ux_vdof = par_fe_space.DofToVDof(i, 0);
    const int uy_vdof = par_fe_space.DofToVDof(i, 1);
    const double disp_error_x = displacement_error(ux_vdof);
    const double disp_error_y = displacement_error(uy_vdof);
    displacement_l2_error(i) = std::sqrt(disp_error_x * disp_error_x + disp_error_y * disp_error_y);

    const double force_error_x = contact_force_error(ux_vdof);
    const double force_error_y = contact_force_error(uy_vdof);
    contact_force_l2_error(i) = std::sqrt(force_error_x * force_error_x + force_error_y * force_error_y);
  }

  for (int i = 0; i < displacement_size; ++i) {
    const double displacement_error_abs = std::abs(exact_displacement_true(i) - displacement_true(i));
    if (displacement_error_abs > max_displacement_error) {
      max_displacement_error = displacement_error_abs;
      max_displacement_error_tdof = i;
    }

    const double contact_force_error_abs = std::abs(contact_force_error_true(i));
    if (contact_force_error_abs > max_contact_force_error) {
      max_contact_force_error = contact_force_error_abs;
      max_contact_force_error_tdof = i;
    }
  }

  {
    std::ofstream diagnostics(opts.name + "_dof_diagnostics_rank" + std::to_string(mesh.GetMyRank()) + ".csv");
    diagnostics << "true_dof,component,displacement,exact_displacement,displacement_error,contact_force,"
                   "exact_contact_force,contact_force_error\n";
    for (int i = 0; i < displacement_size; ++i) {
      diagnostics << i << ',' << (i < displacement_size / 2 ? 0 : 1) << ',' << displacement_true(i) << ','
                  << exact_displacement_true(i) << ',' << exact_displacement_true(i) - displacement_true(i) << ','
                  << contact_force(i) << ',' << exact_contact_force(i) << ',' << contact_force_error_true(i) << '\n';
    }
  }

  visit_dc.SetCycle(opts.num_timesteps);
  visit_dc.SetTime(1.0);
  visit_dc.Save();

  if (axom::slic::isRoot()) {
    std::cout << "EnergyMortar LM patch options:\n"
              << "  num_elements: " << opts.num_elements << "\n"
              << "  lower_num_elements: " << opts.lower_num_elements << "\n"
              << "  upper_num_elements: " << opts.upper_num_elements << "\n"
              << "  num_steps: " << opts.num_timesteps << "\n"
              << "  eta_smoothing: " << (opts.eta_smoothing ? "true" : "false") << "\n"
              << "  eta_smoothing_type: " << opts.eta_smoothing_type << "\n"
              << "  eta_angle_smoothing_start_angle: " << opts.eta_angle_smoothing_start_angle << "\n"
              << "  projection_smoothing: " << (opts.projection_smoothing ? "true" : "false") << "\n"
              << "  projection_smoothing_curve: " << opts.projection_smoothing_curve << "\n"
              << "  reference_geometry: " << (opts.reference_geometry ? "true" : "false") << "\n"
              << "  linear_solver: " << opts.linear_solver << "\n"
              << "  direct_pressure_regularization: " << opts.direct_pressure_regularization << "\n"
              << "EnergyMortar LM patch summary:\n"
              << "  total_newton_iterations: " << total_newton_iterations << "\n"
              << "  max_newton_iterations: " << max_newton_iterations << "\n"
              << "  l2_error_vector: " << l2_err_vec << "\n"
              << "  l2_error_x: " << l2_err_x << "\n"
              << "  l2_error_y: " << l2_err_y << "\n"
              << "  max_displacement_error: " << max_displacement_error << " at true dof "
              << max_displacement_error_tdof << "\n"
              << "  max_contact_force_error: " << max_contact_force_error << " at true dof "
              << max_contact_force_error_tdof << "\n"
              << "  dof_diagnostics: " << opts.name << "_dof_diagnostics_rank*.csv\n";
  }

  if (opts.check_error && l2_err_vec > opts.error_tolerance) {
    SLIC_ERROR_ROOT("EnergyMortar LM patch error exceeded tolerance " << opts.error_tolerance << ".");
    return 1;
  }

  return 0;
}
