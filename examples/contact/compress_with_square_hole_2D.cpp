// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "axom/slic.hpp"
#include "axom/slic/core/SimpleLogger.hpp"
#include "mfem.hpp"

#include "smith/infrastructure/application_manager.hpp"
#include "smith/numerics/solver_config.hpp"
#include "smith/physics/contact/contact_config.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/solid_mechanics_contact.hpp"
#include "smith/physics/state/state_manager.hpp"
#include "smith/smith.hpp"
#include "smith/smith_config.hpp"

namespace {

struct SolidNeoHookeanMaterial {
  using State = smith::Empty;

  template <int dim, typename DispGradType>
  SMITH_HOST_DEVICE auto operator()(State&, const smith::tensor<DispGradType, dim, dim>& du_dX) const
  {
    using std::log1p;

    constexpr auto I = smith::Identity<dim>();

    auto F = I + du_dX;
    auto J = det(F);
    if (J <= 0.0) {
      return static_cast<double>(NAN) * F;
    }

    auto logJ = log1p(detApIm1(du_dX));
    auto FinvT = inv(transpose(F));
    auto lambda = K - (2.0 / dim) * G;
    auto B_minus_I = du_dX * transpose(du_dX) + transpose(du_dX) + du_dX;
    auto TK = lambda * logJ * I + G * B_minus_I;

    return dot(TK, FinvT);
  }

  double density{};
  double K{};
  double G{};
};

}  // namespace

int main(int argc, char* argv[])
{
  smith::ApplicationManager application_manager(argc, argv);

  constexpr int p = 1;
  constexpr int dim = 2;
  constexpr int bottom_attr = 3;
  constexpr int right_attr = 4;
  constexpr int top_attr = 1;
  constexpr int left_attr = 2;
  constexpr int middle_attr = 5;

  std::string mesh_file = SMITH_REPO_DIR "/data/meshes/single_square_hole_frame.g";
  std::string output_dir = ".";
  double target_strain = 0.85;
  int num_time_steps = 125;
  double final_pseudo_time = 1.0;
  double penalty = 5e9;

  double K_solid = 7.3e6;
  double nu = 0.33;
  double E_solid = 3.0 * K_solid * (1.0 - 2.0 * nu);
  double G_solid = E_solid / (2.0 * (1.0 + nu));

  axom::CLI::App app{"2D square-hole frame compression with interior self-contact"};
  app.add_option("--mesh-file", mesh_file, "Mesh file to load");
  app.add_option("--output-dir", output_dir, "Output directory");
  app.add_option("--target-strain", target_strain);
  app.add_option("--num-time-steps", num_time_steps);
  app.add_option("--final-pseudo-time", final_pseudo_time);
  app.add_option("--penalty", penalty);
  CLI11_PARSE(app, argc, argv);

  if (!std::filesystem::exists(mesh_file)) {
    SLIC_ERROR_ROOT("Missing mesh file: " << mesh_file << ". Generate it with: cubitx --nogui -batch "
                                          << SMITH_REPO_DIR "/data/meshes/single_square_hole_frame.jou");
  }

  std::filesystem::create_directories(output_dir);

  axom::sidre::DataStore datastore;
  const std::string name = "RyansCompressionReactionSingleSquareHole2D";
  smith::StateManager::initialize(datastore, name + "_data");

  auto mesh = std::make_shared<smith::Mesh>(mesh_file, "single_square_hole_frame", 0, 0);
  mesh->mfemParMesh().CheckElementOrientation(true);

  if (mesh->mfemParMesh().GetNodes() == nullptr) {
    mesh->mfemParMesh().SetCurvature(1, false, -1, 0);
  }

  mesh->addDomainOfBodyElements("solid", smith::by_attr<dim>(1));
  mesh->addDomainOfBoundaryElements("bottom", smith::by_attr<dim>(bottom_attr));
  mesh->addDomainOfBoundaryElements("right", smith::by_attr<dim>(right_attr));
  mesh->addDomainOfBoundaryElements("top", smith::by_attr<dim>(top_attr));
  mesh->addDomainOfBoundaryElements("left", smith::by_attr<dim>(left_attr));
  mesh->addDomainOfBoundaryElements("interior", smith::by_attr<dim>(middle_attr));

  mfem::Vector min_corner;
  mfem::Vector max_corner;
  mesh->mfemParMesh().GetBoundingBox(min_corner, max_corner);
  const double specimen_height = max_corner[1] - min_corner[1];
  const double target_top_displacement = -target_strain * specimen_height;
  const double dt = final_pseudo_time / static_cast<double>(num_time_steps);

  smith::NonlinearSolverOptions nonlinear_options{
      .nonlin_solver = smith::NonlinearSolver::TrustRegion,
      .relative_tol = 1.0e-8,
      .absolute_tol = 1.0e-9,
      .max_iterations = 2000,
      .max_line_search_iterations = 10,
      .print_level = 1,
  };

  smith::LinearSolverOptions linear_options{.linear_solver = smith::LinearSolver::CG,
                                            .preconditioner = smith::Preconditioner::HypreAMG,
                                            .relative_tol = 1.0e-9,
                                            .absolute_tol = 1.0e-10,
                                            .max_iterations = 10000,
                                            .print_level = 0};

  smith::ContactOptions contact_options{.method = smith::ContactMethod::EnergyAreaPenalty,
                                        .enforcement = smith::ContactEnforcement::Penalty,
                                        .type = smith::ContactType::Frictionless,
                                        .penalty = penalty,
                                        .penalty2 = 0.0,
                                        .jacobian = smith::ContactJacobian::Exact,
                                        .penalty_smoothing = smith::PenaltySmoothing::Smooth};

  smith::SolidMechanicsContact<p, dim> solid_solver(nonlinear_options, linear_options,
                                                    smith::solid_mechanics::default_quasistatic_options, name, mesh,
                                                    std::vector<std::string>{}, 0, 0, false, true);

  SolidNeoHookeanMaterial solid_material{.density = 1000.0, .K = K_solid, .G = G_solid};
  solid_solver.setMaterial(solid_material, mesh->domain("solid"));

  auto applied_displacement = [=](const smith::tensor<double, dim>&, double t) {
    smith::tensor<double, dim> u{0.0, 0.0};
    const double load_factor = std::clamp((final_pseudo_time > 0.0) ? (t / final_pseudo_time) : 1.0, 0.0, 1.0);
    u[1] = target_top_displacement * load_factor;
    return u;
  };

  solid_solver.setDisplacementBCs(applied_displacement, mesh->domain("top"));
  solid_solver.setFixedBCs(mesh->domain("bottom"));
  solid_solver.addContactInteraction(1, {middle_attr}, {middle_attr}, contact_options);
  solid_solver.addContactInteraction(2, {left_attr}, {left_attr}, contact_options);
  solid_solver.addContactInteraction(3, {right_attr}, {right_attr}, contact_options);

  solid_solver.completeSetup();

  const std::string visit_name = (std::filesystem::path(output_dir) / name).string();
  solid_solver.outputStateToDisk(visit_name);

  for (int step = 0; step < num_time_steps; ++step) {
    solid_solver.advanceTimestep(dt);
    solid_solver.outputStateToDisk(visit_name);
  }

  return 0;
}
