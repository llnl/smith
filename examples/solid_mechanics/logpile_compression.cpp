// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "axom/slic.hpp"
#include "mfem.hpp"
#include "smith/smith.hpp"

namespace {

struct Flexible80AGent {
  using State = smith::Empty;

  SMITH_HOST_DEVICE double shearModulus() const { return youngs_modulus / (2.0 * (1.0 + poisson_ratio)); }

  SMITH_HOST_DEVICE double bulkModulus() const { return youngs_modulus / (3.0 * (1.0 - 2.0 * poisson_ratio)); }

  template <typename T, int dim>
  SMITH_HOST_DEVICE auto operator()(State&, const smith::tensor<T, dim, dim>& du_dX) const
  {
    static_assert(dim == 3, "The logpile example requires a three-dimensional mesh.");
    using std::log1p;
    using std::pow;

    constexpr auto I = smith::DenseIdentity<dim>();
    auto F = I + du_dX;
    auto J = smith::det(F);
    auto nan_stress = (youngs_modulus * static_cast<double>(NAN)) * F;
    if ((J <= 0.0) || (youngs_modulus <= 0.0) || (poisson_ratio <= -1.0) || (poisson_ratio >= 0.5)
        || (limiting_chain_extensibility <= 0.0)) {
      return nan_stress;
    }

    auto F_bar = pow(J, -1.0 / 3.0) * F;
    auto I1_bar = smith::inner(F_bar, F_bar);
    auto locking_margin = 1.0 - (I1_bar - 3.0) / limiting_chain_extensibility;
    if (locking_margin <= 0.0) {
      return nan_stress;
    }

    auto deviatoric_stress = (shearModulus() / locking_margin) * pow(J, -1.0 / 3.0)
                             * (F_bar - (I1_bar / 3.0) * smith::inv(smith::transpose(F_bar)));
    auto volumetric_stress = bulkModulus() * log1p(smith::detApIm1(du_dX)) * smith::inv(smith::transpose(F));
    return deviatoric_stress + volumetric_stress;
  }

  double density;
  double youngs_modulus;
  double poisson_ratio;
  double limiting_chain_extensibility;
};

}  // namespace

int main(int argc, char* argv[])
{
  smith::ApplicationManager application_manager(argc, argv);

  constexpr int order = 1;
  constexpr int dim = 3;
  constexpr int num_steps = 50;
  constexpr double compression = 0.5;

  const std::string simulation_name = "logpile_compression";
  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, simulation_name + "_data");

  const std::string mesh_filename = SMITH_REPO_DIR "/data/meshes/small_logpile_with_plates_001.g";
  auto input_mesh = smith::buildMeshFromFile(mesh_filename);
  mfem::Vector min_corner;
  mfem::Vector max_corner;
  input_mesh.GetBoundingBox(min_corner, max_corner);

  auto mesh = std::make_shared<smith::Mesh>(std::move(input_mesh), simulation_name + "_mesh", 0, 0);

  const double initial_height = max_corner(1) - min_corner(1);
  const double boundary_tolerance = 1.0e-12 * initial_height;
  auto on_y_face = [boundary_tolerance](double y) {
    return [boundary_tolerance, y](std::vector<smith::vec3> vertices, int) {
      return std::all_of(vertices.begin(), vertices.end(), [boundary_tolerance, y](const auto& vertex) {
        return std::abs(vertex[1] - y) <= boundary_tolerance;
      });
    };
  };
  mesh->addDomainOfBoundaryElements("bottom_plate", on_y_face(min_corner(1)));
  mesh->addDomainOfBoundaryElements("top_plate", on_y_face(max_corner(1)));

  auto linear_options = smith::solid_mechanics::default_linear_options;
  linear_options.relative_tol = 1.0e-9;
  linear_options.absolute_tol = 1.0e-12;
  linear_options.max_iterations = 5000;

  smith::NonlinearSolverOptions nonlinear_options{.nonlin_solver = smith::NonlinearSolver::TrustRegion,
                                                  .relative_tol = 1.0e-7,
                                                  .absolute_tol = 1.0e-8,
                                                  .max_iterations = 200,
                                                  .print_level = 1,
                                                  .warm_start = true};

  smith::SolidMechanics<order, dim> solid_solver(nonlinear_options, linear_options,
                                                  smith::solid_mechanics::default_quasistatic_options,
                                                  simulation_name, mesh);

  constexpr double youngs_modulus = 0.5 * 9.132e6;
  constexpr double poisson_ratio = 0.475;
  constexpr double limiting_chain_extensibility = 104.14;
  solid_solver.setMaterial(Flexible80AGent{.density = 1.0,
                                           .youngs_modulus = youngs_modulus,
                                           .poisson_ratio = poisson_ratio,
                                           .limiting_chain_extensibility = limiting_chain_extensibility},
                           mesh->entireBody());

  solid_solver.setFixedBCs(mesh->domain("bottom_plate"));
  auto compress_top = [initial_height](smith::vec3, double time) {
    smith::vec3 displacement{};
    displacement[1] = -compression * initial_height * time;
    return displacement;
  };
  solid_solver.setDisplacementBCs(compress_top, mesh->domain("top_plate"));

  solid_solver.completeSetup();

  const std::string paraview_name = simulation_name + "_paraview";
  solid_solver.outputStateToDisk(paraview_name);

  constexpr double dt = 1.0 / num_steps;
  for (int step = 0; step < num_steps; ++step) {
    SLIC_INFO_ROOT("Advancing logpile compression step " << step + 1 << " of " << num_steps);
    solid_solver.advanceTimestep(dt);
    solid_solver.outputStateToDisk(paraview_name);
  }

  return 0;
}
