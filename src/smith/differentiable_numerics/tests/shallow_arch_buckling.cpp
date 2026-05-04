// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "mpi.h"
#include "mfem.hpp"

#include "smith/infrastructure/application_manager.hpp"
#include "smith/numerics/functional/domain.hpp"
#include "smith/numerics/functional/tensor.hpp"
#include "smith/numerics/solver_config.hpp"
#include "smith/differentiable_numerics/time_info_solid_materials.hpp"
#include "smith/differentiable_numerics/paraview_writer.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/state/state_manager.hpp"
#include "smith/differentiable_numerics/solid_mechanics_system.hpp"
#include "smith/differentiable_numerics/multiphysics_time_integrator.hpp"

namespace smith {
namespace {

constexpr double length = 10.0;
constexpr double thickness = 0.25;
constexpr double end_tol = 1.0e-8;
constexpr double top_tol = 1.0e-8;

std::string solver_name = "TrustRegion";
int print_level = 2;
int nonlinear_max_iters = 30000;
bool warm_start = false;

NonlinearSolver selectedNonlinearSolver()
{
  if (solver_name == "NewtonLineSearch") return NonlinearSolver::NewtonLineSearch;
  if (solver_name == "TrustRegion") return NonlinearSolver::TrustRegion;
  throw std::runtime_error("Unknown --solver '" + solver_name + "'. Use NewtonLineSearch or TrustRegion.");
}

void parseCommandLine(int& argc, char** argv)
{
  int write_arg = 1;
  for (int read_arg = 1; read_arg < argc; ++read_arg) {
    const std::string arg = argv[read_arg];
    if (arg.rfind("--solver=", 0) == 0)
      solver_name = arg.substr(9);
    else if (arg.rfind("--print-level=", 0) == 0)
      print_level = std::stoi(arg.substr(14));
    else if (arg.rfind("--warm-start=", 0) == 0)
      warm_start = std::stoi(arg.substr(14));
    else if (arg.rfind("--nonlinear-max-iterations=", 0) == 0)
      nonlinear_max_iters = std::stoi(arg.substr(27));
    else {
      argv[write_arg++] = argv[read_arg];
    }
  }
  argc = write_arg;
}

}  // namespace

TEST(ShallowArchBuckling, CompressedThinBeamSnapThrough)
{
  MPI_Barrier(MPI_COMM_WORLD);

  constexpr int dim = 2;
  constexpr int p = 1;
  constexpr int nx = 96;
  constexpr int ny = 4;

  axom::sidre::DataStore datastore;
  smith::StateManager::initialize(datastore, "shallow_arch_buckling");

  auto mesh = std::make_shared<smith::Mesh>(
      mfem::Mesh::MakeCartesian2D(nx, ny, mfem::Element::QUADRILATERAL, true, length, thickness),
      "compressed_beam_mesh", 0, 0);

  mesh->addDomainOfBoundaryElements("left_end", [](std::vector<vec2> v, int) { return average(v)[0] < end_tol; });
  mesh->addDomainOfBoundaryElements("right_end",
                                    [](std::vector<vec2> v, int) { return average(v)[0] > length - end_tol; });
  mesh->addDomainOfBoundaryElements("top_face",
                                    [](std::vector<vec2> v, int) { return average(v)[1] > thickness - top_tol; });

  EXPECT_GT(mesh->domain("left_end").total_elements(), 0);
  EXPECT_GT(mesh->domain("right_end").total_elements(), 0);
  EXPECT_GT(mesh->domain("top_face").total_elements(), 0);

  smith::LinearSolverOptions linear_options{.linear_solver = LinearSolver::CG,
                                            .preconditioner = Preconditioner::HypreJacobi,
                                            .relative_tol = 1.0e-8,
                                            .absolute_tol = 1.0e-14,
                                            .max_iterations = 10000,
                                            .print_level = 0};

  smith::NonlinearSolverOptions nonlinear_options{.nonlin_solver = selectedNonlinearSolver(),
                                                  .relative_tol = 1.0e-8,
                                                  .absolute_tol = 1.0e-10,
                                                  .max_iterations = nonlinear_max_iters,
                                                  .print_level = print_level,
                                                  .warm_start = warm_start};

  using TimeRule = QuasiStaticSecondOrderTimeIntegrationRule;
  auto field_store = std::make_shared<FieldStore>(mesh, 100, "arch_");
  auto solid_system = buildSolidMechanicsSystem<dim, p, TimeRule>(nonlinear_options, linear_options,
                                                                  SolidMechanicsOptions{}, field_store);

  solid_system->setMaterial(solid_mechanics::TimeInfoNeoHookean{.density = 1.0, .K = 100.0, .G = 10.0},
                            mesh->entireBodyName());

  // Left end: fully clamped.
  solid_system->setDisplacementBC(mesh->domain("left_end"));

  // Right end: X prescribed (compression ramp), Y fixed to zero.
  constexpr double final_compression = 0.2;
  solid_system->setDisplacementBC(
      mesh->domain("right_end"),
      [](double t, tensor<double, dim> /*X*/) -> tensor<double, dim> { return {-final_compression * t, 0.0}; });

  // Vertical traction on top face seeds the out-of-plane buckling mode,
  // then reverses to drive snap-through.
  constexpr double seed_down_traction = 1.0e-5;
  constexpr double final_snap_up_traction = 0.02;
  solid_system->addTraction("top_face",
                            [](double t, auto /*X*/, auto /*n*/, auto /*u*/, auto /*v*/, auto /*a*/,
                               auto... /*params*/) -> tensor<double, dim> {
                              double fy;
                              if (t < 0.5) {
                                fy = -seed_down_traction * (t / 0.5);
                              } else {
                                const double snap_ramp = (t - 0.5) / 0.5;
                                fy = -seed_down_traction * (1.0 - snap_ramp) + final_snap_up_traction * snap_ramp;
                              }
                              return {0.0, fy};
                            });

  BcRampOptions ramp_opts;
  ramp_opts.enabled = true;
  solid_system->solver->setBcRampOptions(ramp_opts);

  auto advancer = makeAdvancer(solid_system);
  auto shape_disp = solid_system->field_store->getShapeDisp();
  auto states = solid_system->field_store->getStateFields();
  auto params = solid_system->field_store->getParameterFields();

  auto pv_fields = solid_system->field_store->getOutputFieldStates();
  auto pv_writer = createParaviewWriter(*mesh, pv_fields, "paraview_shallow_arch");
  pv_writer.write(0, 0.0, pv_fields);

  constexpr int num_steps = 1;
  constexpr double dt = 1.0 / num_steps;
  double time = 0.0;
  size_t cycle = 0;

  mfem::out << "Compressed thin beam snap-through: solver = " << solver_name << '\n';

  for (int step = 0; step < num_steps; ++step) {
    TimeInfo t_info(time, dt, cycle);
    std::vector<ReactionState> reactions;
    std::tie(states, reactions) = advancer->advanceState(t_info, shape_disp, states, params);
    pv_fields = solid_system->field_store->getOutputFieldStates();
    time += dt;
    cycle++;
    pv_writer.write(cycle, time, pv_fields);
    mfem::out << "Load step " << step + 1 << "/" << num_steps << " completed (t=" << time << ")\n";
  }

  // Displacement of the beam mid-point should be non-trivial after snap-through.
  double disp_norm = norm(*states[0].get());
  EXPECT_GT(disp_norm, 1.0e-4) << "Expected non-trivial displacement after snap-through.";
}

}  // namespace smith

int main(int argc, char* argv[])
{
  smith::parseCommandLine(argc, argv);
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
