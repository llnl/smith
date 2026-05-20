// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <cmath>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "mfem.hpp"

#include "smith/differentiable_numerics/dirichlet_boundary_conditions.hpp"
#include "smith/differentiable_numerics/field_state.hpp"
#include "smith/differentiable_numerics/nonlinear_block_solver.hpp"
#include "smith/differentiable_numerics/nonlinear_solve.hpp"
#include "smith/differentiable_numerics/paraview_writer.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/numerics/equation_solver.hpp"
#include "smith/numerics/solver_config.hpp"
#include "smith/physics/common.hpp"
#include "smith/physics/dfem_solid_weak_form.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/state/state_manager.hpp"

namespace smith {
namespace {

constexpr double length = 10.0;
constexpr double thickness = 0.025;
constexpr double end_tol = 1.0e-8;
constexpr double final_compression = 0.2;
constexpr double seed_down_traction = 1.0e-5;
constexpr double final_snap_up_traction = 0.02;

// Body-force surrogate for a top-face traction t_y over the beam:
//   total surface force = t_y * length
//   equivalent uniform body-force = t_y * length / Volume = t_y / thickness
// Used because the dfem LocalQFBackend boundary path is not entity-aware in the
// current vendored MFEM snapshot (see dfem_plan.md Priority 1). Body force is
// distributed through-thickness rather than only on the top face but produces
// the same total vertical load — sufficient to drive snap-through.
constexpr double seed_down_body_force = seed_down_traction / thickness;
constexpr double final_snap_up_body_force = final_snap_up_traction / thickness;
struct DfemNeoHookean {
  static constexpr int dim = 2;

  template <typename T, int dim_>
  SMITH_HOST_DEVICE auto pkStress(double, const mfem::future::tensor<T, dim_, dim_>& du_dX,
                                  const mfem::future::tensor<T, dim_, dim_>&) const
  {
    using std::log1p;
    auto I = mfem::future::IdentityMatrix<dim_>();
    const auto lambda = K - (2.0 / 3.0) * G;
    const auto B_minus_I = mfem::future::dot(du_dX, mfem::future::transpose(du_dX)) +
                           mfem::future::transpose(du_dX) + du_dX;
    const auto logJ = log1p(mfem::future::detApIm1(du_dX));
    const auto TK = lambda * logJ * I + G * B_minus_I;
    const auto F = du_dX + I;
    return mfem::future::tuple{mfem::future::dot(TK, mfem::future::inv(mfem::future::transpose(F)))};
  }

  double K;
  double G;
};

struct DfemShallowArchFixture : public testing::Test {
  static constexpr int dim = 2;
  static constexpr int order = 1;

  void SetUp() override
  {
    StateManager::reset();
    StateManager::initialize(sidre_store, "dfem_shallow_arch_buckling");
    graph_store = std::make_unique<gretl::DataStore>(100);

    mesh = std::make_shared<Mesh>(
        mfem::Mesh::MakeCartesian2D(120, 5, mfem::Element::QUADRILATERAL, true, length, thickness),
        "dfem_shallow_arch_mesh", 0, 0);

    mesh->addDomainOfBoundaryElements("left_end",
                                      [](std::vector<vec2> vertices, int) { return average(vertices)[0] < end_tol; });
    mesh->addDomainOfBoundaryElements("right_end", [](std::vector<vec2> vertices, int) {
      return average(vertices)[0] > length - end_tol;
    });
  }

  void TearDown() override { StateManager::reset(); }

  axom::sidre::DataStore sidre_store;
  std::unique_ptr<gretl::DataStore> graph_store;
  std::shared_ptr<Mesh> mesh;
};

TEST_F(DfemShallowArchFixture, NeoHookeanSnapThroughWritesParaview)
{
  using VectorSpace = H1<order, dim>;

  auto disp = std::make_shared<FiniteElementState>(StateManager::newState(VectorSpace{}, "displacement", mesh->tag()));
  auto velo = std::make_shared<FiniteElementState>(StateManager::newState(VectorSpace{}, "velocity", mesh->tag()));
  auto accel = std::make_shared<FiniteElementState>(StateManager::newState(VectorSpace{}, "acceleration", mesh->tag()));
  auto coords = std::make_shared<FiniteElementState>(
      *static_cast<mfem::ParGridFunction*>(mesh->mfemParMesh().GetNodes())->ParFESpace(), "coordinates");
  auto shape_disp = std::make_shared<FiniteElementState>(mesh->newShapeDisplacement());

  disp->UseDevice(true);
  velo->UseDevice(true);
  accel->UseDevice(true);
  coords->UseDevice(true);
  shape_disp->UseDevice(true);

  *disp = 0.0;
  *velo = 0.0;
  *accel = 0.0;
  *shape_disp = 0.0;
  coords->setFromGridFunction(*static_cast<mfem::ParGridFunction*>(mesh->mfemParMesh().GetNodes()));

  auto weak_form = std::make_shared<DfemSolidWeakForm>("dfem_shallow_arch", mesh, disp->space());
  const auto& ir = mfem::IntRules.Get(disp->space().GetFE(0)->GetGeomType(), 2);
  weak_form->setMaterial(mesh->mfemParMesh().attributes, DfemNeoHookean{.K = 100.0, .G = 10.0}, ir);
  weak_form->setBodyForce<dim>(
      mesh->mfemParMesh().attributes,
      [](double t) {
        mfem::future::tensor<double, dim> b{};
        if (t < 0.5) {
          b[1] = -seed_down_body_force * (t / 0.5);
        } else {
          const double snap_ramp = (t - 0.5) / 0.5;
          b[1] = -seed_down_body_force * (1.0 - snap_ramp) + final_snap_up_body_force * snap_ramp;
        }
        return b;
      },
      ir);

  auto disp_bc = DirichletBoundaryConditions(*mesh, disp->space());
  disp_bc.setFixedVectorBCs<dim>(mesh->domain("left_end"));
  disp_bc.setVectorBCs<dim>(mesh->domain("right_end"), std::vector<int>{0}, [](double t, tensor<double, dim>) {
    tensor<double, dim> u{};
    u[0] = -final_compression * t;
    return u;
  });
  disp_bc.setFixedVectorBCs<dim>(mesh->domain("right_end"), 1);

  // TrustRegion + CG + HypreJacobi mirrors physics/tests/shallow_arch_buckling.cpp
  // (the functional reference). Newton + GMRES did not converge under
  // buckling-scale loads; the symmetric tangent on this problem makes CG with a
  // diagonal preconditioner cheap and robust, and TrustRegion is required to
  // handle the negative-eigenvalue branch through snap-through.
  auto solver = buildNonlinearBlockSolver(
      NonlinearSolverOptions{.nonlin_solver = NonlinearSolver::TrustRegion,
                             .relative_tol = 1.0e-8,
                             .absolute_tol = 1.0e-10,
                             .max_iterations = 300,
                             .print_level = 1,
                             .subspace_option = SubSpaceOptions::NEVER,
                             .num_leftmost = 1},
      LinearSolverOptions{.linear_solver = LinearSolver::CG,
                          .preconditioner = Preconditioner::HypreJacobi,
                          .relative_tol = 1.0e-8,
                          .absolute_tol = 1.0e-14,
                          .max_iterations = 100000,
                          .print_level = 0},
      *mesh);

  auto shape = createFieldState(*graph_store, shape_disp);
  std::vector<FieldState> states = {createFieldState(*graph_store, disp), createFieldState(*graph_store, velo),
                                    createFieldState(*graph_store, accel), createFieldState(*graph_store, coords)};

  auto pv_writer =
      createParaviewWriter(mesh->mfemParMesh(), getConstFieldPointers(states), "paraview_dfem_shallow_arch_buckling");
  pv_writer.write(0, 0.0, getConstFieldPointers(states));

  constexpr int num_steps = 5;
  for (int step = 0; step < num_steps; ++step) {
    const double time = static_cast<double>(step + 1) / num_steps;
    states[0] = solve(*weak_form, shape, states, {}, TimeInfo(time, 1.0 / num_steps, static_cast<size_t>(step + 1)),
                      *solver, disp_bc, 0);
    pv_writer.write(static_cast<size_t>(step + 1), time, getConstFieldPointers(states));
  }

  EXPECT_GT(states[0].get()->Norml2(), 1.0e-4);
}

}  // namespace
}  // namespace smith

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv, MPI_COMM_WORLD, true, smith::ExecutionSpace::CPU);
  return RUN_ALL_TESTS();
}
