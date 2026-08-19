// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include <gtest/gtest.h>

#include <memory>
#include <utility>
#include <vector>

#include <mpi.h>
#include "mfem.hpp"
#include "axom/sidre.hpp"

#include "smith/differentiable_numerics/nonlinear_block_solver.hpp"
#include "smith/infrastructure/application_manager.hpp"
#include "smith/numerics/block_preconditioner.hpp"
#include "smith/numerics/solver_config.hpp"
#include "smith/physics/mesh.hpp"
#include "smith/physics/state/finite_element_state.hpp"
#include "smith/physics/state/state_manager.hpp"

namespace smith {
namespace {

class StateScaledIdentityProvider : public BlockOperatorProvider {
 public:
  explicit StateScaledIdentityProvider(int size) : size_(size) { rebuild(1.0); }

  void updateForState(const mfem::Vector& state, const mfem::Array<int>& block_offsets) override
  {
    MFEM_VERIFY(block_offsets.Size() == 3, "StateScaledIdentityProvider expects a two-block state");

    double scale = 0.0;
    for (int i = block_offsets[1]; i < block_offsets[2]; ++i) {
      scale += state[i];
    }

    last_scale_ = scale;
    ++update_count_;
    rebuild(scale);
  }

  const mfem::Operator& currentOperator() const override
  {
    MFEM_VERIFY(current_operator_, "StateScaledIdentityProvider has no current operator");
    return *current_operator_;
  }

  double lastScale() const { return last_scale_; }

  int updateCount() const { return update_count_; }

 private:
  void rebuild(double scale)
  {
    auto op = std::make_unique<mfem::SparseMatrix>(size_);
    for (int i = 0; i < size_; ++i) {
      op->Add(i, i, scale);
    }
    op->Finalize();
    current_operator_ = std::move(op);
  }

  int size_ = 0;
  double last_scale_ = 1.0;
  int update_count_ = 0;
  std::unique_ptr<mfem::Operator> current_operator_;
};

class IdentitySolver : public mfem::Solver {
 public:
  void SetOperator(const mfem::Operator& op) override
  {
    height = op.Height();
    width = op.Width();
  }

  void Mult(const mfem::Vector& x, mfem::Vector& y) const override { y = x; }
};

std::unique_ptr<mfem::HypreParMatrix> makeScaledMassMatrix(mfem::ParFiniteElementSpace& space, double scale)
{
  mfem::ParBilinearForm mass(&space);
  mfem::ConstantCoefficient scale_coef(scale);
  mass.AddDomainIntegrator(new mfem::MassIntegrator(scale_coef));
  mass.Assemble();
  mass.Finalize();
  return std::unique_ptr<mfem::HypreParMatrix>(mass.ParallelAssemble());
}

// Verifies the public custom-preconditioner path refreshes state-dependent block operators.
TEST(StateDependentPreconditioner, UpdatesProviderCurrentOperatorFromNonlinearState)
{
  MPI_Comm comm = MPI_COMM_WORLD;
  axom::sidre::DataStore datastore;
  StateManager::reset();
  StateManager::initialize(datastore, "state_dependent_preconditioner");

  mfem::Mesh serial_mesh = mfem::Mesh::MakeCartesian2D(1, 1, mfem::Element::QUADRILATERAL, 1, 1.0, 1.0);
  Mesh mesh(std::move(serial_mesh), "state_dependent_preconditioner_mesh", 0, 0, comm);

  mfem::H1_FECollection fec(1, mesh.mfemParMesh().Dimension());
  mfem::ParFiniteElementSpace space0(&mesh.mfemParMesh(), &fec, 1, smith::ordering);
  mfem::ParFiniteElementSpace space1(&mesh.mfemParMesh(), &fec, 1, smith::ordering);

  auto u0 = std::make_shared<FiniteElementState>(space0, "u0");
  auto u1 = std::make_shared<FiniteElementState>(space1, "u1");
  *u0 = 1.0;
  *u1 = 2.0;

  const double expected_initial_scale = u1->Sum();

  auto provider = std::make_unique<StateScaledIdentityProvider>(u1->Size());
  auto* provider_ptr = provider.get();
  std::vector<BlockProviderOverride> overrides;
  overrides.emplace_back(1, std::move(provider));

  std::vector<std::unique_ptr<mfem::Solver>> sub_solvers;
  sub_solvers.push_back(std::make_unique<IdentitySolver>());
  sub_solvers.push_back(std::make_unique<IdentitySolver>());

  auto preconditioner = std::make_unique<BlockSchurPreconditioner>(std::move(sub_solvers), BlockSchurType::Diagonal,
                                                                   SchurApproxType::Custom, std::move(overrides));

  NonlinearSolverOptions nonlin_opts;
  nonlin_opts.nonlin_solver = NonlinearSolver::Newton;
  nonlin_opts.max_iterations = 1;
  nonlin_opts.print_level = 0;
  nonlin_opts.absolute_tol = 1.0e-14;
  nonlin_opts.relative_tol = 1.0e-14;

  LinearSolverOptions linear_opts;
  linear_opts.linear_solver = LinearSolver::PrecondOnly;
  linear_opts.preconditioner = Preconditioner::None;
  linear_opts.print_level = 0;

  auto block_solver = buildNonlinearBlockSolver(nonlin_opts, linear_opts, mesh, std::move(preconditioner));

  auto residual = [](const std::vector<NonlinearBlockSolverBase::FieldPtr>& states) {
    std::vector<mfem::Vector> residuals;
    residuals.reserve(states.size());
    for (const auto& state : states) {
      residuals.emplace_back(*state);
    }
    return residuals;
  };

  auto jacobian = [&space0, &space1](const std::vector<NonlinearBlockSolverBase::FieldPtr>&) {
    std::vector<std::vector<NonlinearBlockSolverBase::MatrixPtr>> jac(2);
    jac[0].resize(2);
    jac[1].resize(2);
    jac[0][0] = makeScaledMassMatrix(space0, 1.0);
    jac[0][1] = makeScaledMassMatrix(space0, 0.0);
    jac[1][0] = makeScaledMassMatrix(space1, 0.0);
    jac[1][1] = makeScaledMassMatrix(space1, 1.0);
    return jac;
  };

  [[maybe_unused]] auto solution = block_solver->solve({u0, u1}, residual, jacobian);

  ASSERT_GT(provider_ptr->updateCount(), 0);
  EXPECT_NEAR(provider_ptr->lastScale(), expected_initial_scale, 1.0e-12);

  mfem::Vector x(u1->Size());
  mfem::Vector y(u1->Size());
  x = 1.0;
  provider_ptr->currentOperator().Mult(x, y);

  for (int i = 0; i < y.Size(); ++i) {
    EXPECT_NEAR(y[i], expected_initial_scale, 1.0e-12);
  }
}

}  // namespace
}  // namespace smith

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
