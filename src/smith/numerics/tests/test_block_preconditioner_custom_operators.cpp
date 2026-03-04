#include <gtest/gtest.h>
#include <mpi.h>

#include "mfem.hpp"
#include "smith/numerics/block_preconditioner.hpp"
#include "smith/infrastructure/application_manager.hpp"

using namespace mfem;

/* ============================================================
   Helper utilities
   ============================================================ */

// Build c * I
std::unique_ptr<SparseMatrix> makeScaledIdentity(int n, double c)
{
  auto A = std::make_unique<SparseMatrix>(n);
  for (int i = 0; i < n; i++) { A->Add(i, i, c); }
  A->Finalize();
  return A;
}

// Simple SPD matrix: tridiagonal
std::unique_ptr<SparseMatrix> makeSPDMatrix(int n)
{
  auto A = std::make_unique<SparseMatrix>(n);
  for (int i = 0; i < n; i++)
  {
    A->Add(i, i, 2.0);
    if (i > 0)     A->Add(i, i-1, -1.0);
    if (i < n - 1) A->Add(i, i+1, -1.0);
  }
  A->Finalize();
  return A;
}

class IdentitySolver : public mfem::Solver
{
public:
  IdentitySolver() = default;

  void SetOperator(const mfem::Operator &op) override
  {
    height = op.Height();
    width  = op.Width();
  }

  void Mult(const mfem::Vector &x, mfem::Vector &y) const override
  {
    y = x;
  }
};

// Exact diagonal inverse solver
class ExactDiagonalSolver : public mfem::Solver
{
public:
  ExactDiagonalSolver() = default;

  void SetOperator(const mfem::Operator &op) override
  {
    const auto *A = dynamic_cast<const mfem::SparseMatrix *>(&op);
    MFEM_VERIFY(A, "ExactDiagonalSolver requires SparseMatrix");

    A_ = A;
    height = A_->Height();
    width  = A_->Width();

    diag_.SetSize(height);
    A_->GetDiag(diag_);
  }

  void Mult(const mfem::Vector &x, mfem::Vector &y) const override
  {
    MFEM_ASSERT(A_, "Operator not set");

    y.SetSize(x.Size());
    for (int i = 0; i < x.Size(); i++)
    {
      y[i] = x[i] / diag_[i];
    }
  }

private:
  const mfem::SparseMatrix *A_ = nullptr;
  mfem::Vector diag_;
};

std::vector<std::unique_ptr<Solver>>
makeExactDiagonalSolvers()
{
  std::vector<std::unique_ptr<Solver>> solvers;
  solvers.push_back(std::make_unique<ExactDiagonalSolver>());
  solvers.push_back(std::make_unique<ExactDiagonalSolver>());
  return solvers;
}

// Tridiagonal rectangular sparse matrix with constant main diagonal 2.0, off-diagonals -1.0
std::unique_ptr<SparseMatrix> makeRectTridiagonal(int rows, int cols)
{
  auto A = std::make_unique<SparseMatrix>(rows, cols);

  for (int i = 0; i < rows; i++)
  {
    if (i < cols)         A->Add(i, i, 2.0);       // main diagonal
    if (i > 0 && i-1 < cols) A->Add(i, i-1, -1.0); // lower diagonal
    if (i+1 < cols)       A->Add(i, i+1, -1.0);   // upper diagonal
  }

  A->Finalize();
  return A;
}

/* ============================================================
   Tests 
   ============================================================ */

// Makes sure an error is thrown when length of solvers does not match the
// number of blocks

// If the solver for each block is identity, the block solver is identity
TEST(BlockDiagonalPreconditionerCustom, IdentityActsAsIdentity)
{
  Array<int> offsets({0, 2, 5});

  auto A11 = makeScaledIdentity(2, 2.0);
  auto A22 = makeScaledIdentity(3, 2.0);

  BlockOperator A(offsets);
  A.SetBlock(0,0, A11.get());
  A.SetBlock(1,1, A22.get());

  std::vector<std::unique_ptr<Solver>> solvers;
  solvers.push_back(std::make_unique<ExactDiagonalSolver>());
  solvers.push_back(std::make_unique<ExactDiagonalSolver>());

  // Define custom operators to be used in the preconditioner
  auto M1u = makeScaledIdentity(2, 1.0);  // unique_ptr<mfem::SparseMatrix>
  auto M2u = makeScaledIdentity(3, 1.0);

  std::shared_ptr<const mfem::Operator> M1(std::move(M1u));
  std::shared_ptr<const mfem::Operator> M2(std::move(M2u));

  smith::BlockDiagonalPreconditioner P(offsets, std::move(solvers),
                                            { {0, M1}, {1, M2} });

  P.SetOperator(A); // This actually doesn't use A. It's overidden by M1, M2

  Vector x(5), y(5);
  x.Randomize();

  P.Mult(x, y);

  mfem::Vector diff(x);
  diff -= y;

  EXPECT_NEAR(diff.Norml2(), 0.0, 1e-14);
}

/* ============================================================
   main
   ============================================================ */

int main(int argc, char *argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
