#include <gtest/gtest.h>
#include <mpi.h>

#include "mfem.hpp"
#include "smith/numerics/block_preconditioner.hpp"
#include "smith/infrastructure/application_manager.hpp"

#include "axom/slic.hpp"

bool abort_called = false;

void testAbortHandler()
{
  abort_called = true;  // record abort instead of exiting
}

using namespace mfem;

/* ============================================================
   Helper utilities
   ============================================================ */

// Build c * I
std::unique_ptr<SparseMatrix> makeScaledIdentity(int n, double c)
{
  auto A = std::make_unique<SparseMatrix>(n);
  for (int i = 0; i < n; i++) {
    A->Add(i, i, c);
  }
  A->Finalize();
  return A;
}

// Simple SPD matrix: tridiagonal
std::unique_ptr<SparseMatrix> makeSPDMatrix(int n)
{
  auto A = std::make_unique<SparseMatrix>(n);
  for (int i = 0; i < n; i++) {
    A->Add(i, i, 2.0);
    if (i > 0) A->Add(i, i - 1, -1.0);
    if (i < n - 1) A->Add(i, i + 1, -1.0);
  }
  A->Finalize();
  return A;
}

class IdentitySolver : public mfem::Solver {
 public:
  IdentitySolver() = default;

  void SetOperator(const mfem::Operator& op) override
  {
    height = op.Height();
    width = op.Width();
  }

  void Mult(const mfem::Vector& x, mfem::Vector& y) const override { y = x; }
};

// Exact diagonal inverse solver
class ExactDiagonalSolver : public mfem::Solver {
 public:
  ExactDiagonalSolver() = default;

  void SetOperator(const mfem::Operator& op) override
  {
    const auto* A = dynamic_cast<const mfem::SparseMatrix*>(&op);
    MFEM_VERIFY(A, "ExactDiagonalSolver requires SparseMatrix");

    A_ = A;
    height = A_->Height();
    width = A_->Width();

    diag_.SetSize(height);
    A_->GetDiag(diag_);
  }

  void Mult(const mfem::Vector& x, mfem::Vector& y) const override
  {
    MFEM_ASSERT(A_, "Operator not set");

    y.SetSize(x.Size());
    for (int i = 0; i < x.Size(); i++) {
      y[i] = x[i] / diag_[i];
    }
  }

 private:
  const mfem::SparseMatrix* A_ = nullptr;
  mfem::Vector diag_;
};

std::vector<std::unique_ptr<Solver>> makeExactDiagonalSolvers()
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

  for (int i = 0; i < rows; i++) {
    if (i < cols) A->Add(i, i, 2.0);                    // main diagonal
    if (i > 0 && i - 1 < cols) A->Add(i, i - 1, -1.0);  // lower diagonal
    if (i + 1 < cols) A->Add(i, i + 1, -1.0);           // upper diagonal
  }

  A->Finalize();
  return A;
}

/* ============================================================
   Tests
   ============================================================ */

// Makes sure an error is thrown when length of solvers does not match the
// number of blocks
TEST(BlockDiagonal, ThrowsOnWrongNumberOfSolvers)
{
  // Replace abort handler
  axom::slic::setAbortFunction(testAbortHandler);

  abort_called = false;

  Array<int> offsets({0, 2, 4});
  std::vector<std::unique_ptr<Solver>> solvers;
  solvers.push_back(std::make_unique<IdentitySolver>());

  std::cout << abort_called << std::endl;
  smith::BlockDiagonalPreconditioner P(offsets, std::move(solvers));
  std::cout << abort_called << std::endl;

  EXPECT_TRUE(abort_called);
}

// If the solver for each block is identity, the block solver is identity
TEST(BlockTriangular, IdentityActsAsIdentity)
{
  Array<int> offsets({0, 2, 5});

  auto A11 = makeScaledIdentity(2, 1.0);
  auto A22 = makeScaledIdentity(3, 1.0);

  BlockOperator A(offsets);
  A.SetBlock(0, 0, A11.get());
  A.SetBlock(1, 1, A22.get());

  std::vector<std::unique_ptr<Solver>> solvers;
  solvers.push_back(std::make_unique<IdentitySolver>());
  solvers.push_back(std::make_unique<IdentitySolver>());

  smith::BlockTriangularPreconditioner P(offsets, std::move(solvers), smith::BlockTriangularType::Symmetric);
  P.SetOperator(A);

  Vector x(5), y(5);
  x.Randomize();

  P.Mult(x, y);

  mfem::Vector diff(x);
  diff -= y;

  EXPECT_NEAR(diff.Norml2(), 0.0, 1e-14);
}

// BlockDiagonalPreconditioner ignores off-diagonal blocks
TEST(BlockDiagonal, IgnoresOffDiagonalBlocks)
{
  Array<int> offsets({0, 2, 4});

  auto A11 = makeScaledIdentity(2, 2.0);
  auto A22 = makeScaledIdentity(2, 4.0);
  auto A12 = makeScaledIdentity(2, 1.0);
  auto A21 = makeScaledIdentity(2, 1.0);

  BlockOperator A(offsets);
  A.SetBlock(0, 0, A11.get());
  A.SetBlock(0, 1, A12.get());
  A.SetBlock(1, 0, A21.get());
  A.SetBlock(1, 1, A22.get());

  auto solvers = makeExactDiagonalSolvers();
  smith::BlockDiagonalPreconditioner P(offsets, std::move(solvers));
  P.SetOperator(A);

  Vector b(4), x(4);
  b.Randomize();
  P.Mult(b, x);

  EXPECT_NEAR(x[0], b[0] / 2.0, 1e-12);
  EXPECT_NEAR(x[1], b[1] / 2.0, 1e-12);
  EXPECT_NEAR(x[2], b[2] / 4.0, 1e-12);
  EXPECT_NEAR(x[3], b[3] / 4.0, 1e-12);
}

// LowerTriangularPreconditioner is exact with exact solvers for a lower
// triangular matrix
TEST(BlockTriangular, LowerTriangularExactSolve)
{
  Array<int> offsets({0, 2, 4});

  auto A11 = makeScaledIdentity(2, 2.0);
  auto A22 = makeScaledIdentity(2, 3.0);
  auto A21 = makeScaledIdentity(2, 1.0);

  BlockOperator A(offsets);
  A.SetBlock(0, 0, A11.get());
  A.SetBlock(1, 0, A21.get());
  A.SetBlock(1, 1, A22.get());

  auto solvers = makeExactDiagonalSolvers();
  smith::BlockTriangularPreconditioner P(offsets, std::move(solvers), smith::BlockTriangularType::Lower);
  P.SetOperator(A);

  Vector b(4), x(4), Ax(4);
  b.Randomize();

  P.Mult(b, x);
  A.Mult(x, Ax);

  mfem::Vector res(Ax);
  res -= b;
  EXPECT_NEAR(res.Norml2(), 0.0, 1e-12);
}

// Symmetric BlockTriangularPreconditioner is actually symmetric
TEST(BlockTriangular, SymmetricGSIsSelfAdjoint)
{
  Array<int> offsets({0, 2, 4});

  auto A11 = makeSPDMatrix(2);
  auto A22 = makeSPDMatrix(2);
  auto A12 = makeScaledIdentity(2, 0.5);
  auto A21 = makeScaledIdentity(2, 0.5);

  BlockOperator A(offsets);
  A.SetBlock(0, 0, A11.get());
  A.SetBlock(0, 1, A12.get());
  A.SetBlock(1, 0, A21.get());
  A.SetBlock(1, 1, A22.get());

  auto solvers = makeExactDiagonalSolvers();
  smith::BlockTriangularPreconditioner P(offsets, std::move(solvers), smith::BlockTriangularType::Symmetric);
  P.SetOperator(A);

  Vector x(4), y(4), Px(4), Py(4);
  x.Randomize();
  y.Randomize();

  P.Mult(x, Px);
  P.Mult(y, Py);

  double lhs = InnerProduct(Px, y);
  double rhs = InnerProduct(x, Py);

  EXPECT_NEAR(lhs, rhs, 1e-12);
}

// 3x3 block triangular system, exact solve
TEST(BlockTriangular, LowerTriangularExactSolve_3Blocks)
{
  Array<int> offsets({0, 2, 5, 7});  // block sizes: 2,3,2

  auto sz = [&](int i) { return offsets[i + 1] - offsets[i]; };

  // Diagonal blocks (square, exact solves)
  auto A11 = makeScaledIdentity(sz(0), 2.0);
  auto A22 = makeScaledIdentity(sz(1), 3.0);
  auto A33 = makeScaledIdentity(sz(2), 4.0);

  // Off-diagonal blocks (rectangular, nonzero)
  auto A21 = makeRectTridiagonal(sz(1), sz(0));  // 3x2
  auto A31 = makeRectTridiagonal(sz(2), sz(0));  // 2x2
  auto A32 = makeRectTridiagonal(sz(2), sz(1));  // 2x3

  mfem::BlockOperator A(offsets);

  // Set blocks
  A.SetBlock(0, 0, A11.get());
  A.SetBlock(1, 0, A21.get());
  A.SetBlock(1, 1, A22.get());
  A.SetBlock(2, 0, A31.get());
  A.SetBlock(2, 1, A32.get());
  A.SetBlock(2, 2, A33.get());

  // Exact diagonal solvers for each block
  std::vector<std::unique_ptr<Solver>> solvers;
  solvers.push_back(std::make_unique<ExactDiagonalSolver>());
  solvers.push_back(std::make_unique<ExactDiagonalSolver>());
  solvers.push_back(std::make_unique<ExactDiagonalSolver>());

  smith::BlockTriangularPreconditioner P(offsets, std::move(solvers), smith::BlockTriangularType::Lower);

  P.SetOperator(A);

  mfem::Vector b(sz(0) + sz(1) + sz(2)), x(sz(0) + sz(1) + sz(2)), Ax(sz(0) + sz(1) + sz(2));
  b.Randomize();

  P.Mult(b, x);
  A.Mult(x, Ax);

  mfem::Vector r(Ax);
  r -= b;
  EXPECT_NEAR(r.Norml2(), 0.0, 1e-12);
}

// UpperTriangularPreconditioner is exact with exact solvers for an upper
// triangular matrix
TEST(BlockTriangular, UpperTriangularExactSolve)
{
  Array<int> offsets({0, 2, 4});

  auto A11 = makeScaledIdentity(2, 2.0);
  auto A22 = makeScaledIdentity(2, 3.0);
  auto A12 = makeScaledIdentity(2, 1.0);

  BlockOperator A(offsets);
  A.SetBlock(0, 0, A11.get());
  A.SetBlock(0, 1, A12.get());
  A.SetBlock(1, 1, A22.get());

  auto solvers = makeExactDiagonalSolvers();
  smith::BlockTriangularPreconditioner P(offsets, std::move(solvers), smith::BlockTriangularType::Upper);

  P.SetOperator(A);

  Vector b(4), x(4), Ax(4);
  b.Randomize();

  P.Mult(b, x);
  A.Mult(x, Ax);

  mfem::Vector r(Ax);
  r -= b;
  EXPECT_NEAR(r.Norml2(), 0.0, 1e-12);
}

// Ensures that P^-1 0 = 0
TEST(BlockTriangular, ZeroInputGivesZeroOutput)
{
  Array<int> offsets({0, 2, 5});

  auto A11 = makeScaledIdentity(2, 2.0);
  auto A22 = makeScaledIdentity(3, 3.0);

  BlockOperator A(offsets);
  A.SetBlock(0, 0, A11.get());
  A.SetBlock(1, 1, A22.get());

  std::vector<std::unique_ptr<Solver>> solvers;
  solvers.push_back(std::make_unique<IdentitySolver>());
  solvers.push_back(std::make_unique<IdentitySolver>());

  smith::BlockTriangularPreconditioner P(offsets, std::move(solvers), smith::BlockTriangularType::Symmetric);

  P.SetOperator(A);

  Vector x(5), y(5);
  x = 0.0;

  P.Mult(x, y);
  EXPECT_NEAR(y.Norml2(), 0.0, 0.0);
}

// Block Triangular still works if some off-diagonal blocks are zero/missing
TEST(BlockTriangular, HandlesMissingOffDiagonalBlocks)
{
  Array<int> offsets({0, 2, 4});

  auto A11 = makeScaledIdentity(2, 2.0);
  auto A22 = makeScaledIdentity(2, 3.0);

  BlockOperator A(offsets);
  A.SetBlock(0, 0, A11.get());
  A.SetBlock(1, 1, A22.get());

  auto solvers = makeExactDiagonalSolvers();
  smith::BlockTriangularPreconditioner P(offsets, std::move(solvers), smith::BlockTriangularType::Symmetric);

  EXPECT_NO_THROW(P.SetOperator(A));
}

TEST(BlockDiagonal, WorksForSingleBlock)
{
  // Single block (non-block system)
  auto A0 = makeScaledIdentity(3, 2.0);  // 3x3 diagonal

  // Wrap as a block operator with offsets 0,3
  Array<int> offsets({0, 3});
  BlockOperator A(offsets);
  A.SetBlock(0, 0, A0.get());

  // One solver for the single block
  std::vector<std::unique_ptr<Solver>> solvers;
  solvers.push_back(std::make_unique<ExactDiagonalSolver>());

  smith::BlockDiagonalPreconditioner P(offsets, std::move(solvers));
  EXPECT_NO_THROW(P.SetOperator(A));

  // Apply preconditioner
  Vector b(3), x(3), Ax(3);
  b.Randomize();

  P.Mult(b, x);
  A.Mult(x, Ax);

  // Check that result is exact
  Vector r(Ax);
  r -= b;
  EXPECT_NEAR(r.Norml2(), 0.0, 1e-12);
}

TEST(BlockTriangular, WorksForSingleBlockAllTypes)
{
  auto A0 = makeScaledIdentity(3, 2.0);
  Array<int> offsets({0, 3});
  BlockOperator A(offsets);
  A.SetBlock(0, 0, A0.get());

  for (auto type :
       {smith::BlockTriangularType::Lower, smith::BlockTriangularType::Upper, smith::BlockTriangularType::Symmetric}) {
    std::vector<std::unique_ptr<Solver>> solvers;
    solvers.push_back(std::make_unique<ExactDiagonalSolver>());

    smith::BlockTriangularPreconditioner P(offsets, std::move(solvers), type);
    EXPECT_NO_THROW(P.SetOperator(A));

    Vector b(3), x(3), Ax(3);
    b.Randomize();
    P.Mult(b, x);
    A.Mult(x, Ax);

    Vector r(Ax);
    r -= b;
    EXPECT_NEAR(r.Norml2(), 0.0, 1e-12);
  }
}

/* ============================================================
   main
   ============================================================ */

int main(int argc, char* argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  smith::ApplicationManager applicationManager(argc, argv);
  return RUN_ALL_TESTS();
}
