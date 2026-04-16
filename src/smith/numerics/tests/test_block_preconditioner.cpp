#include <gtest/gtest.h>
#include <mpi.h>

#include "mfem.hpp"
#include "smith/numerics/block_preconditioner.hpp"
#include "smith/infrastructure/application_manager.hpp"

#include "axom/slic.hpp"

using namespace mfem;

bool abort_called = false;

void testAbortHandler()
{
  abort_called = true;  // record abort instead of exiting
}

/* ============================================================
   Helper utilities
   ============================================================ */

namespace {

// Own both the local CSR (used to build the hypre matrix) and the HypreParMatrix.
// This avoids dangling pointers because some HypreParMatrix constructors wrap
// CSR arrays without taking ownership.
struct OwnedHypreParMatrix {
  std::unique_ptr<SparseMatrix> diag;
  std::unique_ptr<HypreParMatrix> A;
};

// Build a HypreParMatrix from a local CSR matrix, assuming serial execution.
// Uses a block-diagonal ParCSR matrix whose local diagonal block is 'diag'.
OwnedHypreParMatrix makeHypreFromLocalDiag(std::unique_ptr<SparseMatrix> diag, HYPRE_BigInt global_num_rows,
                                           HYPRE_BigInt global_num_cols, MPI_Comm comm)
{
  MFEM_VERIFY(diag, "diag must be non-null");
  diag->Finalize();

  auto row_starts = std::array<HYPRE_BigInt, 2>{static_cast<HYPRE_BigInt>(0), global_num_rows};
  auto col_starts = std::array<HYPRE_BigInt, 2>{static_cast<HYPRE_BigInt>(0), global_num_cols};

  auto A = std::make_unique<HypreParMatrix>(comm, global_num_rows, global_num_cols, row_starts.data(),
                                            col_starts.data(), diag.get());

  // Ensure partition arrays are owned by HypreParMatrix (so std::array can die)
  A->CopyRowStarts();
  A->CopyColStarts();

  OwnedHypreParMatrix out;
  out.diag = std::move(diag);
  out.A = std::move(A);
  return out;
}

// Build c * I (square)
OwnedHypreParMatrix makeHypreScaledIdentity(int n, double c, MPI_Comm comm = MPI_COMM_WORLD)
{
  auto diag = std::make_unique<SparseMatrix>(n);
  for (int i = 0; i < n; i++) {
    diag->Add(i, i, c);
  }

  const auto N = static_cast<HYPRE_BigInt>(n);
  return makeHypreFromLocalDiag(std::move(diag), N, N, comm);
}

// Simple SPD matrix: tridiagonal (square)
OwnedHypreParMatrix makeHypreSPDMatrix(int n, MPI_Comm comm = MPI_COMM_WORLD)
{
  auto diag = std::make_unique<SparseMatrix>(n);
  for (int i = 0; i < n; i++) {
    diag->Add(i, i, 2.0);
    if (i > 0) {
      diag->Add(i, i - 1, -1.0);
    }
    if (i < n - 1) {
      diag->Add(i, i + 1, -1.0);
    }
  }

  const auto N = static_cast<HYPRE_BigInt>(n);
  return makeHypreFromLocalDiag(std::move(diag), N, N, comm);
}

// Tridiagonal rectangular matrix with constant main diagonal 2.0, off-diagonals -1.0
OwnedHypreParMatrix makeHypreRectTridiagonal(int rows, int cols, MPI_Comm comm = MPI_COMM_WORLD)
{
  auto diag = std::make_unique<SparseMatrix>(rows, cols);

  for (int i = 0; i < rows; i++) {
    if (i < cols) {
      diag->Add(i, i, 2.0);
    }  // main diagonal
    if (i > 0 && i - 1 < cols) {
      diag->Add(i, i - 1, -1.0);
    }  // lower diagonal
    if (i + 1 < cols) {
      diag->Add(i, i + 1, -1.0);
    }  // upper diagonal
  }

  const auto R = static_cast<HYPRE_BigInt>(rows);
  const auto C = static_cast<HYPRE_BigInt>(cols);
  return makeHypreFromLocalDiag(std::move(diag), R, C, comm);
}

class IdentitySolver : public mfem::Solver {
 public:
  void SetOperator(const mfem::Operator& op) override
  {
    height = op.Height();
    width = op.Width();
  }

  void Mult(const mfem::Vector& x, mfem::Vector& y) const override { y = x; }
};

// Exact diagonal inverse solver for HypreParMatrix
class HypreExactDiagonalSolver : public mfem::Solver {
 public:
  void SetOperator(const mfem::Operator& op) override
  {
    const auto* A = dynamic_cast<const mfem::HypreParMatrix*>(&op);
    MFEM_VERIFY(A, "HypreExactDiagonalSolver requires HypreParMatrix");

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
  const mfem::HypreParMatrix* A_ = nullptr;
  mfem::Vector diag_;
};

std::vector<std::unique_ptr<Solver>> makeExactDiagonalSolvers()
{
  std::vector<std::unique_ptr<Solver>> solvers;
  solvers.push_back(std::make_unique<HypreExactDiagonalSolver>());
  solvers.push_back(std::make_unique<HypreExactDiagonalSolver>());
  return solvers;
}

}  // namespace

/* ============================================================
   Tests
   ============================================================ */

// Makes sure an error is thrown when length of solvers does not match the
// number of blocks
TEST(BlockDiagonal, ThrowsOnWrongNumberOfSolvers)
{
  // Ensure that SLIC uses the abort handler for errors.
  // Replace abort handler
  axom::slic::setAbortFunction(testAbortHandler);
  axom::slic::setAbortOnError(true);
  axom::slic::setAbortOnWarning(false);

  abort_called = false;

  Array<int> offsets({0, 2, 4});
  std::vector<std::unique_ptr<Solver>> solvers;
  solvers.push_back(std::make_unique<IdentitySolver>());

  mfem::BlockOperator A(offsets);
  // Set diagonal blocks so BlockDiagonalPreconditioner can query them.
  auto A11 = makeScaledIdentity(2, 1.0);
  auto A22 = makeScaledIdentity(2, 1.0);
  A.SetBlock(0, 0, A11.get());
  A.SetBlock(1, 1, A22.get());
  smith::BlockDiagonalPreconditioner P(std::move(solvers));
  P.SetOperator(A);

  EXPECT_TRUE(abort_called);
}

// If the solver for each block is identity, the block solver is identity
TEST(BlockTriangular, IdentityActsAsIdentity)
{
  Array<int> offsets({0, 2, 5});

  auto A11o = makeHypreScaledIdentity(2, 1.0);
  auto A22o = makeHypreScaledIdentity(3, 1.0);

  BlockOperator A(offsets);
  A.SetBlock(0, 0, A11o.A.get());
  A.SetBlock(1, 1, A22o.A.get());

  std::vector<std::unique_ptr<Solver>> solvers;
  solvers.push_back(std::make_unique<IdentitySolver>());
  solvers.push_back(std::make_unique<IdentitySolver>());

  smith::BlockTriangularPreconditioner P(std::move(solvers), smith::BlockTriangularType::Symmetric);
  P.SetOperator(A);

  Vector x(5), y(5);
  x.Randomize();

  P.Mult(x, y);

  Vector diff(x);
  diff -= y;

  EXPECT_NEAR(diff.Norml2(), 0.0, 1e-14);
}

// BlockDiagonalPreconditioner ignores off-diagonal blocks
TEST(BlockDiagonal, IgnoresOffDiagonalBlocks)
{
  Array<int> offsets({0, 2, 4});

  auto A11o = makeHypreScaledIdentity(2, 2.0);
  auto A22o = makeHypreScaledIdentity(2, 4.0);
  auto A12o = makeHypreScaledIdentity(2, 1.0);
  auto A21o = makeHypreScaledIdentity(2, 1.0);

  BlockOperator A(offsets);
  A.SetBlock(0, 0, A11o.A.get());
  A.SetBlock(0, 1, A12o.A.get());
  A.SetBlock(1, 0, A21o.A.get());
  A.SetBlock(1, 1, A22o.A.get());

  auto solvers = makeExactDiagonalSolvers();
  smith::BlockDiagonalPreconditioner P(std::move(solvers));
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

  auto A11o = makeHypreScaledIdentity(2, 2.0);
  auto A22o = makeHypreScaledIdentity(2, 3.0);
  auto A21o = makeHypreScaledIdentity(2, 1.0);

  BlockOperator A(offsets);
  A.SetBlock(0, 0, A11o.A.get());
  A.SetBlock(1, 0, A21o.A.get());
  A.SetBlock(1, 1, A22o.A.get());

  auto solvers = makeExactDiagonalSolvers();
  smith::BlockTriangularPreconditioner P(std::move(solvers), smith::BlockTriangularType::Lower);
  P.SetOperator(A);

  Vector b(4), x(4), Ax(4);
  b.Randomize();

  P.Mult(b, x);
  A.Mult(x, Ax);

  Vector res(Ax);
  res -= b;
  EXPECT_NEAR(res.Norml2(), 0.0, 1e-12);
}

// Symmetric BlockTriangularPreconditioner is actually symmetric
TEST(BlockTriangular, SymmetricGSIsSelfAdjoint)
{
  Array<int> offsets({0, 2, 4});

  auto A11o = makeHypreSPDMatrix(2);
  auto A22o = makeHypreSPDMatrix(2);
  auto A12o = makeHypreScaledIdentity(2, 0.5);
  auto A21o = makeHypreScaledIdentity(2, 0.5);

  BlockOperator A(offsets);
  A.SetBlock(0, 0, A11o.A.get());
  A.SetBlock(0, 1, A12o.A.get());
  A.SetBlock(1, 0, A21o.A.get());
  A.SetBlock(1, 1, A22o.A.get());

  auto solvers = makeExactDiagonalSolvers();
  smith::BlockTriangularPreconditioner P(std::move(solvers), smith::BlockTriangularType::Symmetric);
  P.SetOperator(A);

  Vector x(4), y(4), Px(4), Py(4);
  x.Randomize();
  y.Randomize();

  P.Mult(x, Px);
  P.Mult(y, Py);

  const double lhs = InnerProduct(Px, y);
  const double rhs = InnerProduct(x, Py);

  EXPECT_NEAR(lhs, rhs, 1e-12);
}

// 3x3 block triangular system, exact solve
TEST(BlockTriangular, LowerTriangularExactSolve_3Blocks)
{
  Array<int> offsets({0, 2, 5, 7});  // block sizes: 2,3,2
  auto sz = [&](int i) { return offsets[i + 1] - offsets[i]; };

  // Diagonal blocks (square, exact solves)
  auto A11o = makeHypreScaledIdentity(sz(0), 2.0);
  auto A22o = makeHypreScaledIdentity(sz(1), 3.0);
  auto A33o = makeHypreScaledIdentity(sz(2), 4.0);

  // Off-diagonal blocks (rectangular, nonzero)
  auto A21o = makeHypreRectTridiagonal(sz(1), sz(0));  // 3x2
  auto A31o = makeHypreRectTridiagonal(sz(2), sz(0));  // 2x2
  auto A32o = makeHypreRectTridiagonal(sz(2), sz(1));  // 2x3

  BlockOperator A(offsets);

  // Set blocks
  A.SetBlock(0, 0, A11o.A.get());
  A.SetBlock(1, 0, A21o.A.get());
  A.SetBlock(1, 1, A22o.A.get());
  A.SetBlock(2, 0, A31o.A.get());
  A.SetBlock(2, 1, A32o.A.get());
  A.SetBlock(2, 2, A33o.A.get());

  // Exact diagonal solvers for each block
  std::vector<std::unique_ptr<Solver>> solvers;
  solvers.push_back(std::make_unique<HypreExactDiagonalSolver>());
  solvers.push_back(std::make_unique<HypreExactDiagonalSolver>());
  solvers.push_back(std::make_unique<HypreExactDiagonalSolver>());

  smith::BlockTriangularPreconditioner P(std::move(solvers), smith::BlockTriangularType::Lower);

  P.SetOperator(A);

  const int n = sz(0) + sz(1) + sz(2);
  Vector b(n), x(n), Ax(n);
  b.Randomize();

  P.Mult(b, x);
  A.Mult(x, Ax);

  Vector r(Ax);
  r -= b;
  EXPECT_NEAR(r.Norml2(), 0.0, 1e-12);
}

// UpperTriangularPreconditioner is exact with exact solvers for an upper
// triangular matrix
TEST(BlockTriangular, UpperTriangularExactSolve)
{
  Array<int> offsets({0, 2, 4});

  auto A11o = makeHypreScaledIdentity(2, 2.0);
  auto A22o = makeHypreScaledIdentity(2, 3.0);
  auto A12o = makeHypreScaledIdentity(2, 1.0);

  BlockOperator A(offsets);
  A.SetBlock(0, 0, A11o.A.get());
  A.SetBlock(0, 1, A12o.A.get());
  A.SetBlock(1, 1, A22o.A.get());

  auto solvers = makeExactDiagonalSolvers();
  smith::BlockTriangularPreconditioner P(std::move(solvers), smith::BlockTriangularType::Upper);

  P.SetOperator(A);

  Vector b(4), x(4), Ax(4);
  b.Randomize();

  P.Mult(b, x);
  A.Mult(x, Ax);

  Vector r(Ax);
  r -= b;
  EXPECT_NEAR(r.Norml2(), 0.0, 1e-12);
}

// Ensures that P^-1 0 = 0
TEST(BlockTriangular, ZeroInputGivesZeroOutput)
{
  Array<int> offsets({0, 2, 5});

  auto A11o = makeHypreScaledIdentity(2, 2.0);
  auto A22o = makeHypreScaledIdentity(3, 3.0);

  BlockOperator A(offsets);
  A.SetBlock(0, 0, A11o.A.get());
  A.SetBlock(1, 1, A22o.A.get());

  std::vector<std::unique_ptr<Solver>> solvers;
  solvers.push_back(std::make_unique<IdentitySolver>());
  solvers.push_back(std::make_unique<IdentitySolver>());

  smith::BlockTriangularPreconditioner P(std::move(solvers), smith::BlockTriangularType::Symmetric);

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

  auto A11o = makeHypreScaledIdentity(2, 2.0);
  auto A22o = makeHypreScaledIdentity(2, 3.0);

  BlockOperator A(offsets);
  A.SetBlock(0, 0, A11o.A.get());
  A.SetBlock(1, 1, A22o.A.get());

  auto solvers = makeExactDiagonalSolvers();
  smith::BlockTriangularPreconditioner P(std::move(solvers), smith::BlockTriangularType::Symmetric);

  EXPECT_NO_THROW(P.SetOperator(A));
}

TEST(BlockDiagonal, WorksForSingleBlock)
{
  auto A0o = makeHypreScaledIdentity(3, 2.0);

  Array<int> offsets({0, 3});
  BlockOperator A(offsets);
  A.SetBlock(0, 0, A0o.A.get());

  std::vector<std::unique_ptr<Solver>> solvers;
  solvers.push_back(std::make_unique<HypreExactDiagonalSolver>());

  smith::BlockDiagonalPreconditioner P(std::move(solvers));
  EXPECT_NO_THROW(P.SetOperator(A));

  Vector b(3), x(3), Ax(3);
  b.Randomize();

  P.Mult(b, x);
  A.Mult(x, Ax);

  Vector r(Ax);
  r -= b;
  EXPECT_NEAR(r.Norml2(), 0.0, 1e-12);
}

TEST(BlockTriangular, WorksForSingleBlockAllTypes)
{
  auto A0o = makeHypreScaledIdentity(3, 2.0);

  Array<int> offsets({0, 3});
  BlockOperator A(offsets);
  A.SetBlock(0, 0, A0o.A.get());

  for (auto type :
       {smith::BlockTriangularType::Lower, smith::BlockTriangularType::Upper, smith::BlockTriangularType::Symmetric}) {
    std::vector<std::unique_ptr<Solver>> solvers;
    solvers.push_back(std::make_unique<HypreExactDiagonalSolver>());

    smith::BlockTriangularPreconditioner P(std::move(solvers), type);
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

TEST(BlockSchur, ExactSolveforDiagonals)
{
  Array<int> offsets({0, 2, 4});

  // Diagonal blocks (square, exact solves)
  auto A11o = makeHypreScaledIdentity(2, 2.0);
  auto A22o = makeHypreScaledIdentity(2, 3.0);

  // Off-diagonal blocks (rectangular, nonzero)
  auto A12o = makeHypreScaledIdentity(2, 4.0);
  auto A21o = makeHypreScaledIdentity(2, 5.0);

  BlockOperator A(offsets);
  A.SetBlock(0, 0, A11o.A.get());
  A.SetBlock(1, 1, A22o.A.get());
  A.SetBlock(0, 1, A12o.A.get());
  A.SetBlock(1, 0, A21o.A.get());

  std::vector<std::unique_ptr<Solver>> solvers;
  solvers.push_back(std::make_unique<HypreExactDiagonalSolver>());
  solvers.push_back(std::make_unique<HypreExactDiagonalSolver>());

  smith::BlockSchurPreconditioner P(offsets, std::move(solvers), smith::BlockSchurType::Full,
                                    smith::SchurApproxType::DiagInv);
  P.SetOperator(A);

  Vector b(4), x(4), Ax(4);
  b.Randomize();

  P.Mult(b, x);
  A.Mult(x, Ax);

  Vector res(Ax);
  res -= b;
  EXPECT_NEAR(res.Norml2(), 0.0, 1e-12);
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
