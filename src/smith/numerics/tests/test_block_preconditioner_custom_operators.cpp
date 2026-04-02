#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "mfem.hpp"

#define private public
#include "smith/numerics/block_preconditioner.hpp"
#undef private
#include "smith/infrastructure/application_manager.hpp"

using namespace mfem;

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

OwnedHypreParMatrix makeHypreFromLocalDiag(std::unique_ptr<SparseMatrix> diag, HYPRE_BigInt global_num_rows,
                                           HYPRE_BigInt global_num_cols, MPI_Comm comm);

mfem::DenseMatrix denseFrom2x2Operator(const mfem::Operator& op)
{
  MFEM_VERIFY(op.Height() == 2 && op.Width() == 2, "Operator must be 2x2");

  mfem::DenseMatrix M(2, 2);
  mfem::Vector x(2), y(2);

  x = 0.0;
  x[0] = 1.0;
  op.Mult(x, y);
  M(0, 0) = y[0];
  M(1, 0) = y[1];

  x = 0.0;
  x[1] = 1.0;
  op.Mult(x, y);
  M(0, 1) = y[0];
  M(1, 1) = y[1];

  return M;
}

mfem::DenseMatrix invertDense2x2(const mfem::DenseMatrix& A)
{
  MFEM_VERIFY(A.Height() == 2 && A.Width() == 2, "Matrix must be 2x2");
  const double a = A(0, 0);
  const double b = A(0, 1);
  const double c = A(1, 0);
  const double d = A(1, 1);
  const double det = a * d - b * c;
  MFEM_VERIFY(std::abs(det) > 1e-14, "2x2 matrix is singular or nearly singular");

  mfem::DenseMatrix inv(2, 2);
  inv(0, 0) = d / det;
  inv(0, 1) = -b / det;
  inv(1, 0) = -c / det;
  inv(1, 1) = a / det;
  return inv;
}

mfem::DenseMatrix invert2x2HypreParMatrixDense(const mfem::HypreParMatrix& A)
{
  const mfem::DenseMatrix dense = denseFrom2x2Operator(A);
  return invertDense2x2(dense);
}

OwnedHypreParMatrix makeHypreFromDense2x2(const mfem::DenseMatrix& dense, MPI_Comm comm = MPI_COMM_WORLD)
{
  MFEM_VERIFY(dense.Height() == 2 && dense.Width() == 2, "Dense matrix must be 2x2");
  auto diag = std::make_unique<SparseMatrix>(2, 2);
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      diag->Add(i, j, dense(i, j));
    }
  }
  const auto N = static_cast<HYPRE_BigInt>(2);
  return makeHypreFromLocalDiag(std::move(diag), N, N, comm);
}

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

  OwnedHypreParMatrix Aout;
  Aout.diag = std::move(diag);
  Aout.A = std::move(A);
  return Aout;
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

class Exact2x2Solver : public mfem::Solver {
 public:
  void SetOperator(const mfem::Operator& op) override
  {
    const auto* A = dynamic_cast<const mfem::HypreParMatrix*>(&op);
    MFEM_VERIFY(A, "Exact2x2Solver requires HypreParMatrix");
    MFEM_VERIFY(A->Height() == 2 && A->Width() == 2, "Exact2x2Solver requires 2x2 operator");

    inv_ = invert2x2HypreParMatrixDense(*A);
    height = 2;
    width = 2;
  }

  void Mult(const mfem::Vector& x, mfem::Vector& y) const override
  {
    MFEM_VERIFY(x.Size() == 2, "Exact2x2Solver expects size-2 vector");
    y.SetSize(2);
    y[0] = inv_(0, 0) * x[0] + inv_(0, 1) * x[1];
    y[1] = inv_(1, 0) * x[0] + inv_(1, 1) * x[1];
  }

 private:
  mfem::DenseMatrix inv_;
};

std::vector<std::unique_ptr<mfem::Solver>> makeExactDiagonalSolvers(int nblocks)
{
  std::vector<std::unique_ptr<mfem::Solver>> solvers;
  solvers.reserve(static_cast<size_t>(nblocks));
  for (int i = 0; i < nblocks; i++) {
    solvers.push_back(std::make_unique<HypreExactDiagonalSolver>());
  }
  return solvers;
}

std::vector<std::unique_ptr<mfem::Solver>> makeIdentitySolvers(int nblocks)
{
  std::vector<std::unique_ptr<mfem::Solver>> solvers;
  solvers.reserve(static_cast<size_t>(nblocks));
  for (int i = 0; i < nblocks; i++) {
    solvers.push_back(std::make_unique<IdentitySolver>());
  }
  return solvers;
}

std::unique_ptr<const mfem::Operator> makeLocalScaledIdentityOp(int n, double c)
{
  auto* mat = new mfem::SparseMatrix(n);
  for (int i = 0; i < n; i++) {
    mat->Add(i, i, c);
  }
  mat->Finalize();
  return std::unique_ptr<const mfem::Operator>(mat);
}

}  // namespace
/* ============================================================
   Tests
   ============================================================ */

// Makes sure an error is thrown when length of solvers does not match the
// number of blocks

// If the solver for each block is identity, the block solver is identity
TEST(BlockDiagonalPreconditionerCustom, IdentityActsAsIdentity)
{
  Array<int> offsets({0, 2, 5});

  auto A11o = makeHypreScaledIdentity(2, 2.0);
  auto A22o = makeHypreScaledIdentity(3, 2.0);

  BlockOperator A(offsets);
  A.SetBlock(0, 0, A11o.A.get());
  A.SetBlock(1, 1, A22o.A.get());

  std::vector<std::unique_ptr<Solver>> solvers;
  solvers.push_back(std::make_unique<HypreExactDiagonalSolver>());
  solvers.push_back(std::make_unique<HypreExactDiagonalSolver>());

  std::vector<OwnedHypreParMatrix> override_mats;
  override_mats.push_back(makeHypreScaledIdentity(2, 1.0));  // M1
  override_mats.push_back(makeHypreScaledIdentity(3, 1.0));  // M2

  std::vector<std::pair<int, std::unique_ptr<const mfem::Operator>>> overrides;
  overrides.emplace_back(0, std::unique_ptr<const mfem::Operator>(std::move(override_mats[0].A)));
  overrides.emplace_back(1, std::unique_ptr<const mfem::Operator>(std::move(override_mats[1].A)));

  smith::BlockDiagonalPreconditioner P(offsets, std::move(solvers), std::move(overrides));

  P.SetOperator(A);  // This actually doesn't use A, it's overridden by M1, M2

  Vector x(5), y(5);
  x.Randomize();

  P.Mult(x, y);

  mfem::Vector diff(x);
  diff -= y;

  EXPECT_NEAR(diff.Norml2(), 0.0, 1e-14);
}

TEST(BlockDiagonalPreconditionerCustom, PartialOverrideUsesJacobianForOthers)
{
  Array<int> offsets({0, 2, 4});

  auto A11o = makeHypreScaledIdentity(2, 2.0);
  auto A22o = makeHypreScaledIdentity(2, 3.0);

  BlockOperator A(offsets);
  A.SetBlock(0, 0, A11o.A.get());
  A.SetBlock(1, 1, A22o.A.get());

  auto solvers = makeExactDiagonalSolvers(2);

  std::vector<OwnedHypreParMatrix> override_mats;
  override_mats.push_back(makeHypreScaledIdentity(2, 1.0));  // override block 1

  std::vector<smith::BlockOverride> overrides;
  overrides.emplace_back(1, std::unique_ptr<const mfem::Operator>(std::move(override_mats[0].A)));

  smith::BlockDiagonalPreconditioner P(offsets, std::move(solvers), std::move(overrides));
  P.SetOperator(A);

  Vector b(4), x(4);
  b.Randomize();

  P.Mult(b, x);

  // Block 0 uses Jacobian block A11 = 2I
  EXPECT_NEAR(x[0], b[0] / 2.0, 1e-12);
  EXPECT_NEAR(x[1], b[1] / 2.0, 1e-12);
  // Block 1 uses override operator I
  EXPECT_NEAR(x[2], b[2], 1e-12);
  EXPECT_NEAR(x[3], b[3], 1e-12);
}

TEST(BlockDiagonalPreconditionerCustom, OverrideBeatsBadJacobianBlock)
{
  Array<int> offsets({0, 2, 4});

  auto A11o = makeHypreScaledIdentity(2, 0.0);  // singular
  auto A22o = makeHypreScaledIdentity(2, 2.0);

  BlockOperator A(offsets);
  A.SetBlock(0, 0, A11o.A.get());
  A.SetBlock(1, 1, A22o.A.get());

  auto solvers = makeExactDiagonalSolvers(2);

  std::vector<OwnedHypreParMatrix> override_mats;
  override_mats.push_back(makeHypreScaledIdentity(2, 1.0));  // override A11 with I

  std::vector<smith::BlockOverride> overrides;
  overrides.emplace_back(0, std::unique_ptr<const mfem::Operator>(std::move(override_mats[0].A)));

  smith::BlockDiagonalPreconditioner P(offsets, std::move(solvers), std::move(overrides));
  P.SetOperator(A);

  Vector b(4), x(4);
  b.Randomize();
  P.Mult(b, x);

  EXPECT_TRUE(std::isfinite(x[0]));
  EXPECT_TRUE(std::isfinite(x[1]));

  EXPECT_NEAR(x[0], b[0], 1e-12);
  EXPECT_NEAR(x[1], b[1], 1e-12);
  EXPECT_NEAR(x[2], b[2] / 2.0, 1e-12);
  EXPECT_NEAR(x[3], b[3] / 2.0, 1e-12);
}

TEST(BlockTriangularPreconditionerCustom, LowerSweepUsesOverrideDiagonal)
{
  Array<int> offsets({0, 2, 4});

  auto A11o = makeHypreScaledIdentity(2, 2.0);
  auto A22o = makeHypreScaledIdentity(2, 3.0);
  auto A21o = makeHypreScaledIdentity(2, 1.0);

  BlockOperator A(offsets);
  A.SetBlock(0, 0, A11o.A.get());
  A.SetBlock(1, 0, A21o.A.get());
  A.SetBlock(1, 1, A22o.A.get());

  auto solvers = makeExactDiagonalSolvers(2);

  std::vector<OwnedHypreParMatrix> override_mats;
  override_mats.push_back(makeHypreScaledIdentity(2, 6.0));  // override A22_used

  std::vector<smith::BlockOverride> overrides;
  overrides.emplace_back(1, std::unique_ptr<const mfem::Operator>(std::move(override_mats[0].A)));

  smith::BlockTriangularPreconditioner P(offsets, std::move(solvers), smith::BlockTriangularType::Lower,
                                         std::move(overrides));
  P.SetOperator(A);

  Vector b(4), x(4);
  b.Randomize();
  P.Mult(b, x);

  BlockVector bb(b, offsets);
  BlockVector xx(x, offsets);

  Vector x0_expected(2);
  x0_expected = bb.GetBlock(0);
  x0_expected /= 2.0;

  Vector x1_expected(2);
  x1_expected = bb.GetBlock(1);
  x1_expected -= x0_expected;  // A21 = I
  x1_expected /= 6.0;          // override A22 = 6I

  for (int i = 0; i < 2; i++) {
    EXPECT_NEAR(xx.GetBlock(0)[i], x0_expected[i], 1e-12);
    EXPECT_NEAR(xx.GetBlock(1)[i], x1_expected[i], 1e-12);
  }
}

TEST(BlockTriangularPreconditionerCustom, UpperSweepUsesOverrideDiagonal)
{
  Array<int> offsets({0, 2, 4});

  auto A11o = makeHypreScaledIdentity(2, 2.0);
  auto A22o = makeHypreScaledIdentity(2, 3.0);
  auto A12o = makeHypreScaledIdentity(2, 1.0);

  BlockOperator A(offsets);
  A.SetBlock(0, 0, A11o.A.get());
  A.SetBlock(0, 1, A12o.A.get());
  A.SetBlock(1, 1, A22o.A.get());

  auto solvers = makeExactDiagonalSolvers(2);

  std::vector<OwnedHypreParMatrix> override_mats;
  override_mats.push_back(makeHypreScaledIdentity(2, 4.0));  // override A11_used

  std::vector<smith::BlockOverride> overrides;
  overrides.emplace_back(0, std::unique_ptr<const mfem::Operator>(std::move(override_mats[0].A)));

  smith::BlockTriangularPreconditioner P(offsets, std::move(solvers), smith::BlockTriangularType::Upper,
                                         std::move(overrides));
  P.SetOperator(A);

  Vector b(4), x(4);
  b.Randomize();
  P.Mult(b, x);

  BlockVector bb(b, offsets);
  BlockVector xx(x, offsets);

  Vector x1_expected(2);
  x1_expected = bb.GetBlock(1);
  x1_expected /= 3.0;

  Vector x0_expected(2);
  x0_expected = bb.GetBlock(0);
  x0_expected -= x1_expected;  // A12 = I
  x0_expected /= 4.0;          // override A11 = 4I

  for (int i = 0; i < 2; i++) {
    EXPECT_NEAR(xx.GetBlock(0)[i], x0_expected[i], 1e-12);
    EXPECT_NEAR(xx.GetBlock(1)[i], x1_expected[i], 1e-12);
  }
}

TEST(BlockSchurPreconditionerCustom, FullWithExactSchurOverrideIsExactInverse)
{
  Array<int> offsets({0, 2, 4});

  // Use dense (non-diagonal) 2x2 blocks.
  mfem::DenseMatrix A11d(2, 2);
  A11d(0, 0) = 4.0;
  A11d(0, 1) = 1.0;
  A11d(1, 0) = 2.0;
  A11d(1, 1) = 3.0;

  mfem::DenseMatrix A12d(2, 2);
  A12d(0, 0) = 1.0;
  A12d(0, 1) = 2.0;
  A12d(1, 0) = 3.0;
  A12d(1, 1) = 4.0;

  mfem::DenseMatrix A21d(2, 2);
  A21d(0, 0) = -1.0;
  A21d(0, 1) = 1.0;
  A21d(1, 0) = 0.0;
  A21d(1, 1) = 2.0;

  mfem::DenseMatrix A22d(2, 2);
  A22d(0, 0) = 5.0;
  A22d(0, 1) = 1.0;
  A22d(1, 0) = 1.0;
  A22d(1, 1) = 4.0;

  auto A11o = makeHypreFromDense2x2(A11d);
  auto A12o = makeHypreFromDense2x2(A12d);
  auto A21o = makeHypreFromDense2x2(A21d);
  auto A22o = makeHypreFromDense2x2(A22d);

  const mfem::DenseMatrix A11inv = invert2x2HypreParMatrixDense(*A11o.A);

  auto mul2x2 = [](const mfem::DenseMatrix& L, const mfem::DenseMatrix& R) {
    mfem::DenseMatrix out_mat(2, 2);
    out_mat(0, 0) = L(0, 0) * R(0, 0) + L(0, 1) * R(1, 0);
    out_mat(0, 1) = L(0, 0) * R(0, 1) + L(0, 1) * R(1, 1);
    out_mat(1, 0) = L(1, 0) * R(0, 0) + L(1, 1) * R(1, 0);
    out_mat(1, 1) = L(1, 0) * R(0, 1) + L(1, 1) * R(1, 1);
    return out_mat;
  };

  const mfem::DenseMatrix tmp = mul2x2(A11inv, A12d);
  const mfem::DenseMatrix prod = mul2x2(A21d, tmp);

  mfem::DenseMatrix Sd(2, 2);
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      Sd(i, j) = A22d(i, j) - prod(i, j);
    }
  }
  (void)invertDense2x2(Sd);

  std::vector<OwnedHypreParMatrix> override_mats;
  override_mats.push_back(makeHypreFromDense2x2(Sd));

  BlockOperator A(offsets);
  A.SetBlock(0, 0, A11o.A.get());
  A.SetBlock(0, 1, A12o.A.get());
  A.SetBlock(1, 0, A21o.A.get());
  A.SetBlock(1, 1, A22o.A.get());

  std::vector<std::unique_ptr<Solver>> solvers;
  solvers.push_back(std::make_unique<Exact2x2Solver>());
  solvers.push_back(std::make_unique<Exact2x2Solver>());

  std::vector<smith::BlockOverride> overrides;
  overrides.emplace_back(1, std::unique_ptr<const mfem::Operator>(std::move(override_mats[0].A)));

  smith::BlockSchurPreconditioner P(offsets, std::move(solvers), smith::BlockSchurType::Full,
                                    smith::SchurApproxType::Custom, std::move(overrides));
  P.SetOperator(A);

  Vector rhs(4), x(4), Ax(4);
  rhs.Randomize();
  P.Mult(rhs, x);
  A.Mult(x, Ax);

  Vector r(Ax);
  r -= rhs;
  EXPECT_NEAR(r.Norml2(), 0.0, 1e-12);
}

TEST(BlockSchurPreconditionerCustom, Block0OverrideIsUsed)
{
  constexpr int n = 2;
  Array<int> offsets({0, n, 2 * n});

  auto A11o = makeHypreScaledIdentity(n, 2.0);
  auto A12o = makeHypreScaledIdentity(n, 0.0);
  auto A21o = makeHypreScaledIdentity(n, 0.0);
  auto A22o = makeHypreScaledIdentity(n, 3.0);

  BlockOperator A(offsets);
  A.SetBlock(0, 0, A11o.A.get());
  A.SetBlock(0, 1, A12o.A.get());
  A.SetBlock(1, 0, A21o.A.get());
  A.SetBlock(1, 1, A22o.A.get());

  auto solvers = makeExactDiagonalSolvers(2);

  std::vector<OwnedHypreParMatrix> override_mats;
  override_mats.push_back(makeHypreScaledIdentity(n, 4.0));

  std::vector<smith::BlockOverride> overrides;
  overrides.emplace_back(0, std::unique_ptr<const mfem::Operator>(std::move(override_mats[0].A)));

  smith::BlockSchurPreconditioner P(offsets, std::move(solvers), smith::BlockSchurType::Diagonal,
                                    smith::SchurApproxType::A22Only, std::move(overrides));
  P.SetOperator(A);

  Vector b(2 * n), x(2 * n);
  b.Randomize();
  P.Mult(b, x);

  EXPECT_NEAR(x[0], b[0] / 4.0, 1e-12);
  EXPECT_NEAR(x[1], b[1] / 4.0, 1e-12);
  EXPECT_NEAR(x[2], b[2] / 3.0, 1e-12);
  EXPECT_NEAR(x[3], b[3] / 3.0, 1e-12);
}

TEST(BlockSchurPreconditionerCustom, CustomOverrideNotConsumedOnRepeatedSetOperator)
{
  constexpr int n = 2;
  Array<int> offsets({0, n, 2 * n});

  auto A11o = makeHypreScaledIdentity(n, 2.0);
  auto A12o = makeHypreScaledIdentity(n, 0.0);
  auto A21o = makeHypreScaledIdentity(n, 0.0);
  auto A22o = makeHypreScaledIdentity(n, 3.0);

  BlockOperator A(offsets);
  A.SetBlock(0, 0, A11o.A.get());
  A.SetBlock(0, 1, A12o.A.get());
  A.SetBlock(1, 0, A21o.A.get());
  A.SetBlock(1, 1, A22o.A.get());

  auto solvers = makeExactDiagonalSolvers(2);

  std::vector<OwnedHypreParMatrix> override_mats;
  override_mats.push_back(makeHypreScaledIdentity(n, 7.0));

  std::vector<smith::BlockOverride> overrides;
  overrides.emplace_back(1, std::unique_ptr<const mfem::Operator>(std::move(override_mats[0].A)));

  smith::BlockSchurPreconditioner P(offsets, std::move(solvers), smith::BlockSchurType::Diagonal,
                                    smith::SchurApproxType::Custom, std::move(overrides));

  P.SetOperator(A);
  ASSERT_TRUE(P.block_op_overrides_[1] != nullptr);
  P.SetOperator(A);
  ASSERT_TRUE(P.block_op_overrides_[1] != nullptr);

  Vector b(2 * n), x(2 * n);
  b.Randomize();
  P.Mult(b, x);

  EXPECT_NEAR(x[2], b[2] / 7.0, 1e-12);
  EXPECT_NEAR(x[3], b[3] / 7.0, 1e-12);
}

TEST(BlockDiagonalPreconditionerCustom, ThrowsOnOutOfRangeOverrideIndex)
{
  Array<int> offsets({0, 2, 4});
  EXPECT_THROW(
      {
        auto solvers = makeIdentitySolvers(2);
        std::vector<smith::BlockOverride> overrides;
        overrides.emplace_back(2, makeLocalScaledIdentityOp(2, 1.0));
        smith::BlockDiagonalPreconditioner P(offsets, std::move(solvers), std::move(overrides));
      },
      std::out_of_range);
}

TEST(BlockDiagonalPreconditionerCustom, ThrowsOnNullOverrideOperator)
{
  Array<int> offsets({0, 2, 4});
  EXPECT_THROW(
      {
        auto solvers = makeIdentitySolvers(2);
        std::vector<smith::BlockOverride> overrides;
        overrides.emplace_back(0, std::unique_ptr<const mfem::Operator>());
        smith::BlockDiagonalPreconditioner P(offsets, std::move(solvers), std::move(overrides));
      },
      std::invalid_argument);
}

TEST(BlockDiagonalPreconditionerCustom, ThrowsOnDuplicateOverrideIndex)
{
  Array<int> offsets({0, 2, 4});
  EXPECT_THROW(
      {
        auto solvers = makeIdentitySolvers(2);
        std::vector<smith::BlockOverride> overrides;
        overrides.emplace_back(0, makeLocalScaledIdentityOp(2, 1.0));
        overrides.emplace_back(0, makeLocalScaledIdentityOp(2, 2.0));
        smith::BlockDiagonalPreconditioner P(offsets, std::move(solvers), std::move(overrides));
      },
      std::invalid_argument);
}

TEST(BlockTriangularPreconditionerCustom, ThrowsOnOutOfRangeOverrideIndex)
{
  Array<int> offsets({0, 2, 4});
  EXPECT_THROW(
      {
        auto solvers = makeIdentitySolvers(2);
        std::vector<smith::BlockOverride> overrides;
        overrides.emplace_back(2, makeLocalScaledIdentityOp(2, 1.0));
        smith::BlockTriangularPreconditioner P(offsets, std::move(solvers), smith::BlockTriangularType::Lower,
                                               std::move(overrides));
      },
      std::out_of_range);
}

TEST(BlockTriangularPreconditionerCustom, ThrowsOnNullOverrideOperator)
{
  Array<int> offsets({0, 2, 4});
  EXPECT_THROW(
      {
        auto solvers = makeIdentitySolvers(2);
        std::vector<smith::BlockOverride> overrides;
        overrides.emplace_back(0, std::unique_ptr<const mfem::Operator>());
        smith::BlockTriangularPreconditioner P(offsets, std::move(solvers), smith::BlockTriangularType::Lower,
                                               std::move(overrides));
      },
      std::invalid_argument);
}

TEST(BlockTriangularPreconditionerCustom, ThrowsOnDuplicateOverrideIndex)
{
  Array<int> offsets({0, 2, 4});
  EXPECT_THROW(
      {
        auto solvers = makeIdentitySolvers(2);
        std::vector<smith::BlockOverride> overrides;
        overrides.emplace_back(1, makeLocalScaledIdentityOp(2, 1.0));
        overrides.emplace_back(1, makeLocalScaledIdentityOp(2, 2.0));
        smith::BlockTriangularPreconditioner P(offsets, std::move(solvers), smith::BlockTriangularType::Lower,
                                               std::move(overrides));
      },
      std::invalid_argument);
}

TEST(BlockSchurPreconditionerCustom, ThrowsOnOutOfRangeOverrideIndex)
{
  Array<int> offsets({0, 2, 4});
  EXPECT_THROW(
      {
        auto solvers = makeIdentitySolvers(2);
        std::vector<smith::BlockOverride> overrides;
        overrides.emplace_back(2, makeLocalScaledIdentityOp(2, 1.0));
        smith::BlockSchurPreconditioner P(offsets, std::move(solvers), smith::BlockSchurType::Diagonal,
                                          smith::SchurApproxType::A22Only, std::move(overrides));
      },
      std::out_of_range);
}

TEST(BlockSchurPreconditionerCustom, ThrowsOnNullOverrideOperator)
{
  Array<int> offsets({0, 2, 4});
  EXPECT_THROW(
      {
        auto solvers = makeIdentitySolvers(2);
        std::vector<smith::BlockOverride> overrides;
        overrides.emplace_back(0, std::unique_ptr<const mfem::Operator>());
        smith::BlockSchurPreconditioner P(offsets, std::move(solvers), smith::BlockSchurType::Diagonal,
                                          smith::SchurApproxType::A22Only, std::move(overrides));
      },
      std::invalid_argument);
}

TEST(BlockSchurPreconditionerCustom, ThrowsOnDuplicateOverrideIndex)
{
  Array<int> offsets({0, 2, 4});
  EXPECT_THROW(
      {
        auto solvers = makeIdentitySolvers(2);
        std::vector<smith::BlockOverride> overrides;
        overrides.emplace_back(1, makeLocalScaledIdentityOp(2, 1.0));
        overrides.emplace_back(1, makeLocalScaledIdentityOp(2, 2.0));
        smith::BlockSchurPreconditioner P(offsets, std::move(solvers), smith::BlockSchurType::Diagonal,
                                          smith::SchurApproxType::A22Only, std::move(overrides));
      },
      std::invalid_argument);
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
