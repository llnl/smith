#include "smith/numerics/block_preconditioner.hpp"
#include "mfem.hpp"
#include "axom/slic/core/SimpleLogger.hpp"
#include "axom/fmt.hpp"

namespace smith {

BlockDiagonalPreconditioner::BlockDiagonalPreconditioner(std::vector<std::unique_ptr<mfem::Solver>> solvers)
    : num_blocks_(static_cast<int>(solvers.size())), block_jacobian_(nullptr), mfem_solvers_(std::move(solvers))
{
}

void BlockDiagonalPreconditioner::Mult(const mfem::Vector& in, mfem::Vector& out) const { solver_diag_->Mult(in, out); }

void BlockDiagonalPreconditioner::SetOperator(const mfem::Operator& jacobian)
{
  height = jacobian.Height();
  width = jacobian.Width();
  // Cast the supplied jacobian to a block operator object
  block_jacobian_ = dynamic_cast<const mfem::BlockOperator*>(&jacobian);
  MFEM_VERIFY(block_jacobian_, "Jacobian must be a BlockOperator");

  SLIC_ERROR_ROOT_IF(
      block_jacobian_->NumRowBlocks() != num_blocks_ || block_jacobian_->NumColBlocks() != num_blocks_,
      axom::fmt::format("BlockDiagonalPreconditioner solver count ({}) must match block operator size ({}x{})",
                        num_blocks_, block_jacobian_->NumRowBlocks(), block_jacobian_->NumColBlocks()));

  block_offsets_.MakeRef(const_cast<mfem::Array<int>&>(block_jacobian_->RowOffsets()));
  solver_diag_ = std::make_unique<mfem::BlockOperator>(block_offsets_);

  // For each diagonal block A_ii, configure the corresponding solver
  for (int i = 0; i < num_blocks_; i++) {
    const mfem::Operator& A_ii = block_jacobian_->GetBlock(i, i);

    // Attach operator to solver
    mfem_solvers_[static_cast<size_t>(i)]->SetOperator(A_ii);

    // Place the solver into the diagonal block of solver_diag_
    solver_diag_->SetBlock(i, i, mfem_solvers_[static_cast<size_t>(i)].get());
  }
}

BlockDiagonalPreconditioner::~BlockDiagonalPreconditioner() {}

BlockTriangularPreconditioner::BlockTriangularPreconditioner(std::vector<std::unique_ptr<mfem::Solver>> solvers,
                                                             BlockTriangularType type)
    : num_blocks_(static_cast<int>(solvers.size())),
      block_jacobian_(nullptr),
      mfem_solvers_(std::move(solvers)),
      type_(type)
{
}

void BlockTriangularPreconditioner::LowerSweep(const mfem::Vector& in, mfem::Vector& out) const
{
  mfem::BlockVector b(const_cast<mfem::Vector&>(in), block_offsets_);
  mfem::BlockVector x(out, block_offsets_);

  // Forward sweep: i = 0 .. num_blocks_ - 1
  for (int i = 0; i < num_blocks_; i++) {
    mfem::Vector& bi = b.GetBlock(i);
    mfem::Vector& xi = x.GetBlock(i);

    // rhs_i = b_i
    mfem::Vector rhs_i(bi.Size());
    rhs_i = bi;

    // Subtract sum_{j < i} A_ij x_j
    for (int j = 0; j < i; j++) {
      if (block_jacobian_->IsZeroBlock(i, j)) {
        continue;  // no coupling
      }
      const mfem::Operator& A_ij = block_jacobian_->GetBlock(i, j);

      mfem::Vector tmp(rhs_i.Size());
      const mfem::Vector& xj = x.GetBlock(j);

      A_ij.Mult(xj, tmp);    // tmp = A_ij x_j
      rhs_i.Add(-1.0, tmp);  // rhs_i -= A_ij x_j
    }

    // Solve A_ii x_i = rhs_i with the i-th block solver
    mfem_solvers_[static_cast<size_t>(i)]->Mult(rhs_i, xi);
  }
}

void BlockTriangularPreconditioner::UpperSweep(const mfem::Vector& in, mfem::Vector& out) const
{
  mfem::BlockVector b(const_cast<mfem::Vector&>(in), block_offsets_);
  mfem::BlockVector x(out, block_offsets_);

  // Backward sweep: i = num_blocks_ - 1 .. 0
  for (int i = num_blocks_ - 1; i >= 0; i--) {
    mfem::Vector& bi = b.GetBlock(i);
    mfem::Vector& xi = x.GetBlock(i);

    // rhs_i = b_i
    mfem::Vector rhs_i(bi.Size());
    rhs_i = bi;

    // Subtract sum_{j > i} A_ij x_j
    for (int j = i + 1; j < num_blocks_; j++) {
      if (block_jacobian_->IsZeroBlock(i, j)) {
        continue;  // no coupling
      }
      const mfem::Operator& A_ij = block_jacobian_->GetBlock(i, j);

      mfem::Vector tmp(rhs_i.Size());
      const mfem::Vector& xj = x.GetBlock(j);

      A_ij.Mult(xj, tmp);    // tmp = A_ij x_j
      rhs_i.Add(-1.0, tmp);  // rhs_i -= A_ij x_j
    }

    // Solve A_ii x_i = rhs_i
    mfem_solvers_[static_cast<size_t>(i)]->Mult(rhs_i, xi);
  }
}

void BlockTriangularPreconditioner::Mult(const mfem::Vector& in, mfem::Vector& out) const
{
  switch (type_) {
    case BlockTriangularType::Lower:
      // x = P_lower^{-1} b
      LowerSweep(in, out);
      break;

    case BlockTriangularType::Upper:
      // x = P_upper^{-1} b
      UpperSweep(in, out);
      break;

    case BlockTriangularType::Symmetric: {
      // Symmetric: x = P_upper^{-1} D P_lower^{-1} b
      // 1) tmp = P_lower^{-1} b
      mfem::Vector tmp(out.Size());
      LowerSweep(in, tmp);

      // 2) tmp = D * tmp where D = diag(A_ii)
      {
        mfem::BlockVector tmp_block(tmp, block_offsets_);

        for (int i = 0; i < num_blocks_; i++) {
          mfem::Vector& tmp_i = tmp_block.GetBlock(i);
          mfem::Vector tmp_i_scaled(tmp_i.Size());

          const mfem::Operator& A_ii = block_jacobian_->GetBlock(i, i);
          A_ii.Mult(tmp_i, tmp_i_scaled);  // tmp_i_scaled = A_ii * tmp_i

          tmp_i = tmp_i_scaled;  // write back into block vector
        }
      }

      // 3) out = P_upper^{-1} tmp
      UpperSweep(tmp, out);
      break;
    }
  }
}

void BlockTriangularPreconditioner::SetOperator(const mfem::Operator& jacobian)
{
  height = jacobian.Height();
  width = jacobian.Width();
  // Cast the supplied jacobian to a block operator object
  block_jacobian_ = dynamic_cast<const mfem::BlockOperator*>(&jacobian);
  MFEM_VERIFY(block_jacobian_, "Jacobian must be a BlockOperator");

  SLIC_ERROR_ROOT_IF(
      block_jacobian_->NumRowBlocks() != num_blocks_ || block_jacobian_->NumColBlocks() != num_blocks_,
      axom::fmt::format("BlockTriangularPreconditioner solver count ({}) must match block operator size ({}x{})",
                        num_blocks_, block_jacobian_->NumRowBlocks(), block_jacobian_->NumColBlocks()));

  block_offsets_.MakeRef(const_cast<mfem::Array<int>&>(block_jacobian_->RowOffsets()));

  // Configure all diagonal solves
  for (int i = 0; i < num_blocks_; i++) {
    const mfem::Operator& A_ii = block_jacobian_->GetBlock(i, i);
    mfem_solvers_[static_cast<size_t>(i)]->SetOperator(A_ii);
  }
}

BlockTriangularPreconditioner::~BlockTriangularPreconditioner() {}

BlockSchurPreconditioner::BlockSchurPreconditioner(std::vector<std::unique_ptr<mfem::Solver>> solvers,
                                                   BlockSchurType type)
    : block_jacobian_(nullptr), mfem_solvers_(std::move(solvers)), type_(type)
{
}

void BlockSchurPreconditioner::LowerBlock(const mfem::Vector& in, mfem::Vector& out) const
{
  // Interpret in, out as block vectors: in = [b1; b2], out = [x1; x2]
  mfem::BlockVector b(const_cast<mfem::Vector&>(in), block_offsets_);
  mfem::BlockVector x(out, block_offsets_);

  mfem::Vector& b1 = b.GetBlock(0);
  mfem::Vector& b2 = b.GetBlock(1);
  mfem::Vector& x1 = x.GetBlock(0);
  mfem::Vector& x2 = x.GetBlock(1);

  // 1) Solve A11 x1 = b1
  mfem_solvers_[0]->Mult(b1, x1);

  // 2) Build x2 = b2 - A21 x1
  A_21_->Mult(x1, x2);  // x2 = A21 x1
  x2.Neg();             // x2 = -A21 x1
  x2 += b2;             // x2 = b2 - A21 x1

  // 3) Reassign x1.
  x1 = b1;
}

void BlockSchurPreconditioner::UpperBlock(const mfem::Vector& in, mfem::Vector& out) const
{
  // Interpret in, out as block vectors: in = [b1; b2], out = [x1; x2]
  mfem::BlockVector b(const_cast<mfem::Vector&>(in), block_offsets_);
  mfem::BlockVector x(out, block_offsets_);

  mfem::Vector& b1 = b.GetBlock(0);
  mfem::Vector& b2 = b.GetBlock(1);
  mfem::Vector& x1 = x.GetBlock(0);
  mfem::Vector& x2 = x.GetBlock(1);

  // 1) Build x1 = A12 b2
  mfem::Vector rhs1(b1.Size());
  A_12_->Mult(b2, rhs1);  // rhs1 = A12 b2

  // 2) Solve A11 x1 = rhs1
  mfem_solvers_[0]->Mult(rhs1, x1);

  // 3) Build b1 - A11^-1 A12 b2
  x1.Neg();  // x1 = -x1
  x1 += b1;  // = b1 - A12 x2

  // 4) Assign x2
  x2 = b2;
}

void BlockSchurPreconditioner::Mult(const mfem::Vector& in, mfem::Vector& out) const
{
  switch (type_) {
    case BlockSchurType::Diagonal: {
      // x = [A11^-1, 0; 0, S^-1] b
      solver_diag_->Mult(in, out);
      break;
    }

    case BlockSchurType::Lower: {
      // x = [A11^-1, 0; 0, S^-1][I, 0; -A21 A11^-1, I] b
      mfem::Vector tmp(out.Size());
      LowerBlock(in, tmp);
      solver_diag_->Mult(tmp, out);
      break;
    }

    case BlockSchurType::Upper: {
      // x = [I, -A11^-1 A12; 0, I][A11^-1, 0; 0, S^-1] b
      mfem::Vector tmp(out.Size());
      solver_diag_->Mult(in, tmp);
      UpperBlock(tmp, out);
      break;
    }

    case BlockSchurType::Full: {
      // x = [I, -A11^-1 A12; 0, I][A11^-1, 0; 0, S^-1][I, 0; -A21 A11^-1, I] b
      mfem::Vector tmp(out.Size());
      mfem::Vector tmp2(out.Size());
      LowerBlock(in, tmp);
      solver_diag_->Mult(tmp, tmp2);
      UpperBlock(tmp2, out);
      break;
    }
  }
}

void BlockSchurPreconditioner::SetOperator(const mfem::Operator& jacobian)
{
  height = jacobian.Height();
  width = jacobian.Width();
  block_jacobian_ = dynamic_cast<const mfem::BlockOperator*>(&jacobian);
  MFEM_VERIFY(block_jacobian_, "Jacobian must be a BlockOperator");

  SLIC_ERROR_ROOT_IF(block_jacobian_->NumRowBlocks() != 2 || block_jacobian_->NumColBlocks() != 2,
                     axom::fmt::format("BlockSchurPreconditioner requires a 2x2 block operator, got {}x{}",
                                       block_jacobian_->NumRowBlocks(), block_jacobian_->NumColBlocks()));
  SLIC_ERROR_ROOT_IF(
      mfem_solvers_.size() != 2,
      axom::fmt::format("BlockSchurPreconditioner requires exactly 2 solvers, got {}", mfem_solvers_.size()));

  block_offsets_.MakeRef(const_cast<mfem::Array<int>&>(block_jacobian_->RowOffsets()));
  if (!solver_diag_) {
    solver_diag_ = std::make_unique<mfem::BlockOperator>(block_offsets_);
  }

  auto* A11 = dynamic_cast<const mfem::HypreParMatrix*>(&block_jacobian_->GetBlock(0, 0));
  auto* A12 = dynamic_cast<const mfem::HypreParMatrix*>(&block_jacobian_->GetBlock(0, 1));
  auto* A21 = dynamic_cast<const mfem::HypreParMatrix*>(&block_jacobian_->GetBlock(1, 0));
  auto* A22 = dynamic_cast<const mfem::HypreParMatrix*>(&block_jacobian_->GetBlock(1, 1));

  MFEM_VERIFY(A11 && A12 && A21 && A22,
              "All blocks must be HypreParMatrix for assembled Schur complement preconditioner.");

  if (type_ == BlockSchurType::Lower || type_ == BlockSchurType::Full) {
    A_21_ = A21;
  }
  if (type_ == BlockSchurType::Upper || type_ == BlockSchurType::Full) {
    A_12_ = A12;
  }
  // Diagonal preconditioner for block (0,0)
  mfem_solvers_[0]->SetOperator(*A11);

  // Extract the diagonal of A11 (no inversion!)
  mfem::HypreParVector* Md = new mfem::HypreParVector(A11->GetComm(), A11->GetGlobalNumRows(), A11->GetRowStarts());
  A11->GetDiag(*Md);

  // Scale ROWS of A12 by Md^{-1}
  mfem::HypreParMatrix* A12_scaled = new mfem::HypreParMatrix(*A12);
  A12_scaled->InvScaleRows(*Md);
  delete Md;

  // Now compute A21 * (diag(A11)^{-1} * A12)
  mfem::HypreParMatrix* A21DinvA12 = mfem::ParMult(A21, A12_scaled);
  delete A12_scaled;

  // S_approx = A22 - A21 * diag(A11)^{-1} * A12
  S_approx_.reset(mfem::Add(1.0, *A22, -1.0, *A21DinvA12));
  delete A21DinvA12;

  // Set the Schur complement preconditioner for block (1,1)
  mfem_solvers_[1]->SetOperator(*S_approx_);

  // Set up block diagonal operator
  solver_diag_->SetBlock(0, 0, mfem_solvers_[0].get());
  solver_diag_->SetBlock(1, 1, mfem_solvers_[1].get());
}

BlockSchurPreconditioner::~BlockSchurPreconditioner() {}
}  // namespace smith
