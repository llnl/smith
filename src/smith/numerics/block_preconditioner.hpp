#pragma once

#include <memory>
#include <functional>
#include <vector>
#include "mfem.hpp"

namespace smith {

/**
 * @brief Optional override for a diagonal block operator.
 *
 * The integer is the block index i and the operator replaces the Jacobian block
 * A_ii (or, for 2x2 Schur systems, the block used to build/approximate the
 * (1,1) Schur operator).
 *
 * Ownership of the operator is transferred to the preconditioner.
 */
using BlockOverride = std::pair<int, std::unique_ptr<const mfem::Operator>>;

/**
 * @class BlockDiagonalPreconditioner
 * @brief Simple block diagonal preconditioner for block systems.
 *
 * Stores one solver per block and applies them to the diagonal blocks of a
 * block Jacobian.
 *
 * Call SetOperator() with an mfem::BlockOperator, then use Mult() to apply the
 * preconditioner.
 */
class BlockDiagonalPreconditioner : public mfem::Solver {
 public:
  /**
   * @brief Construct a new N by N block diagonal preconditioner.
   *
   * @param solvers One solver per block (size must match number of blocks).
   * @param overrides Optional list of (block index, operator) pairs used in place
   *        of the corresponding Jacobian diagonal block.
   */
  BlockDiagonalPreconditioner(std::vector<std::unique_ptr<mfem::Solver>> solvers,
                              std::vector<BlockOverride> overrides = {});

  /**
   * @brief The action of the precondition on the block vector (b_1, ..., b_n)
   *
   * @param in The block input vector (b_1, ..., b_n)
   * @param out The block output vector P^-1(b_1, ..., b_n)
   */
  virtual void Mult(const mfem::Vector& in, mfem::Vector& out) const;

  /**
   * @brief Set the preconditioner to use the supplied linearized block Jacobian
   *
   * @param jacobian The supplied linearized Jacobian. Note that it is always a block operator
   */
  virtual void SetOperator(const mfem::Operator& jacobian);

  virtual ~BlockDiagonalPreconditioner();

  int numSubSolvers() const { return num_blocks_; }

  mfem::Solver* subSolver(int i) const
  {
    MFEM_VERIFY(i >= 0 && i < num_blocks_, "BlockDiagonalPreconditioner::subSolver index out of range");
    return mfem_solvers_[static_cast<size_t>(i)].get();
  }

 private:
  // Offsets for extracting block vector segments, populated by SetOperator().
  mfem::Array<int> block_offsets_;

  // Number of blocks
  const int num_blocks_;

  // Jacobian view for block access
  const mfem::BlockOperator* block_jacobian_;

  // The diagonal part of the preconditioner containing BoomerAMG applications
  std::unique_ptr<mfem::BlockOperator> solver_diag_;

  // mfem solvers for each block
  mutable std::vector<std::unique_ptr<mfem::Solver>> mfem_solvers_;

  // size num_blocks_, nullptr means "use Jacobian diagonal block"
  std::vector<std::unique_ptr<const mfem::Operator>> block_op_overrides_;
};

/**
 * @enum BlockTriangularType
 * @brief Selects the block triangular sweep used by BlockTriangularPreconditioner.
 */
enum class BlockTriangularType
{
  Lower,    /**< Forward (lower triangular) sweep. */
  Upper,    /**< Backward (upper triangular) sweep. */
  Symmetric /**< Apply a symmetric combination of lower and upper sweeps. */
};

/**
 * @class BlockTriangularPreconditioner
 * @brief Simple block triangular preconditioner for block systems.
 *
 * Stores one solver per diagonal block and applies a block sweep using the
 * supplied block Jacobian.
 *
 * Call SetOperator() with an mfem::BlockOperator, then use Mult() to apply the
 * preconditioner.
 */
class BlockTriangularPreconditioner : public mfem::Solver {
 public:
  /**
   * @brief Construct a new nxn block triangular preconditioner.
   *
   * @param solvers One solver per diagonal block (size must match number of blocks).
   * @param type Sweep type (lower, upper, or symmetric).
   * @param overrides Optional list of (block index, operator) pairs used in place
   *        of the corresponding Jacobian diagonal block.
   */
  BlockTriangularPreconditioner(std::vector<std::unique_ptr<mfem::Solver>> solvers,
                                BlockTriangularType type = BlockTriangularType::Lower,
                                std::vector<BlockOverride> overrides = {});

  /**
   * @brief The action of the precondition on the block vector (b_1, ..., b_n)
   *
   * @param in The block input vector (b_1, ..., b_n)
   * @param out The block output vector P^-1(b_1, ..., b_n)
   */
  virtual void Mult(const mfem::Vector& in, mfem::Vector& out) const;

  /**
   * @brief Set the preconditioner to use the supplied linearized block Jacobian
   *
   * @param jacobian The supplied linearized Jacobian. Note that it is always a block operator
   */
  virtual void SetOperator(const mfem::Operator& jacobian);

  virtual ~BlockTriangularPreconditioner();

  int numSubSolvers() const { return num_blocks_; }

  mfem::Solver* subSolver(int i) const
  {
    MFEM_VERIFY(i >= 0 && i < num_blocks_, "BlockTriangularPreconditioner::subSolver index out of range");
    return mfem_solvers_[static_cast<size_t>(i)].get();
  }

 private:
  // Offsets for extracting block vector segments, populated by SetOperator().
  mfem::Array<int> block_offsets_;

  // Number of blocks
  const int num_blocks_;

  // Jacobian view for block access
  const mfem::BlockOperator* block_jacobian_;

  // mfem solvers for each block
  mutable std::vector<std::unique_ptr<mfem::Solver>> mfem_solvers_;

  // Block Triangular type
  BlockTriangularType type_;

  /**
   * @brief The action of the lower sweep on the block vector (b_1, ..., b_n)
   *
   * @param in The block input vector (b_1, ..., b_n)
   * @param out The block output vector P_lower^-1(b_1, ..., b_n)
   */
  void LowerSweep(const mfem::Vector& in, mfem::Vector& out) const;

  /**
   * @brief The action of the upper sweep on the block vector (b_1, ..., b_n)
   *
   * @param in The block input vector (b_1, ..., b_n)
   * @param out The block output vector P_upper^-1(b_1, ..., b_n)
   */
  void UpperSweep(const mfem::Vector& in, mfem::Vector& out) const;

  // size num_blocks_, nullptr means "use Jacobian diagonal block"
  std::vector<std::unique_ptr<const mfem::Operator>> block_op_overrides_;
};

/**
 * @enum BlockSchurType
 * @brief Selects the block Schur preconditioner variant.
 */
enum class BlockSchurType
{
  Diagonal, /**< Block diagonal: apply $ A_{11}^{-1} $ and $ S^{-1} $ only. */
  Lower,    /**< Lower factor form. */
  Upper,    /**< Upper factor form. */
  Full      /**< Full factor form (lower, diagonal, upper). */
};

/**
 * @enum SchurApproxType
 * @brief Selects how the (1,1) Schur operator is approximated.
 */
enum class SchurApproxType
{
  DiagInv, /**< Use assembled \f$ S \approx A_{22} - A_{21} \\mathrm{diag}(A_{11})^{-1} A_{12} \f$. */
  A22Only, /**< Use \f$ S \approx A_{22} \f$. */
  Custom,  /**< Use a custom operator provided via the overrides list for block index 1. */
};

/**
 * @class BlockSchurPreconditioner
 * @brief Simple 2x2 block Schur complement preconditioner for block systems.
 *
 * Uses two solvers, one for $ A_{11} $ and one for an approximate Schur complement $ S $.
 * Call SetOperator() with an mfem::BlockOperator, then use Mult() to apply the
 * selected Schur preconditioner type.
 */
class BlockSchurPreconditioner : public mfem::Solver {
 public:
  /**
   * @brief Construct a new 2x2 block Schur complement preconditioner.
   *
   * @param solvers Two solvers, for $ A_{11} $ and the Schur complement approximation.
   * @param type Preconditioner variant (diagonal, lower, upper, or full).
   * @param approxType Schur complement approximation strategy for the (1,1) block.
   * @param overrides Optional list of (block index, operator) pairs used in place
   *        of the corresponding Jacobian diagonal block. For Schur systems, index
   *        0 overrides $A_{11}$ and index 1 provides a custom Schur operator when
   *        approxType is SchurApproxType::Custom.
   */
  BlockSchurPreconditioner(std::vector<std::unique_ptr<mfem::Solver>> solvers,
                           BlockSchurType type = BlockSchurType::Diagonal,
                           SchurApproxType approxType = SchurApproxType::DiagInv,
                           std::vector<BlockOverride> overrides = {});

  /**
   * @brief The action of the precondition on the block vector (b_1, b_2)
   *
   * @param in The block input vector (b_1, b_2)
   * @param out The block output vector P^-1(b_1, b_2)
   */
  virtual void Mult(const mfem::Vector& in, mfem::Vector& out) const;

  /**
   * @brief Set the preconditioner to use the supplied linearized block Jacobian.
   *
   * The Schur complement approximation is given by S_approx = A22 - A21 * diag(A11)^{-1} * A12
   *
   * @param jacobian The supplied linearized Jacobian. Note that it is always a block operator
   */
  virtual void SetOperator(const mfem::Operator& jacobian);

  virtual ~BlockSchurPreconditioner();

  int numSubSolvers() const { return static_cast<int>(mfem_solvers_.size()); }

  mfem::Solver* subSolver(int i) const
  {
    MFEM_VERIFY(i >= 0 && i < static_cast<int>(mfem_solvers_.size()), "BlockSchurPreconditioner::subSolver index out of range");
    return mfem_solvers_[static_cast<size_t>(i)].get();
  }

 private:
  // Offsets for extracting block vector segments, populated by SetOperator().
  mfem::Array<int> block_offsets_;

  // Jacobian view for block access
  const mfem::BlockOperator* block_jacobian_;

  // The diagonal part of the preconditioner containing BoomerAMG applications
  std::unique_ptr<mfem::BlockOperator> solver_diag_;

  // mfem solvers for each block
  mutable std::vector<std::unique_ptr<mfem::Solver>> mfem_solvers_;

  // Views of the linearized Jacobian blocks
  const mfem::Operator* A_12_ = nullptr;
  const mfem::Operator* A_21_ = nullptr;

  // Schur complement approximation operator used by solver for block (1,1).
  //
  // For DiagInv and A22Only, the approximation is rebuilt on each SetOperator call and stored in
  // S_approx_owned_. For Custom, the approximation is provided via block_op_overrides_[1] and referenced
  // non-owningly via S_approx_view_.
  mutable std::unique_ptr<const mfem::Operator> S_approx_owned_;
  const mfem::Operator* S_approx_view_ = nullptr;

  BlockSchurType type_;

  SchurApproxType approxType_;

  /**
   * @brief The action of the lower sweep on the block vector (b_1, b_2)
   *
   * @param in The block input vector (b_1, b_2)
   * @param out The block output vector [I, 0; -A21 A11^-1, I] (b_1, b_2)
   */
  void LowerBlock(const mfem::Vector& in, mfem::Vector& out) const;

  /**
   * @brief The action of the upper block on the block vector (b_1, b_2)
   *
   * @param in The block input vector (b_1, b_2)
   * @param out The block output vector [I - A11^-1 A12; 0, I](b_1, b_2)
   */
  void UpperBlock(const mfem::Vector& in, mfem::Vector& out) const;

  // size num_blocks_, nullptr means "use Jacobian diagonal block"
  std::vector<std::unique_ptr<const mfem::Operator>> block_op_overrides_;

  mfem::HypreParMatrix* BuildSchurDiagApprox_(const mfem::HypreParMatrix& A11, const mfem::HypreParMatrix& A12,
                                              const mfem::HypreParMatrix& A21, const mfem::HypreParMatrix& A22) const;
};
}  // namespace smith
