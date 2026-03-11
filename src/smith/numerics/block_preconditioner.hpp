#pragma once

#include <memory>
#include <functional>
#include "mfem.hpp"

namespace smith {

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
   * @param offsets Offsets describing the block layout.
   * @param solvers One solver per block (size must match number of blocks).
   */
  BlockDiagonalPreconditioner(mfem::Array<int>& offsets, std::vector<std::unique_ptr<mfem::Solver>> solvers);

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

 private:
  // Offsets for extracting block vector segments
  mfem::Array<int>& block_offsets_;

  // Number of blocks
  const int n_blocks_;

  // Jacobian view for block access
  const mfem::BlockOperator* block_jacobian_;

  // The diagonal part of the preconditioner containing BoomerAMG applications
  mfem::BlockOperator solver_diag_;

  // mfem solvers for each block
  mutable std::vector<std::unique_ptr<mfem::Solver>> mfem_solvers_;
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
   * @param offsets Offsets describing the block layout.
   * @param solvers One solver per diagonal block (size must match number of blocks).
   * @param type Sweep type (lower, upper, or symmetric).
   */
  BlockTriangularPreconditioner(mfem::Array<int>& offsets, std::vector<std::unique_ptr<mfem::Solver>> solvers,
                                BlockTriangularType type = BlockTriangularType::Lower);

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

 private:
  // Offsets for extracting block vector segments
  mfem::Array<int>& block_offsets_;

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
   * @param offsets Offsets describing the 2-block layout.
   * @param solvers Two solvers, for $ A_{11} $ and the Schur complement approximation.
   * @param type Preconditioner variant (diagonal, lower, upper, or full).
   */
  BlockSchurPreconditioner(mfem::Array<int>& offsets, std::vector<std::unique_ptr<mfem::Solver>> solvers,
                           BlockSchurType type = BlockSchurType::Diagonal);

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

 private:
  // Offsets for extracting block vector segments
  mfem::Array<int>& block_offsets_;

  // Jacobian view for block access
  const mfem::BlockOperator* block_jacobian_;

  // The diagonal part of the preconditioner containing BoomerAMG applications
  mfem::BlockOperator solver_diag_;

  // mfem solvers for each block
  mutable std::vector<std::unique_ptr<mfem::Solver>> mfem_solvers_;

  // Views of the linearized Jacobian blocks
  const mfem::Operator *A_12_, *A_21_;

  mutable mfem::HypreParMatrix* S_approx_;

  BlockSchurType type_;

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
};
}  // namespace smith
