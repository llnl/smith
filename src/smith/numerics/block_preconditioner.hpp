#pragma once

#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "mfem.hpp"

#include "smith/numerics/state_dependent_solver.hpp"

namespace smith {
/**
 * @brief Supplies the current concrete operator for a block solver override.
 */
class BlockOperatorProvider {
 public:
  virtual ~BlockOperatorProvider() = default;

  /**
   * @brief Refresh the owned operator for the current nonlinear state.
   * @param state Monolithic state vector at the current nonlinear iterate.
   * @param block_offsets Offsets describing the block layout of @a state.
   */
  virtual void updateForState([[maybe_unused]] const mfem::Vector& state,
                              [[maybe_unused]] const mfem::Array<int>& block_offsets)
  {
  }

  /**
   * @brief Return the current concrete operator.
   */
  virtual const mfem::Operator& currentOperator() const = 0;
};

/**
 * @brief Builder that rebuilds an operator from the current nonlinear state.
 *
 * Builders are invoked by BlockPreconditioner::updateForState(). In nonlinear
 * block solves created with a custom state-dependent preconditioner, that update
 * is wired into the Newton loop before the preconditioner is configured with the
 * current Jacobian.
 */
using StateDependentBlockOperatorBuilder =
    std::function<std::unique_ptr<mfem::Operator>(const mfem::Vector&, const mfem::Array<int>&)>;

/**
 * @brief Optional provider override for a diagonal block operator.
 *
 * The block index i identifies the provider that supplies the operator used
 * in place of the Jacobian block A_ii. For 2x2 Schur systems, index 1 supplies
 * the custom Schur operator when approxType is SchurApproxType::Custom.
 */
struct BlockProviderOverride {
  /**
   * @brief Construct from a block index and owned provider.
   * @param block_index_in Block index to override.
   * @param provider_in Provider supplying the override operator.
   */
  BlockProviderOverride(int block_index_in, std::unique_ptr<BlockOperatorProvider> provider_in)
      : block_index(block_index_in), provider(std::move(provider_in))
  {
  }

  int block_index;                                  ///< Block index to override.
  std::unique_ptr<BlockOperatorProvider> provider;  ///< Provider supplying the override operator.
};

/**
 * @brief Build an override from a fixed concrete operator.
 * @param block_index Block index to override.
 * @param op Fixed concrete operator.
 */
BlockProviderOverride makeFixedBlockProviderOverride(int block_index, std::unique_ptr<const mfem::Operator> op);

/**
 * @brief Build an override from a state-dependent operator builder.
 * @param block_index Block index to override.
 * @param builder Callable that returns a new concrete operator for a state.
 * @param initial_operator Optional operator to use before the first update.
 *
 * The builder is called from BlockPreconditioner::updateForState() with the
 * current monolithic nonlinear state and its block offsets. The rebuilt operator
 * is then used the next time SetOperator() configures the block solvers.
 *
 * Provide @a initial_operator when SetOperator() may be called before the first
 * state update. Otherwise, currentOperator() will fail until updateForState()
 * has produced an operator.
 */
BlockProviderOverride makeStateDependentBlockProviderOverride(
    int block_index, StateDependentBlockOperatorBuilder builder,
    std::unique_ptr<mfem::Operator> initial_operator = nullptr);

/**
 * @class BlockPreconditioner
 * @brief Base class for block preconditioners that own one sub-solver per block.
 */
class BlockPreconditioner : public mfem::Solver, public StateDependentSolver {
 public:
  /** @brief Return the number of sub-solvers owned by this preconditioner. */
  int numSubSolvers() const { return num_blocks_; }

  /**
   * @brief Access a sub-solver by index.
   * @param i Sub-solver index in [0, numSubSolvers()).
   * @return Pointer to the requested sub-solver (owned by this object).
   */
  mfem::Solver* subSolver(int i) const
  {
    MFEM_VERIFY(i >= 0 && i < num_blocks_, "BlockPreconditioner::subSolver index out of range");
    return mfem_solvers_[static_cast<size_t>(i)].get();
  }

  /// @overload
  void updateForState(const mfem::Vector& state, const mfem::Array<int>& block_offsets) override;

  virtual ~BlockPreconditioner();

 protected:
  /**
   * @brief Construct a block preconditioner from one owned solver per block.
   * @param solvers Sub-solvers owned by this preconditioner.
   */
  explicit BlockPreconditioner(std::vector<std::unique_ptr<mfem::Solver>> solvers);

  /// @brief Offsets for extracting block vector segments, populated by SetOperator().
  mfem::Array<int> block_offsets_;

  /// @brief Number of blocks in the block system.
  const int num_blocks_;

  /// @brief Non-owning view of the block Jacobian supplied by SetOperator().
  const mfem::BlockOperator* block_jacobian_;

  /// @brief Owned MFEM solver for each block.
  mutable std::vector<std::unique_ptr<mfem::Solver>> mfem_solvers_;

  /// @brief Per-block operator providers; null entries use the corresponding Jacobian diagonal block.
  std::vector<std::unique_ptr<BlockOperatorProvider>> block_op_providers_;
};

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
class BlockDiagonalPreconditioner : public BlockPreconditioner {
 public:
  /**
   * @brief Construct a new N by N block diagonal preconditioner.
   *
   * @param solvers One solver per block (size must match number of blocks).
   * @param overrides Optional provider overrides used in place of the
   *        corresponding Jacobian diagonal blocks.
   */
  BlockDiagonalPreconditioner(std::vector<std::unique_ptr<mfem::Solver>> solvers,
                              std::vector<BlockProviderOverride> overrides = {});

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
  // The diagonal part of the preconditioner containing BoomerAMG applications
  std::unique_ptr<mfem::BlockOperator> solver_diag_;
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
class BlockTriangularPreconditioner : public BlockPreconditioner {
 public:
  /**
   * @brief Construct a new nxn block triangular preconditioner.
   *
   * @param solvers One solver per diagonal block (size must match number of blocks).
   * @param type Sweep type (lower, upper, or symmetric).
   * @param overrides Optional provider overrides used in place of the
   *        corresponding Jacobian diagonal blocks.
   */
  BlockTriangularPreconditioner(std::vector<std::unique_ptr<mfem::Solver>> solvers,
                                BlockTriangularType type = BlockTriangularType::Lower,
                                std::vector<BlockProviderOverride> overrides = {});

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
 * @enum SchurApproxType
 * @brief Selects how the (1,1) Schur operator is approximated.
 */
enum class SchurApproxType
{
  DiagInv, /**< Use assembled \f$ S \approx A_{22} - A_{21} \\mathrm{diag}(A_{11})^{-1} A_{12} \f$. */
  A22Only, /**< Use \f$ S \approx A_{22} \f$. */
  Custom   /**< Use a custom operator provider for block index 1. */
};

/**
 * @class BlockSchurPreconditioner
 * @brief Simple 2x2 block Schur complement preconditioner for block systems.
 *
 * Uses two solvers, one for $ A_{11} $ and one for an approximate Schur complement $ S $.
 * Call SetOperator() with an mfem::BlockOperator, then use Mult() to apply the
 * selected Schur preconditioner type.
 */
class BlockSchurPreconditioner : public BlockPreconditioner {
 public:
  /**
   * @brief Construct a new 2x2 block Schur complement preconditioner.
   *
   * @param solvers Two solvers, for $ A_{11} $ and the Schur complement approximation.
   * @param type Preconditioner variant (diagonal, lower, upper, or full).
   * @param approxType Schur complement approximation strategy for the (1,1) block.
   * @param overrides Optional provider overrides. Index 0 overrides $A_{11}$ and
   *        index 1 provides a custom Schur operator when approxType is SchurApproxType::Custom.
   */
  BlockSchurPreconditioner(std::vector<std::unique_ptr<mfem::Solver>> solvers,
                           BlockSchurType type = BlockSchurType::Diagonal,
                           SchurApproxType approxType = SchurApproxType::DiagInv,
                           std::vector<BlockProviderOverride> overrides = {});

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
  // The diagonal part of the preconditioner containing BoomerAMG applications
  std::unique_ptr<mfem::BlockOperator> solver_diag_;

  // Views of the linearized Jacobian blocks
  const mfem::Operator* A_12_ = nullptr;
  const mfem::Operator* A_21_ = nullptr;

  // Schur complement approximation operator used by solver for block (1,1).
  //
  // For DiagInv and A22Only, the approximation is rebuilt on each SetOperator call and stored in
  // S_approx_owned_. For Custom, the approximation is provided via block_op_providers_[1] and referenced
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

  mfem::HypreParMatrix* BuildSchurDiagApprox_(const mfem::HypreParMatrix& A11, const mfem::HypreParMatrix& A12,
                                              const mfem::HypreParMatrix& A21, const mfem::HypreParMatrix& A22) const;
};
}  // namespace smith
