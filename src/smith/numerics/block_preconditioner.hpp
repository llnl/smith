#pragma once

#include <memory>
#include <functional>
#include "mfem.hpp"

namespace smith {

class BlockDiagonalPreconditioner : public mfem::Solver {
 public:
  BlockDiagonalPreconditioner(mfem::Array<int>& offsets, std::vector<std::unique_ptr<mfem::Solver>> solvers);

  virtual void Mult(const mfem::Vector& in, mfem::Vector& out) const;

  virtual void SetOperator(const mfem::Operator& jacobian);

  virtual ~BlockDiagonalPreconditioner();

 private:
  // Offsets for extracting block vector segments
  mfem::Array<int>& block_offsets_;

  // Number of blocks
  const int nblocks_;

  // Jacobian view for block access
  const mfem::BlockOperator* block_jacobian_;

  // The diagonal part of the preconditioner containing BoomerAMG applications
  mfem::BlockOperator solver_diag_;

  // mfem solvers for each block
  mutable std::vector<std::unique_ptr<mfem::Solver>> mfem_solvers_;
};

enum class BlockTriangularType
{
  Lower,
  Upper,
  Symmetric
};

class BlockTriangularPreconditioner : public mfem::Solver {
 public:
  BlockTriangularPreconditioner(mfem::Array<int>& offsets, std::vector<std::unique_ptr<mfem::Solver>> solvers,
                                BlockTriangularType type = BlockTriangularType::Lower);

  virtual void Mult(const mfem::Vector& in, mfem::Vector& out) const;

  virtual void SetOperator(const mfem::Operator& jacobian);

  virtual ~BlockTriangularPreconditioner();

 private:
  // Offsets for extracting block vector segments
  mfem::Array<int>& block_offsets_;

  // Number of blocks
  const int nblocks_;

  // Jacobian view for block access
  const mfem::BlockOperator* block_jacobian_;

  // mfem solvers for each block
  mutable std::vector<std::unique_ptr<mfem::Solver>> mfem_solvers_;

  // Block Triangular type
  BlockTriangularType type_;

  // Lower and Upper sweeps to be used in Mult
  void LowerSweep(const mfem::Vector& in, mfem::Vector& out) const;
  void UpperSweep(const mfem::Vector& in, mfem::Vector& out) const;
};

enum class BlockSchurType
{
  Diagonal,
  Lower,
  Upper,
  Full
};

class BlockSchurPreconditioner : public mfem::Solver {
 public:
  BlockSchurPreconditioner(mfem::Array<int>& offsets, std::vector<std::unique_ptr<mfem::Solver>> solvers,
                           BlockSchurType type = BlockSchurType::Diagonal);

  virtual void Mult(const mfem::Vector& in, mfem::Vector& out) const;

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

  // Lower and Upper blocks to be used in Mult
  void LowerBlock(const mfem::Vector& in, mfem::Vector& out) const;
  void UpperBlock(const mfem::Vector& in, mfem::Vector& out) const;
};
}  // namespace smith
