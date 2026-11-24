#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>
#include <mpi.h>

#include "smith/numerics/equation_solver.hpp"
#include "smith/numerics/stdfunction_operator.hpp"
#include "smith/physics/weak_form.hpp"
#include "smith/physics/state/finite_element_dual.hpp"
#include "smith/physics/state/finite_element_state.hpp"
#include "mfem.hpp"

namespace smith {

namespace composable {
struct auxiliary_equation_properties {
  bool independent_block_diagonal = false;
  bool matrix_free_solve = false;
  // Reuse available auxiliary precomputed data until explicitly reset.
  bool skip_recomputation = false;
  bool symmetric = false;
  // Absolute entry threshold for the general Jacobian correction.
  double jacobian_drop_tolerance = 1.0e-10;
};

class BlockInverses {
public:
  mfem::Array<int> block_offsets;
private:
  mfem::Array<int> factor_offsets;
  mfem::Vector factor_data;
  mfem::Array<int> pivot_data;
  bool symmetric = false;

 public:
  using Factorization = std::variant<mfem::LUFactors, mfem::CholeskyFactors>;

  /// @brief default constructor: empty
  BlockInverses() = default;

  /// @brief construct the element inverses for a finite element space
  /// of a sparse matrix that has block independent structure
  /// @param symmetric Use Cholesky; blocks must be symmetric positive definite.
  BlockInverses(const mfem::Array<int>& offsets, const mfem::HypreParMatrix& mat, bool symmetric) : symmetric{symmetric}
  {
    computeFactorization(offsets, mat);
  }

  /// @brief Replace the stored factors, retaining the LU/Cholesky choice.
  void computeFactorization(const mfem::Array<int>& offsets, const mfem::HypreParMatrix& mat)
  {
    SLIC_ERROR_IF(offsets.Size() == 0 || offsets[0] != 0 || offsets.Last() != mat.Height() || mat.Height() != mat.Width() ||
        mat.GetGlobalNumRows() != mat.GetGlobalNumCols(),
     "BlockInverses: offsets must partition a square matrix"
    );
    block_offsets = offsets;
    factor_offsets.SetSize(offsets.Size());
    factor_offsets[0] = 0;
    for (int b = 0; b + 1 < offsets.Size(); ++b) {
      const int n = offsets[b + 1] - offsets[b];
      SLIC_ERROR_IF(n <= 0, "BlockInverses: block sizes must be positive");
      factor_offsets[b + 1] = factor_offsets[b] + n * n;
    }
    factor_data.SetSize(factor_offsets.Last());
    pivot_data.SetSize(symmetric ? 0 : offsets.Last());

    // NOTE: we assume no off diagonal entries
    // Checking this is expensive so we trust the user
    mfem::SparseMatrix diag;
    HYPRE_BigInt* column_map;
    mat.GetDiag(diag);

    const auto* row_offsets = diag.HostReadI();
    const auto* columns = diag.HostReadJ();
    const auto* values = diag.HostReadData();
    for (int iblock = 0; iblock < block_offsets.Size() - 1; ++iblock) {
      int factor_offset = factor_offsets[iblock];
      int block_start = block_offsets[iblock];
      int block_end = block_offsets[iblock + 1];
      int pivot_offset = block_start;
      int n = block_end - block_start;

      mfem::DenseMatrix block(factor_data.GetData() + factor_offset, n, n);
      block = 0.0;

      for (int i = block_start; i < block_end; ++i) {
        for (int k = row_offsets[i]; k < row_offsets[i + 1]; ++k) {
          block(i - block_start, columns[k] - block_start) = values[k];
        }
      }

      // factorize the block
      bool success;
      if (symmetric) {
        mfem::CholeskyFactors factorization{factor_data.GetData() + factor_offset};
        success = factorization.Factor(n);
      } else {
        mfem::LUFactors factorization{factor_data.GetData() + factor_offset, pivot_data.GetData() + pivot_offset};
        success = factorization.Factor(n);
      }
      SLIC_ERROR_IF( not success, "BlockInverses: factorization failed on block " + std::to_string(iblock));
    }
  }

  // @brief get the factorization of the given block
  // @param block index the index of the block to get a factorization for
  // @return a non-owning factorization and the size of the block
  [[nodiscard]]
  std::pair<Factorization, int> getFactorization(int block_index)
  {
    SLIC_ERROR_IF(block_index < 0 || block_index + 1 >= block_offsets.Size(),
      "BlockInverses: invalid block index");
    int factor_offset = factor_offsets[block_index];
    int block_start = block_offsets[block_index];
    int block_end = block_offsets[block_index + 1];
    int pivot_offset = block_start;
    int n = block_end - block_start;
    if (symmetric) {
      return {Factorization{mfem::CholeskyFactors{factor_data.GetData() + factor_offset}}, n};
    } else {
      return {
          Factorization{mfem::LUFactors{factor_data.GetData() + factor_offset, pivot_data.GetData() + pivot_offset}},
          n};
    }
  }

};
}  // namespace composable

/// Base for a weak form whose residual depends on solved auxiliary fields.
/// For the three-field solid, the primary form is momentum and the two
/// auxiliary entries can hold the kinematic (e_v) and constitutive (p) forms.
/// Auxiliary constraints are independent; nesting compositions represents dependencies.
template <std::size_t NumAuxiliaryFields>
class ComposableWeakForm : public WeakForm {
 public:
  // shared pointer for a field (the finite element state)
  using FieldPtr = std::shared_ptr<FiniteElementState>;

  // The primary weak form takes auxiliary variables that get eliminated internally
  // Contains:
  // A field as a mutable unknown and initial guess for auxiliary solve
  // A weak form for the constraint that enforces the auxiliary variable connection
  // to primary variables
  // A solver that can solve the constraint
  // Properties to inform optimization
  using AuxiliarySolve = std::tuple<FieldPtr, std::shared_ptr<WeakForm>, std::shared_ptr<EquationSolver>,
                                    composable::auxiliary_equation_properties>;

  // Array of all the auxiliary solves
  using AuxiliarySolves = std::array<AuxiliarySolve, NumAuxiliaryFields>;

  // helper precomputed data that can aid auxiliary solves and gradients
  using AuxiliarySolveHelper = std::variant<std::monostate, composable::BlockInverses>;

 protected:

  using HypreMatPtr = std::unique_ptr<mfem::HypreParMatrix>;

  std::shared_ptr<WeakForm> primary_weak_form_;
  AuxiliarySolves auxiliary_solves_;
  mutable std::array<AuxiliarySolveHelper, NumAuxiliaryFields> aux_precomputed_data_;

 public:
  /// @brief make a composable weak form
  /// @param name a human-readable name for this weak form
  /// @param primary_weak_form a weak form that takes primary and auxiliary variables.
  ///   The auxiliary variable fields are appended to the @p fields argument of
  ///   @ref smith::WeakForm::residual "WeakForm::residual()".
  /// @param auxiliary_solves tuples of {field, weak form, solver, properties}
  ///   for computing auxiliary variables
  ///   Each auxiliary weak form is independent and takes the incoming fields,
  ///   followed by its own trial field for the @p fields argument of
  ///   @ref smith::WeakForm::residual "WeakForm::residual()".
  ComposableWeakForm(std::string name, std::shared_ptr<WeakForm> primary_weak_form, AuxiliarySolves auxiliary_solves)
      : WeakForm(std::move(name)),
        primary_weak_form_(std::move(primary_weak_form)),
        auxiliary_solves_(std::move(auxiliary_solves))
  {
  }

  /// @brief Clear cached auxiliary solve data so it is rebuilt on the next use.
  /// @note Call on all MPI ranks after changes that invalidate cached Jacobians
  /// or auxiliary DOF layouts. Auxiliary nonlinear solves still run normally.
  void reset_auxiliary_precomputed_solve() const
  {
    for (auto& precomputed_data : aux_precomputed_data_) {
      precomputed_data = std::monostate{};
    }
  }

  // Include the response of each auxiliary field in every derivative.
  mfem::Vector residual(const TimeInfo& time_info, ConstFieldPtr shape_disp, const std::vector<ConstFieldPtr>& fields,
                        const std::vector<ConstQuadratureFieldPtr>& quad_fields = {}) const override
  {
    solveAuxiliaryFields(time_info, shape_disp, fields, quad_fields);
    return primary_weak_form_->residual(time_info, shape_disp, append_all_aux(fields), quad_fields);
  }

  std::unique_ptr<mfem::HypreParMatrix> jacobian(
      const TimeInfo& time_info, ConstFieldPtr shape_disp, const std::vector<ConstFieldPtr>& fields,
      const std::vector<double>& field_argument_tangents,
      const std::vector<ConstQuadratureFieldPtr>& quad_fields = {}) const override
  {
    // For a given auxiliary field p we effectively have the following block matrix 
    //
    // | ∂_u J     ∂_p J   |
    // | ∂_u C_p   ∂_p C_p |
    //
    // This results in a Schur correction of 
    // ∂_u J - ∂_p J (∂_p C_p)^-1 ∂_u C_p

    if (field_argument_tangents.size() != fields.size()) {
      throw std::invalid_argument("ComposableWeakForm: tangent count must match input field count");
    }

    // solve for auxiliary fields p^*
    solveAuxiliaryFields(time_info, shape_disp, fields, quad_fields);

    auto all_fields = append_all_aux(fields);

    // Compute ∂_u J
    auto primary_tangents = field_argument_tangents;
    primary_tangents.resize(all_fields.size(), 0.0);
    auto dJ_du = primary_weak_form_->jacobian(time_info, shape_disp, all_fields, primary_tangents, quad_fields);

    // if all all tangents are 0, the primary jacobian will give a 0 matrix 
    // replicate that behavior
    bool has_tangent = false;
    for (double tangent : field_argument_tangents) has_tangent |= tangent != 0.0;
    if (!has_tangent) return dJ_du;
    SLIC_ERROR_IF(!dJ_du, "ComposableWeakForm: missing primary Jacobian");

    // Get the Schur Complement contribution for each Auxiliary field
    for (std::size_t k = 0; k < NumAuxiliaryFields; ++k) {
      const auto& [aux_field, aux_form, solver, properties] = auxiliary_solves_[k];
      auto auxiliary_inputs = append_aux(fields, aux_field.get());

      mfem::Array<int> block_offsets;
      const bool use_blocks = properties.independent_block_diagonal &&
                              elementBlockOffsets(aux_field->space(), block_offsets);
      auto& precomputed_data = aux_precomputed_data_[k];
      const bool reuse_factors = use_blocks && properties.skip_recomputation &&
                                 std::holds_alternative<composable::BlockInverses>(precomputed_data);

      // Compute ∂_p C_p when its factorization cannot be reused.
      HypreMatPtr dCp_dp;
      if (!reuse_factors) {
        dCp_dp = auxiliaryJacobian(*aux_form, time_info, shape_disp, auxiliary_inputs, quad_fields);
        SLIC_ERROR_IF(!dCp_dp, "ComposableWeakForm: missing auxiliary self Jacobian");
      }

      // Compute ∂_u C_p
      auto input_tangents = field_argument_tangents;
      input_tangents.push_back(0.0);
      auto dCp_du= aux_form->jacobian(time_info, shape_disp, auxiliary_inputs, input_tangents, quad_fields);
      // Compute ∂_p J
      std::vector<double> auxiliary_tangents(all_fields.size(), 0.0);
      auxiliary_tangents[fields.size() + k] = 1.0;
      auto dJ_dp=
          primary_weak_form_->jacobian(time_info, shape_disp, all_fields, auxiliary_tangents, quad_fields);

      // error check the remaining jacobian computations returned something
      SLIC_ERROR_IF(!dJ_dp|| !dCp_du,
        "ComposableWeakForm: missing auxiliary Jacobian");

      // form the Shur Complement contribution 
      std::unique_ptr<mfem::HypreParMatrix> correction;
      if (use_blocks) {
        // Factor C_p once per element when needed and solve the columns touching it.
        if (!reuse_factors) {
          precomputed_data.template emplace<composable::BlockInverses>(block_offsets, *dCp_dp, properties.symmetric);
        }
        auto& dCp_dp_factors = std::get<composable::BlockInverses>(precomputed_data);
        correction = blockCorrection(aux_field->space(), dCp_dp_factors, *dCp_du, *dJ_dp);
      } else {
        // General case: one collective forward solve per global input column.
        correction = generalCorrection(*dCp_dp, *dCp_du, *dJ_dp, *solver, properties.jacobian_drop_tolerance);
      }
      // Elimination can introduce entries outside the sparsity pattern of J_u.
      dJ_du.reset(mfem::Add(1.0, *dJ_du, 1.0, *correction));
    }
    dJ_du->CopyRowStarts();
    dJ_du->CopyColStarts();
    return dJ_du;
  }

  void jvp(const TimeInfo&, ConstFieldPtr, const std::vector<ConstFieldPtr>&,
           const std::vector<ConstQuadratureFieldPtr>&, ConstFieldPtr, const std::vector<ConstFieldPtr>&,
           const std::vector<ConstQuadratureFieldPtr>&, DualFieldPtr) const override
  {
    throw std::logic_error("ComposableWeakForm::jvp is not implemented");
  }

  void vjp(const TimeInfo&, ConstFieldPtr, const std::vector<ConstFieldPtr>&,
           const std::vector<ConstQuadratureFieldPtr>&, ConstFieldPtr, DualFieldPtr, const std::vector<DualFieldPtr>&,
           const std::vector<QuadratureFieldPtr>&) const override
  {
    throw std::logic_error("ComposableWeakForm::vjp is not implemented");
  }

 private:
  static HYPRE_BigInt localStart(const mfem::HypreParMatrix& matrix, bool rows)
  {
    int rank;
    MPI_Comm_rank(matrix.GetComm(), &rank);
    const int index = HYPRE_AssumedPartitionCheck() ? 0 : rank;
    return (rows ? matrix.GetRowStarts() : matrix.GetColStarts())[index];
  }

  /// @brief form the Schur correction for a given auxiliary field using
  /// one forward solve per global input column.
  static HypreMatPtr generalCorrection(const mfem::HypreParMatrix& dCp_dp,
                                       const mfem::HypreParMatrix& dCp_du,
                                       const mfem::HypreParMatrix& dJ_dp,
                                       EquationSolver& solver,
                                       double drop_tolerance)
  {
    // check contracts
    SLIC_ERROR_IF(!std::isfinite(drop_tolerance) or drop_tolerance < 0.0,
                 "General Jacobian correction requires a finite, nonnegative drop tolerance.");

    auto& linear_solver = solver.linearSolver();
    linear_solver.SetOperator(dCp_dp);
    const HYPRE_BigInt first_column = localStart(dCp_du, false);
    mfem::Vector du(dCp_du.Width()), rhs(dCp_dp.Height()), aux_variation(dCp_dp.Width());
    mfem::Vector correction_column(dJ_dp.Height());
    std::vector<std::vector<std::pair<HYPRE_BigInt, mfem::real_t>>> correction_rows(dJ_dp.Height());

    // All ranks visit the same columns, including columns owned by other ranks.
    for (HYPRE_BigInt j = 0; j < dCp_du.GetGlobalNumCols(); ++j) {
      // Get a dense RHS from dCp_du by applying it to the jth input basis vector.
      du = 0.0;
      if (j >= first_column && j < first_column + du.Size()) du[static_cast<int>(j - first_column)] = 1.0;
      dCp_du.Mult(du, rhs);

      // delta q = -(dCp_dp)^-1 dCp_du delta u.
      aux_variation = 0.0;
      linear_solver.Mult(rhs, aux_variation);
      aux_variation *= -1.0;
      dJ_dp.Mult(aux_variation, correction_column);

      // Apply the threshold to the final correction, after multiplication by dJ_dp.
      const auto* values = correction_column.HostRead();
      for (int i = 0; i < correction_column.Size(); ++i) {
        if (std::abs(values[i]) > drop_tolerance) correction_rows[i].emplace_back(j, values[i]);
      }
    }

    // Assemble local rows of the correction using global column indices.
    std::vector<int> correction_row_offsets(1, 0);
    std::vector<HYPRE_BigInt> correction_columns;
    std::vector<mfem::real_t> correction_values;
    for (const auto& row : correction_rows) {
      if (row.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) - correction_columns.size()) {
        throw std::overflow_error("ComposableWeakForm: Jacobian correction exceeds local CSR capacity");
      }
      for (const auto& [column, value] : row) {
        correction_columns.push_back(column);
        correction_values.push_back(value);
      }
      correction_row_offsets.push_back(static_cast<int>(correction_columns.size()));
    }

    // The CSR constructor copies the arrays and the residual/input field partitions.
    return std::make_unique<mfem::HypreParMatrix>(
        dCp_du.GetComm(), dJ_dp.Height(), dJ_dp.GetGlobalNumRows(), dCp_du.GetGlobalNumCols(),
        correction_row_offsets.data(), correction_columns.data(), correction_values.data(),
        dJ_dp.GetRowStarts(), dCp_du.GetColStarts());
  }

  // A contiguous, element-local true-DOF layout allows direct block extraction.
  // Shared or interleaved DOFs use the general path on every rank.
  static bool elementBlockOffsets(const mfem::ParFiniteElementSpace& space, mfem::Array<int>& offsets)
  {
    offsets.SetSize(space.GetNE() + 1);
    offsets[0] = 0;
    int valid = space.GetVSize() == space.GetTrueVSize();
    mfem::Array<int> dofs;
    for (int e = 0; e < space.GetNE() && valid; ++e) {
      space.GetElementVDofs(e, dofs);
      for (int& dof : dofs) {
        dof = space.GetLocalTDofNumber(dof < 0 ? -1 - dof : dof);
      }
      dofs.Sort();
      for (int j = 0; j < dofs.Size(); ++j) valid &= dofs[j] == offsets[e] + j;
      offsets[e + 1] = offsets[e] + dofs.Size();
    }
    valid = valid && offsets.Last() == space.GetTrueVSize();
    int all_valid;
    MPI_Allreduce(&valid, &all_valid, 1, MPI_INT, MPI_MIN, space.GetComm());
    return all_valid != 0;
  }

  /// @brief form the Schur correction for a given auxiliary field when 
  /// the jacobian of the auxiliary constraint with respect to the auxiliary variable
  /// (∂_p C_p) has independent diagonal blocks.
  static HypreMatPtr blockCorrection(const mfem::ParFiniteElementSpace& aux_pfes,
                                     composable::BlockInverses& dCp_dp_factors,
                                     const mfem::HypreParMatrix& dCp_du,
                                     const mfem::HypreParMatrix& dJ_dp)
  {

    // check contracts
    SLIC_ERROR_IF(aux_pfes.GetParMesh()->Nonconforming(),
                "Element independent auxiliary field requires a conforming mesh.");
    SLIC_ERROR_IF((aux_pfes.GetVDim() != 1 and aux_pfes.GetOrdering() != mfem::Ordering::byVDIM)
                  or dynamic_cast<const mfem::L2_FECollection *>(aux_pfes.FEColl()) == nullptr,
                "Element independent auxiliary field requires contiguous dofs.");

    mfem::SparseMatrix diag, offd;
    HYPRE_BigInt* offd_column_map;
    dCp_du.GetDiag(diag);
    dCp_du.GetOffd(offd, offd_column_map);
    const auto* diag_rows = diag.HostReadI();
    const auto* diag_columns = diag.HostReadJ();
    const auto* diag_values = diag.HostReadData();
    const auto* offd_rows = offd.HostReadI();
    const auto* offd_columns = offd.HostReadJ();
    const auto* offd_values = offd.HostReadData();
    const HYPRE_BigInt first_column = localStart(dCp_du, false);
    const auto& offsets = dCp_dp_factors.block_offsets;

    // Assemble local rows of the auxiliary variation using global column indices.
    std::vector<int> aux_row_offsets(1, 0);
    std::vector<HYPRE_BigInt> aux_columns;
    std::vector<mfem::real_t> aux_values;

    for(int b = 0; b < offsets.Size() - 1; ++b) {
      // block indices 
      // Factorization offsets already index local true DOFs; the end is exclusive.
      const int istart_ltdof = offsets[b];
      const int iend_ltdof = offsets[b + 1];

      auto [D_factor, n] = dCp_dp_factors.getFactorization(b);
      SLIC_ERROR_IF(n != iend_ltdof - istart_ltdof,
          "Factorization size does not match the element tdof blocks");


      // J_k contains the global columns with stored entries in this block's rows, including explicit zeros.
      std::vector<HYPRE_BigInt> J_k;
      // Local columns come first, then remote columns; each group is sorted.
      const int diag_start = diag_rows[istart_ltdof];
      const int offd_start = offd_rows[istart_ltdof];

      // Map diag and offd entries to RHS columns independently.
      auto make_rhs_mapping = [&J_k](int start, int end, auto global_column) {
        std::vector<std::pair<HYPRE_BigInt, int>> entries;
        entries.reserve(end - start);
        for (int j = start; j < end; ++j) {
          entries.emplace_back(global_column(j), j - start);
        }
        std::sort(entries.begin(), entries.end());

        std::vector<int> entry_to_rhs(end - start);
        for (const auto& [column, entry] : entries) {
          if (J_k.empty() || J_k.back() != column) J_k.push_back(column);
          entry_to_rhs[entry] = static_cast<int>(J_k.size() - 1);
        }
        return entry_to_rhs;
      };

      auto diag_to_rhs = make_rhs_mapping(diag_start, diag_rows[iend_ltdof],
                                         [&](int j) { return first_column + diag_columns[j]; });
      // Remote columns are disjoint from the local columns already in J_k.
      auto offd_to_rhs = make_rhs_mapping(offd_start, offd_rows[iend_ltdof],
                                         [&](int j) { return offd_column_map[offd_columns[j]]; });

      // get a dense RHS from dCp_du
      // Rows follow the block's local true DOFs; columns follow J_k.
      mfem::DenseMatrix rhs(n, static_cast<int>(J_k.size()));
      rhs = 0.0;
      for (int i = istart_ltdof; i < iend_ltdof; ++i) {
        for (int j = diag_rows[i]; j < diag_rows[i + 1]; ++j) {
          const int rhs_column = diag_to_rhs[j - diag_start];
          rhs(i - istart_ltdof, rhs_column) += diag_values[j];
        }
        for (int j = offd_rows[i]; j < offd_rows[i + 1]; ++j) {
          const int rhs_column = offd_to_rhs[j - offd_start];
          rhs(i - istart_ltdof, rhs_column) += offd_values[j];
        }
      }

      // solve (dCp_dp)^-1 dCp_du using the factorization
      mfem::DenseMatrix DinvR(rhs);
      if (DinvR.Width() > 0) {
        std::visit([&](const auto& factorization) {
          factorization.Solve(n, DinvR.Width(), DinvR.HostReadWrite());
        }, D_factor);
      }

      // delta q = -(dCp_dp)^-1 dCp_du delta u. Retain every entry in I_k x J_k,
      // including zeros, since the block solve can fill the entire rectangle.
      for (int i = 0; i < n; ++i) {
        if (J_k.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) - aux_columns.size()) {
          throw std::overflow_error("ComposableWeakForm: auxiliary variation exceeds local CSR capacity");
        }
        aux_columns.insert(aux_columns.end(), J_k.begin(), J_k.end());
        for (int j = 0; j < DinvR.Width(); ++j) {
          aux_values.push_back(-DinvR(i, j));
        }
        aux_row_offsets.push_back(static_cast<int>(aux_columns.size()));
      }
    }

    // The CSR constructor copies the arrays and the input field partitions.
    auto aux_variation_mat = std::make_unique<mfem::HypreParMatrix>(
        dCp_du.GetComm(), dCp_du.Height(), dCp_du.GetGlobalNumRows(), dCp_du.GetGlobalNumCols(),
        aux_row_offsets.data(), aux_columns.data(), aux_values.data(),
        dCp_du.GetRowStarts(), dCp_du.GetColStarts());

    // Own the product's partitions so it outlives aux_variation_mat.
    return HypreMatPtr(mfem::ParMult(&dJ_dp, aux_variation_mat.get(), true));
  }

  static std::vector<ConstFieldPtr> append_aux(const std::vector<ConstFieldPtr>& fields, ConstFieldPtr auxiliary_field)
  {
    auto inputs = fields;
    inputs.push_back(auxiliary_field);
    return inputs;
  }

  static std::unique_ptr<mfem::HypreParMatrix> auxiliaryJacobian(
      const WeakForm& weak_form, const TimeInfo& time_info, ConstFieldPtr shape_disp,
      const std::vector<ConstFieldPtr>& inputs, const std::vector<ConstQuadratureFieldPtr>& quad_fields)
  {
    std::vector<double> tangents(inputs.size(), 0.0);
    tangents.back() = 1.0;
    return weak_form.jacobian(time_info, shape_disp, inputs, tangents, quad_fields);
  }

  std::vector<ConstFieldPtr> append_all_aux(const std::vector<ConstFieldPtr>& primary_fields) const
  {
    std::vector<ConstFieldPtr> inputs(primary_fields);
    inputs.reserve(primary_fields.size() + NumAuxiliaryFields);
    for (const auto& [auxiliary_field, weak_form, solver, properties] : auxiliary_solves_) {
      inputs.push_back(auxiliary_field.get());
    }
    return inputs;
  }

  /// @brief solve for the auxiliary fields given the primary fields
  /// NOTE: updates the data in stored auxiliary fields
  ///
  /// @param time_info the timestep information for time dependent problems
  /// @param shape_disp the displacement of the mesh nodes
  /// @param fields the primary fields
  /// @param quad_fields additional fields defined at the quadrature points
  void solveAuxiliaryFields(const TimeInfo& time_info, ConstFieldPtr shape_disp,
                            const std::vector<ConstFieldPtr>& fields,
                            const std::vector<ConstQuadratureFieldPtr>& quad_fields) const
  {
    SLIC_ERROR_IF(!primary_weak_form_, "ComposableWeakForm: primary weak form empty/null.");

    for (const auto& [auxiliary_field, weak_form, solver, properties] : auxiliary_solves_) {
      SLIC_ERROR_IF(!weak_form || !solver || !auxiliary_field,
                    "ComposableWeakForm: weak form, solver, or field missing from an auxiliary solve");

      // solvers require a invocable that takes the auxiliary field
      // to compute residuals/jacobians
      // This takes the auxiliary field from the invocation and appends it to the primary fields
      // so that the auxiliary weak form has the appropriate inputs
      auto make_inputs = [&](const mfem::Vector& aux_data) {
        // copy the data into the field then append it to the primary inputs
        *auxiliary_field = aux_data;
        std::vector<ConstFieldPtr> inputs(fields);
        inputs.push_back(auxiliary_field.get());
        return inputs;
      };

      std::unique_ptr<mfem::HypreParMatrix> J;
      smith::mfem_ext::StdFunctionOperator op(
          auxiliary_field->Size(),
          // residual
          [&](const mfem::Vector& aux_data, mfem::Vector& r) {
            r = weak_form->residual(time_info, shape_disp, make_inputs(aux_data), quad_fields);
          },
          // jacobian
          [&](const mfem::Vector& aux_data) -> mfem::Operator& {
            auto inputs = make_inputs(aux_data);
            // select only the aux variables for getting derivatives wrt
            std::vector<double> tangent_weights(inputs.size(), 0.0);
            tangent_weights.back() = 1.0;
            J = weak_form->jacobian(time_info, shape_disp, inputs, tangent_weights, quad_fields);
            if (!J) throw std::logic_error("ComposableWeakForm: missing auxiliary self Jacobian");
            return *J;
          });

      // Keep the solver iterate separate: evaluating a trial point updates the
      // field used by the weak form, but must not overwrite Newton's iterate.
      mfem::Vector solution(*auxiliary_field);
      solver->setOperator(op);
      solver->solve(solution);
      *auxiliary_field = solution;
    }
  }
};

}  // namespace smith
