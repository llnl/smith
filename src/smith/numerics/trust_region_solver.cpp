// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "smith/numerics/trust_region_solver.hpp"

#ifdef SMITH_USE_SLEPC

#include <iostream>

#include "smith/infrastructure/profiling.hpp"
#include "smith/numerics/dense_petsc.hpp"

namespace smith {

/**
 * @brief Get the global size of a mfem vector
 * @param parallel_v Vector to check global size
 * @param comm Parallel communicator
 */
int globalSize(const mfem::Vector& parallel_v, const MPI_Comm& comm)
{
  int local_size = parallel_v.Size();
  int global_size;
  MPI_Allreduce(&local_size, &global_size, 1, MPI_INT, MPI_SUM, comm);
  return global_size;
}

/// @brief struct which aids in moving between mfem::Vector and petsc BV
struct BasisVectors {
  /**
   * @brief Construct with a representative state to set sizes
   * @param state The state which is used to set sizes for basis vectors
   */
  BasisVectors(const mfem::Vector& state) : local_rows(state.Size()), global_rows(globalSize(state, PETSC_COMM_WORLD))
  {
    VecCreateMPI(PETSC_COMM_WORLD, local_rows, global_rows, &v);

    PetscInt iStart, iEnd;
    VecGetOwnershipRange(v, &iStart, &iEnd);

    col_indices.reserve(size_t(local_rows));
    for (int i = iStart; i < iEnd; ++i) {
      col_indices.push_back(i);
    }
  }

  /**
   * @brief Destructor
   */
  ~BasisVectors() { VecDestroy(&v); }

  /**
   * @brief Construct petsc BV from vector of mfem::Vector
   * @param states The states used to construct basis vectors
   */
  BV constructBases(const std::vector<const mfem::Vector*>& states) const
  {
    size_t num_cols = states.size();
    BV Q;
    BVCreate(PETSC_COMM_SELF, &Q);
    BVSetType(Q, BVVECS);
    BVSetSizesFromVec(Q, v, static_cast<int>(num_cols));
    for (size_t c = 0; c < num_cols; ++c) {
      VecSetValues(v, local_rows, &col_indices[0], &(*states[c])[0], INSERT_VALUES);
      VecAssemblyBegin(v);
      VecAssemblyEnd(v);
      int c_int = static_cast<int>(c);
      BVInsertVec(Q, c_int, v);
    }
    return Q;
  }

 private:
  const int local_rows;
  const int global_rows;

  std::vector<int> col_indices;
  Vec v;
};

/**
 * @brief Create a petsc vector from a mfem::Vector
 * @param state The state used to create an mfem::Vector
 */
Vec petscVec(const mfem::Vector& state)
{
  const int local_rows = state.Size();
  const int global_rows = globalSize(state, PETSC_COMM_WORLD);

  Vec v;
  VecCreateMPI(PETSC_COMM_WORLD, local_rows, global_rows, &v);

  PetscInt iStart, iEnd;
  VecGetOwnershipRange(v, &iStart, &iEnd);

  std::vector<int> col_indices;
  col_indices.reserve(static_cast<size_t>(local_rows));
  for (int i = iStart; i < iEnd; ++i) {
    col_indices.push_back(i);
  }

  VecSetValues(v, local_rows, &col_indices[0], &state[0], INSERT_VALUES);

  VecAssemblyBegin(v);
  VecAssemblyEnd(v);

  return v;
}

/**
 * @brief Copy a petsc vector to an mfem::Vector
 * @param v The petsc vector
 * @param s The mfem vector
 */
void copy(const Vec& v, mfem::Vector& s)
{
  const int local_rows = s.Size();
  PetscInt iStart, iEnd;
  VecGetOwnershipRange(v, &iStart, &iEnd);

  SLIC_ERROR_IF(local_rows != iEnd - iStart,
                "Inconsistency between local t-dof vector size and petsc start and end indices");

  std::vector<int> col_indices;
  col_indices.reserve(static_cast<size_t>(local_rows));
  for (int i = iStart; i < iEnd; ++i) {
    col_indices.push_back(i);
  }

  VecGetValues(v, local_rows, &col_indices[0], &s[0]);
}

/**
 * @brief The reduced matrix in the space of {s}
 * @param s The vector of mfem::Vector of directions
 * @param As The vector of mfem::Vector of a global matrix A operated on directions
 */
Mat dot(const std::vector<const mfem::Vector*>& s, const std::vector<const mfem::Vector*>& As)
{
  SLIC_ERROR_IF(s.size() != As.size(),
                "Search directions and their linear operator result must have same number of columns");
  size_t num_cols = s.size();
  int num_cols_int = static_cast<int>(num_cols);
  Mat sAs;
  MatCreateSeqDense(PETSC_COMM_SELF, num_cols_int, num_cols_int, NULL, &sAs);
  for (size_t i = 0; i < num_cols; ++i) {
    for (size_t j = 0; j < num_cols; ++j) {
      MatSetValue(sAs, static_cast<int>(i), static_cast<int>(j), mfem::InnerProduct(PETSC_COMM_WORLD, *s[i], *As[j]),
                  INSERT_VALUES);
    }
  }
  MatAssemblyBegin(sAs, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(sAs, MAT_FINAL_ASSEMBLY);
  return sAs;
}

/**
 * @brief The reduced vector s.T*b
 * @param s The vector of mfem::vector of directions
 * @param b The right hand size vector to be reduced
 */
Vec dot(const std::vector<const mfem::Vector*>& s, const mfem::Vector& b)
{
  size_t num_cols = s.size();
  Vec sb;
  VecCreateSeq(PETSC_COMM_SELF, static_cast<int>(num_cols), &sb);
  for (size_t i = 0; i < num_cols; ++i) {
    VecSetValue(sb, static_cast<int>(i), mfem::InnerProduct(PETSC_COMM_WORLD, *s[i], b), INSERT_VALUES);
  }
  return sb;
}

/**
 * @brief The qr decomposition of the state vectors
 * @param states The vector of mfem::vectors of directions
 * @return Pair of BV Q and DenseMat R
 */
auto qr(const std::vector<const mfem::Vector*>& states)
{
  BasisVectors bvs(*states[0]);
  BV Q = bvs.constructBases(states);

  Mat R;
  int num_cols = static_cast<int>(states.size());
  MatCreateSeqDense(PETSC_COMM_SELF, num_cols, num_cols, NULL, &R);
  auto error = BVOrthogonalize(Q, R);

  if (error) throw PetscException("BVOrthogonalize failed.");

  return std::make_pair(Q, DenseMat(R));
}

/**
 * @brief compute the quadratic energy from small dense matrices and vectors
 * @param A The stiffness matrix
 * @param b The rhs vector
 * @param x The current solution vector
 * @return The quadratic, linearized energy approximation
 */
double quadraticEnergy(const DenseMat& A, const DenseVec& b, const DenseVec& x)
{
  DenseVec Ax = A * x;
  double xAx = dot(x, Ax);
  double xb = dot(x, b);
  return 0.5 * xAx - xb;
}

/**
 * @brief compute the pnorm_squared
 * @param bvv input vector
 * @param sig eigenvectors
 */
double pnorm_squared(const DenseVec& bvv, const DenseVec& sig)
{
  auto bvv_div_sig_squared = bvv / (sig * sig);
  return sum(bvv_div_sig_squared);
}

/**
 * @brief compute the qnorm_squared
 * @param bvv input vector
 * @param sig eigenvectors
 */
double qnorm_squared(const DenseVec& bvv, const DenseVec& sig)
{
  auto bvv_div_sig_cubed = bvv / (sig * sig * sig);
  return sum(bvv_div_sig_cubed);
  // return bvv.dot((1.0 / (sig * sig * sig)).matrix());
}

//  returns:
//    minimum energy solution within delta
//    N leftmost eigenvectors
//    N smallest eigenvalue
//    success status

/**
 * @brief solve the trust region problem exactly using a variant of the Moore Sorensen algorithm
 * @param A matrix
 * @param b rhs
 * @param delta trust region radius
 * @param num_leftmost the number of leftmost eigenvector/values to output
 * returns the solution vector, a std::vector of leftmost vectors
 * a std::vector of leftmost eigenvalues and the energy change (relative to x=0)
 */
auto exactTrustRegionSolve(DenseMat A, const DenseVec& b, double delta, int num_leftmost)
{
  // minimize 1/2 x^T A x - b^T x, s.t. norm(x) <= delta
  auto [isize, jsize] = A.size();
  auto isize2 = b.size();
  SLIC_ERROR_IF(isize != jsize, "Exact trust region solver requires square matrices");
  SLIC_ERROR_IF(isize != isize2,
                "The right hand size for exact trust region solve must be consistent with the input matrix size");

  auto [sigs, V] = eigh(A);
  std::vector<DenseVec> leftmosts;
  std::vector<double> minsigs;
  size_t num_leftmost_possible(size_t(std::min(num_leftmost, isize)));
  for (size_t i = 0; i < num_leftmost_possible; ++i) {
    leftmosts.emplace_back(V[i]);
    minsigs.emplace_back(sigs[i]);
  }

  const auto& leftMost = V[0];
  double minSig = sigs[0];

  // bv = V.T b, V has columns which are eigenvectors
  DenseVec bv(isize);
  for (size_t i = 0; i < size_t(isize); ++i) {
    bv.setValue(i, dot(V[i], b));
  }

  DenseVec bvOverSigs = bv / sigs;
  double sigScale = sum(abs(sigs)) / isize;
  double eps = 1e-12 * sigScale;

  // Check if solution is inside the trust region
  if ((minSig >= eps) && (norm(bvOverSigs) <= delta)) {
    return std::make_tuple(A.solve(b), leftmosts, minsigs, true);
  }

  // if we get here, the solution must be on the tr boundary
  // consider bounding the initial guess, see More' Sorenson paper
  double lam = minSig < eps ? -minSig + eps : 0.0;

  // try to solve this for lam:
  // (A + lam I)p = b, such that norm(p) = Delta
  DenseVec sigsPlusLam = sigs + lam;

  bvOverSigs = bv / sigsPlusLam;

  // Check for the hard case
  if ((minSig < eps) && (norm(bvOverSigs) < delta)) {
    DenseVec p(isize);
    p = 0.0;
    for (int i = 0; i < isize; ++i) {
      p.add(bv[i], V[size_t(i)]);
    }

    const auto& z = leftMost;
    double pz = dot(p, z);
    double pp = dot(p, p);
    double ddmpp = std::max(delta * delta - pp, 0.0);

    double tau1 = -pz + std::sqrt(pz * pz + ddmpp);
    double tau2 = -pz - std::sqrt(pz * pz + ddmpp);

    DenseVec x1(p);
    DenseVec x2(p);
    x1.add(tau1, z);
    x2.add(tau2, z);

    double e1 = quadraticEnergy(A, b, x1);
    double e2 = quadraticEnergy(A, b, x2);

    DenseVec x = e1 < e2 ? x1 : x2;

    return std::make_tuple(x, leftmosts, minsigs, true);
  }
  DenseVec bvbv = bv * bv;
  sigsPlusLam = sigs + lam;

  double pNormSq = pnorm_squared(bvbv, sigsPlusLam);
  double pNorm = std::sqrt(pNormSq);
  double bError = (pNorm - delta) / delta;

  // consider an out if it doesn't converge, or use a better initial guess, or bound the lam from below and above.
  size_t iters = 0;
  size_t maxIters = 30;
  while ((std::abs(bError) > 1e-9) && (iters++ < maxIters)) {
    double qNormSq = qnorm_squared(bvbv, sigsPlusLam);
    lam += (pNormSq / qNormSq) * bError;
    sigsPlusLam = sigs + lam;
    pNormSq = pnorm_squared(bvbv, sigsPlusLam);
    pNorm = std::sqrt(pNormSq);
    bError = (pNorm - delta) / delta;
  }

  bool success = true;
  if (iters >= maxIters) {
    success = false;
  }

  bvOverSigs = bv / sigsPlusLam;

  DenseVec x(isize);
  x = 0.0;
  for (int i = 0; i < isize; ++i) {
    x.add(bvOverSigs[i], V[size_t(i)]);
  }

  double e1 = quadraticEnergy(A, b, x);
  double e2 = quadraticEnergy(A, b, -x);

  if (e2 < e1) {
    x *= -delta / norm(x);
  } else {
    x *= delta / norm(x);
  }

  return std::make_tuple(x, leftmosts, minsigs, success);
}

/// @brief remove the vector at location j and return what is left
std::vector<const mfem::Vector*> remove_at(const std::vector<const mfem::Vector*>& a, size_t j)
{
  std::vector<const mfem::Vector*> b;
  for (size_t i = 0; i < a.size(); ++i) {
    if (i != j) {
      b.emplace_back(a[i]);
    }
  }
  return b;
}

/// @brief returns the solution, as well as a list of the N leftmost eigenvectors
/// and their eigenvalues, and the predicted model energy change
std::tuple<mfem::Vector, std::vector<std::shared_ptr<mfem::Vector>>, std::vector<double>, double> solveSubspaceProblem(
    const std::vector<const mfem::Vector*>& states, const std::vector<const mfem::Vector*>& Astates,
    const mfem::Vector& b, double delta, int num_leftmost)
{
  SMITH_MARK_FUNCTION;
  DenseMat sAs1 = dot(states, Astates);
  DenseMat sAs = sym(sAs1);

  if (sAs.hasNan()) {
    throw PetscException("States in subspace solve contain NaNs.");
    return std::make_tuple(b, std::vector<std::shared_ptr<mfem::Vector>>{}, std::vector<double>{}, 0);
  }

  auto [Q_parallel, R] = qr(states);

  if (R.hasNan()) {
    throw PetscException("R from qr returning with a NaN.");
    return std::make_tuple(b, std::vector<std::shared_ptr<mfem::Vector>>{}, std::vector<double>{}, 0);
  }

  auto [rows, cols] = R.size();
  SLIC_ERROR_IF(rows != cols, "R matrix is not square in subspace problem solve\n");

  double trace_mag = 0.0;
  for (int i = 0; i < rows; ++i) {
    trace_mag += std::abs(R(i, i));
  }

  // remove any nearly colinear state
  for (int i = 0; i < rows; ++i) {
    if (R(i, i) < 1e-9 * trace_mag) {
      // printf("removing after QR state number %d\n", i);
      auto statesNew = remove_at(states, size_t(i));
      auto AstatesNew = remove_at(Astates, size_t(i));
      return solveSubspaceProblem(statesNew, AstatesNew, b, delta, num_leftmost);
    }
  }

  auto Rinv = inverse(R);
  DenseMat pAp = sAs.PtAP(Rinv);

  Vec b_parallel = petscVec(b);
  std::vector<double> pb_vec(states.size());
  BVDotVec(Q_parallel, b_parallel, &pb_vec[0]);
  DenseVec pb(pb_vec);

  auto [reduced_x, leftvecs, leftvals, success] = exactTrustRegionSolve(pAp, pb, delta, num_leftmost);

  double energy = quadraticEnergy(pAp, pb, reduced_x);

  Vec x_parallel;
  VecDuplicate(b_parallel, &x_parallel);

  std::vector<double> reduced_x_vec = reduced_x.getValues();
  BVMultVec(Q_parallel, 1.0, 0.0, x_parallel, &reduced_x_vec[0]);
  mfem::Vector sol(b);
  copy(x_parallel, sol);

  std::vector<std::shared_ptr<mfem::Vector>> leftmosts;
  for (size_t i = 0; i < leftvecs.size(); ++i) {
    auto reduced_leftvec = leftvecs[i].getValues();
    BVMultVec(Q_parallel, 1.0, 0.0, x_parallel, &reduced_leftvec[0]);
    leftmosts.emplace_back(std::make_shared<mfem::Vector>(b));
    copy(x_parallel, *leftmosts[i]);
  }

  BVDestroy(&Q_parallel);
  VecDestroy(&b_parallel);
  VecDestroy(&x_parallel);
  return std::make_tuple(sol, leftmosts, leftvals, energy);
}

/// @brief Remove any obvious dependent directions, namely ones which are scaled version of previous directions
/// The case where they are linear combinations of previous direction will be handled in the QR solver
std::pair<std::vector<const mfem::Vector*>, std::vector<const mfem::Vector*>> removeDependentDirections(
    std::vector<const mfem::Vector*> directions, std::vector<const mfem::Vector*> A_directions)
{
  SMITH_MARK_FUNCTION;
  std::vector<double> norms;
  size_t num_dirs = directions.size();

  for (size_t i = 0; i < num_dirs; ++i) {
    norms.push_back(std::sqrt(mfem::InnerProduct(PETSC_COMM_WORLD, *directions[i], *directions[i])));
  }

  std::vector<std::pair<const mfem::Vector*, size_t>> kepts;
  for (size_t i = 0; i < num_dirs; ++i) {
    bool keepi = true;
    if (norms[i] == 0) keepi = false;
    for (auto&& kept_and_j : kepts) {
      size_t j = kept_and_j.second;
      double dot_ij = mfem::InnerProduct(PETSC_COMM_WORLD, *directions[i], *kept_and_j.first);
      if (dot_ij > 0.999 * norms[i] * norms[j]) {
        keepi = false;
      }
    }
    // if (!keepi) printf("not keeping %zu\n",i);
    if (keepi) {
      kepts.emplace_back(std::make_pair(directions[i], i));
    }
  }

  std::vector<const mfem::Vector*> directions_new;
  std::vector<const mfem::Vector*> A_directions_new;

  for (auto kept_and_j : kepts) {
    directions_new.push_back(directions[kept_and_j.second]);
    A_directions_new.push_back(A_directions[kept_and_j.second]);
  }

  return std::make_pair(directions_new, A_directions_new);
}

}  // namespace smith

#endif  // SMITH_USE_SLEPC
