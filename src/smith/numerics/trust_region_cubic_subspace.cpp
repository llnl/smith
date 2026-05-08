// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#include "smith/numerics/trust_region_solver.hpp"

#include <cmath>

#include "smith/infrastructure/profiling.hpp"

namespace smith {

#ifdef MFEM_USE_LAPACK

namespace {

double dot(const mfem::Vector& a, const mfem::Vector& b)
{
  return a * b;
}

void symmetrize(mfem::DenseMatrix& A)
{
  MFEM_VERIFY(A.Height() == A.Width(), "symmetrize requires square matrix.");
  for (int i = 0; i < A.Height(); ++i) {
    for (int j = 0; j < i; ++j) {
      const double value = 0.5 * (A(i, j) + A(j, i));
      A(i, j) = value;
      A(j, i) = value;
    }
  }
}

mfem::Vector matrixColumn(const mfem::DenseMatrix& A, int j)
{
  mfem::Vector col(A.Height());
  for (int i = 0; i < A.Height(); ++i) {
    col[i] = A(i, j);
  }
  return col;
}

mfem::DenseMatrix columnsToMatrix(const std::vector<mfem::Vector>& cols)
{
  mfem::DenseMatrix A(cols.empty() ? 0 : cols[0].Size(), static_cast<int>(cols.size()));
  for (int j = 0; j < A.Width(); ++j) {
    for (int i = 0; i < A.Height(); ++i) {
      A(i, j) = cols[size_t(j)][i];
    }
  }
  return A;
}

mfem::DenseMatrix denseDot(const std::vector<const mfem::Vector*>& s, const std::vector<const mfem::Vector*>& As)
{
  MFEM_VERIFY(s.size() == As.size(), "Dense dot requires matching direction counts.");
  mfem::DenseMatrix result(static_cast<int>(s.size()));
  for (int i = 0; i < result.Height(); ++i) {
    for (int j = 0; j < result.Width(); ++j) {
      result(i, j) = innerProduct(*s[size_t(i)], *As[size_t(j)], MPI_COMM_WORLD);
    }
  }
  return result;
}

mfem::Vector denseDot(const std::vector<const mfem::Vector*>& s, const mfem::Vector& b)
{
  mfem::Vector result(static_cast<int>(s.size()));
  for (int i = 0; i < result.Size(); ++i) {
    result[i] = innerProduct(*s[size_t(i)], b, MPI_COMM_WORLD);
  }
  return result;
}

mfem::DenseMatrix orthonormalBasisTransform(const mfem::DenseMatrix& gram)
{
  mfem::DenseMatrix gram_copy(gram);
  mfem::Vector evals;
  mfem::DenseMatrix evecs;
  gram_copy.Eigensystem(evals, evecs);

  double trace_mag = 0.0;
  for (int i = 0; i < evals.Size(); ++i) {
    trace_mag += std::abs(evals[i]);
  }

  std::vector<mfem::Vector> kept_columns;
  for (int i = 0; i < evals.Size(); ++i) {
    if (evals[i] > 1e-9 * trace_mag) {
      mfem::Vector col = matrixColumn(evecs, i);
      col /= std::sqrt(evals[i]);
      kept_columns.emplace_back(std::move(col));
    }
  }

  return columnsToMatrix(kept_columns);
}

mfem::DenseMatrix tripleProduct(const mfem::DenseMatrix& L, const mfem::DenseMatrix& A, const mfem::DenseMatrix& R)
{
  mfem::DenseMatrix tmp(A.Height(), R.Width());
  mfem::Mult(A, R, tmp);
  mfem::DenseMatrix out(L.Width(), R.Width());
  mfem::MultAtB(L, tmp, out);
  return out;
}

mfem::Vector projectWithTranspose(const mfem::DenseMatrix& A, const mfem::Vector& x)
{
  mfem::Vector out(A.Width());
  A.MultTranspose(x, out);
  return out;
}

mfem::DenseMatrix orthonormalBasisWithFirstVector(const mfem::Vector& first)
{
  const int n = first.Size();
  mfem::DenseMatrix Q(n);
  Q = 0.0;

  mfem::Vector q0(first);
  q0 /= q0.Norml2();
  for (int i = 0; i < n; ++i) {
    Q(i, 0) = q0[i];
  }

  int col = 1;
  for (int seed = 0; seed < n && col < n; ++seed) {
    mfem::Vector candidate(n);
    candidate = 0.0;
    candidate[seed] = 1.0;
    for (int j = 0; j < col; ++j) {
      const mfem::Vector qj = matrixColumn(Q, j);
      candidate.Add(-dot(candidate, qj), qj);
    }
    const double norm = candidate.Norml2();
    if (norm > 1.0e-12) {
      candidate /= norm;
      for (int i = 0; i < n; ++i) {
        Q(i, col) = candidate[i];
      }
      ++col;
    }
  }

  MFEM_VERIFY(col == n, "Failed to build orthonormal basis for cubic tensor completion.");
  return Q;
}

std::vector<mfem::DenseMatrix> completeSymmetricCubicTensor(const mfem::DenseMatrix& deltaA,
                                                            const mfem::Vector& previous_step)
{
  const int n = previous_step.Size();
  const double step_norm = previous_step.Norml2();
  MFEM_VERIFY(step_norm > 0.0, "Cannot complete cubic tensor with zero previous step.");

  const mfem::DenseMatrix Q = orthonormalBasisWithFirstVector(previous_step);
  mfem::DenseMatrix delta_hat = tripleProduct(Q, deltaA, Q);
  symmetrize(delta_hat);

  std::vector<mfem::DenseMatrix> tensor_hat(static_cast<size_t>(n), mfem::DenseMatrix(n));
  for (auto& matrix : tensor_hat) {
    matrix = 0.0;
  }

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      const double value = delta_hat(i, j) / step_norm;
      tensor_hat[0](i, j) = value;
      tensor_hat[size_t(i)](0, j) = value;
      tensor_hat[size_t(i)](j, 0) = value;
    }
  }

  std::vector<mfem::DenseMatrix> tensor(static_cast<size_t>(n), mfem::DenseMatrix(n));
  for (auto& matrix : tensor) {
    matrix = 0.0;
  }

  for (int a = 0; a < n; ++a) {
    for (int b = 0; b < n; ++b) {
      for (int c = 0; c < n; ++c) {
        double value = 0.0;
        for (int alpha = 0; alpha < n; ++alpha) {
          for (int beta = 0; beta < n; ++beta) {
            for (int gamma = 0; gamma < n; ++gamma) {
              value += Q(a, alpha) * Q(b, beta) * Q(c, gamma) * tensor_hat[size_t(alpha)](beta, gamma);
            }
          }
        }
        tensor[size_t(a)](b, c) = value;
      }
    }
  }

  return tensor;
}

mfem::Vector combineDirections(const std::vector<const mfem::Vector*>& states, const mfem::Vector& coeffs)
{
  mfem::Vector out(*states[0]);
  out = 0.0;
  for (int i = 0; i < coeffs.Size(); ++i) {
    out.Add(coeffs[i], *states[size_t(i)]);
  }
  return out;
}

void verifyCubicInputs(const mfem::DenseMatrix& A, const mfem::Vector& b, const std::vector<mfem::DenseMatrix>& cubic,
                       double delta)
{
  MFEM_VERIFY(A.Height() == A.Width(), "Dense cubic trust-region matrix must be square.");
  MFEM_VERIFY(A.Height() == b.Size(), "Dense cubic trust-region linear term has incompatible size.");
  MFEM_VERIFY(delta >= 0.0, "Dense cubic trust-region radius must be nonnegative.");
  MFEM_VERIFY(static_cast<int>(cubic.size()) == b.Size(), "Dense cubic tensor must have one matrix per dimension.");
  for (const auto& matrix : cubic) {
    MFEM_VERIFY(matrix.Height() == b.Size() && matrix.Width() == b.Size(),
                "Dense cubic tensor matrix has incompatible size.");
  }
}

double cubicEnergy(const mfem::DenseMatrix& A, const mfem::Vector& b, const std::vector<mfem::DenseMatrix>& cubic,
                   const mfem::Vector& x)
{
  mfem::Vector Ax(x.Size());
  A.Mult(x, Ax);
  double energy = 0.5 * dot(x, Ax) - dot(x, b);
  for (int k = 0; k < x.Size(); ++k) {
    cubic[size_t(k)].Mult(x, Ax);
    energy += (x[k] * dot(x, Ax)) / 6.0;
  }
  return energy;
}

mfem::Vector cubicGradient(const mfem::DenseMatrix& A, const mfem::Vector& b,
                           const std::vector<mfem::DenseMatrix>& cubic, const mfem::Vector& x)
{
  mfem::Vector grad(x.Size());
  A.Mult(x, grad);
  grad -= b;

  mfem::Vector tmp(x.Size());
  for (int i = 0; i < x.Size(); ++i) {
    double correction = 0.0;
    cubic[size_t(i)].Mult(x, tmp);
    correction += dot(x, tmp);
    for (int k = 0; k < x.Size(); ++k) {
      for (int j = 0; j < x.Size(); ++j) {
        correction += x[k] * (cubic[size_t(k)](i, j) + cubic[size_t(k)](j, i)) * x[j];
      }
    }
    grad[i] += correction / 6.0;
  }

  return grad;
}

void projectToBall(mfem::Vector& x, double delta)
{
  const double norm = x.Norml2();
  if (norm > delta && norm > 0.0) {
    x *= delta / norm;
  }
}

mfem::Vector solveQuadraticCandidate(mfem::DenseMatrix A, const mfem::Vector& b, double delta)
{
  const int n = b.Size();
  mfem::DenseMatrix shifted(A);
  double trace = 0.0;
  for (int i = 0; i < n; ++i) {
    trace += std::abs(A(i, i));
  }
  const double regularization = std::max(1.0e-14, 1.0e-12 * trace / std::max(n, 1));
  for (int i = 0; i < n; ++i) {
    shifted(i, i) += regularization;
  }

  mfem::DenseMatrixInverse inv(shifted);
  mfem::Vector x(n);
  inv.Mult(b, x);
  projectToBall(x, delta);
  return x;
}

mfem::Vector projectedGradientSolve(const mfem::DenseMatrix& A, const mfem::Vector& b,
                                    const std::vector<mfem::DenseMatrix>& cubic, mfem::Vector x, double delta)
{
  double energy = cubicEnergy(A, b, cubic, x);
  constexpr int max_iters = 200;
  constexpr double grad_tol = 1.0e-11;

  for (int iter = 0; iter < max_iters; ++iter) {
    mfem::Vector grad = cubicGradient(A, b, cubic, x);
    if (grad.Norml2() <= grad_tol * std::max(1.0, b.Norml2())) {
      break;
    }

    double step = 0.25;
    bool accepted = false;
    for (int ls = 0; ls < 30; ++ls) {
      mfem::Vector trial(x);
      trial.Add(-step, grad);
      projectToBall(trial, delta);
      const double trial_energy = cubicEnergy(A, b, cubic, trial);
      if (trial_energy < energy - 1.0e-14) {
        x = trial;
        energy = trial_energy;
        accepted = true;
        break;
      }
      step *= 0.5;
    }
    if (!accepted) {
      break;
    }
  }

  return x;
}

}  // namespace

DenseCubicTrustRegionResult solveDenseCubicTrustRegionProblemMfem(const mfem::DenseMatrix& A, const mfem::Vector& b,
                                                                  const std::vector<mfem::DenseMatrix>& cubic,
                                                                  double delta)
{
  SMITH_MARK_FUNCTION;
  verifyCubicInputs(A, b, cubic, delta);

  mfem::Vector best(b.Size());
  best = 0.0;
  double best_energy = cubicEnergy(A, b, cubic, best);
  if (delta == 0.0 || b.Size() == 0) {
    return std::make_tuple(best, best_energy);
  }

  std::vector<mfem::Vector> starts;
  starts.emplace_back(best);
  starts.emplace_back(solveQuadraticCandidate(A, b, delta));

  mfem::Vector direction(b);
  if (direction.Norml2() > 0.0) {
    direction *= delta / direction.Norml2();
    starts.emplace_back(direction);
    direction *= -1.0;
    starts.emplace_back(direction);
  }

  for (int i = 0; i < b.Size(); ++i) {
    mfem::Vector axis(b.Size());
    axis = 0.0;
    axis[i] = delta;
    starts.emplace_back(axis);
    axis[i] = -delta;
    starts.emplace_back(axis);
  }

  for (const auto& start : starts) {
    mfem::Vector candidate = projectedGradientSolve(A, b, cubic, start, delta);
    const double energy = cubicEnergy(A, b, cubic, candidate);
    if (energy < best_energy) {
      best = candidate;
      best_energy = energy;
    }
  }

  return std::make_tuple(best, best_energy);
}

TrustRegionSubspaceResult solveCubicSubspaceProblemMfem(
    const std::vector<const mfem::Vector*>& directions, const std::vector<const mfem::Vector*>& A_directions,
    const std::vector<const mfem::Vector*>& previous_A_directions, const mfem::Vector& previous_step,
    const mfem::Vector& b, double delta, int num_leftmost, bool* used_cubic)
{
  SMITH_MARK_FUNCTION;
  MFEM_VERIFY(directions.size() == A_directions.size(), "Cubic subspace directions and A_directions differ.");
  MFEM_VERIFY(directions.size() == previous_A_directions.size(),
              "Cubic subspace directions and previous_A_directions differ.");
  MFEM_VERIFY(!directions.empty(), "Cubic subspace solve requires at least one direction.");

  mfem::DenseMatrix ss = denseDot(directions, directions);
  symmetrize(ss);
  mfem::DenseMatrix T = orthonormalBasisTransform(ss);
  MFEM_VERIFY(T.Width() > 0, "No independent directions in cubic MFEM subspace solve.");

  mfem::DenseMatrix sAs = denseDot(directions, A_directions);
  symmetrize(sAs);
  mfem::DenseMatrix pAp = tripleProduct(T, sAs, T);
  symmetrize(pAp);

  mfem::DenseMatrix sDeltaA = denseDot(directions, previous_A_directions);
  sDeltaA *= -1.0;
  sDeltaA += sAs;
  symmetrize(sDeltaA);
  mfem::DenseMatrix pDeltaAp = tripleProduct(T, sDeltaA, T);
  symmetrize(pDeltaAp);

  mfem::Vector previous_coeffs = denseDot(directions, previous_step);
  previous_coeffs = projectWithTranspose(T, previous_coeffs);
  const double previous_norm_squared = dot(previous_coeffs, previous_coeffs);

  std::vector<mfem::DenseMatrix> cubic(size_t(T.Width()), mfem::DenseMatrix(T.Width()));
  for (auto& matrix : cubic) {
    matrix = 0.0;
  }
  if (previous_norm_squared > 0.0) {
    cubic = completeSymmetricCubicTensor(pDeltaAp, previous_coeffs);
  }

  const mfem::Vector sb = denseDot(directions, b);
  const mfem::Vector pb = projectWithTranspose(T, sb);
  auto [reduced_x, energy] = solveDenseCubicTrustRegionProblemMfem(pAp, pb, cubic, delta);

  mfem::Vector coeffs(T.Height());
  T.Mult(reduced_x, coeffs);
  mfem::Vector sol = combineDirections(directions, coeffs);

  auto [quadratic_sol, leftmosts, leftvals, quadratic_energy] =
      solveSubspaceProblemMfem(directions, A_directions, b, delta, num_leftmost);
  (void)quadratic_energy;

  const mfem::Vector quadratic_s_coeffs = denseDot(directions, quadratic_sol);
  const mfem::Vector quadratic_reduced_x = projectWithTranspose(T, quadratic_s_coeffs);
  const double quadratic_cubic_energy = cubicEnergy(pAp, pb, cubic, quadratic_reduced_x);
  if (quadratic_cubic_energy <= energy) {
    if (used_cubic != nullptr) {
      *used_cubic = false;
    }
    return std::make_tuple(quadratic_sol, leftmosts, leftvals, quadratic_cubic_energy);
  }

  if (used_cubic != nullptr) {
    *used_cubic = true;
  }
  return std::make_tuple(sol, leftmosts, leftvals, energy);
}

#else

DenseCubicTrustRegionResult solveDenseCubicTrustRegionProblemMfem(const mfem::DenseMatrix&, const mfem::Vector& b,
                                                                  const std::vector<mfem::DenseMatrix>&, double)
{
  throw PetscException("MFEM dense cubic trust-region solve requires MFEM LAPACK support.");
  return std::make_tuple(b, 0.0);
}

TrustRegionSubspaceResult solveCubicSubspaceProblemMfem(const std::vector<const mfem::Vector*>&,
                                                        const std::vector<const mfem::Vector*>&,
                                                        const std::vector<const mfem::Vector*>&,
                                                        const mfem::Vector&, const mfem::Vector& b, double, int, bool*)
{
  throw PetscException("MFEM dense cubic trust-region solve requires MFEM LAPACK support.");
  return std::make_tuple(b, std::vector<std::shared_ptr<mfem::Vector>> {}, std::vector<double> {}, 0.0);
}

#endif  // MFEM_USE_LAPACK

}  // namespace smith
