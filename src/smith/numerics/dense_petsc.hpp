// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

#ifdef SMITH_USE_SLEPC

#include <slepcsvd.h>
#include <slepcbv.h>
#include <vector>

struct DenseVec;

/// Dense Matrix class which wraps petsc matrix for the case of a SeqDense matrix (on 1 processor)
struct DenseMat {
  /// @brief copy constructor
  /// @param a matrix
  DenseMat(const Mat& a) : A(a) {}

  /// @brief constructor
  /// @param a matrix
  DenseMat(const DenseMat& a)
  {
    MatDuplicate(a.A, MAT_COPY_VALUES, &A);
    MatCopy(a.A, A, SAME_NONZERO_PATTERN);
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
  }

  /// @brief destructor
  ~DenseMat() { MatDestroy(&A); }

  /// @brief size
  auto size() const
  {
    int isize;
    int jsize;
    MatGetSize(A, &isize, &jsize);
    return std::make_pair(isize, jsize);
  }

  /// @brief index into
  double operator()(int i, int j) const
  {
    double val;
    MatGetValue(A, i, j, &val);
    return val;
  }

  /// @brief set value
  void setValue(int i, int j, double val) { MatSetValues(A, 1, &i, 1, &j, &val, INSERT_VALUES); }

  /// @brief matrix-vector multiply
  DenseVec operator*(const DenseVec& v) const;

  /// @brief solve
  DenseVec solve(const DenseVec& v) const;

  /// @brief multiply this by P transpose on left and P on the right
  DenseMat PtAP(const DenseMat& P) const;

  /// @brief print utility
  void print(std::string first = "") const
  {
    if (first.size()) {
      std::cout << first << ": ";
    }
    MatView(A, PETSC_VIEWER_STDOUT_SELF);
  }

  /// @brief check for nans
  bool hasNan() const
  {
    auto [rows, cols] = size();
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        double val = (*this)(i, j);
        if (val != val) return true;
      }
    }
    return false;
  }

  /// @brief  reassemble petsc dense matrix after values have been modified
  void reassemble()
  {
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
  }

  /// petsc matrix
  Mat A;
};

/// matrix inverse
/// @param a matrix
DenseMat inverse(const DenseMat& a)
{
  Mat inv;
  MatDuplicate(a.A, MAT_COPY_VALUES, &inv);
  MatSeqDenseInvert(inv);
  return inv;
}

/// compute the symmetric part
/// @param a matrix
DenseMat sym(const DenseMat& a)
{
  DenseMat b = a;
  auto [rows, cols] = b.size();
  SLIC_ERROR_IF(rows != cols, "Calling sym on a non-square DenseMat");

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < i; ++j) {
      auto val = 0.5 * a(i, j) + 0.5 * a(j, i);
      b.setValue(i, j, val);
      b.setValue(j, i, val);
    }
  }

  b.reassemble();

  return b;
}

/// Dense Vector class which wraps petsc vector for the case of a SeqDense vector (on 1 processor)
struct DenseVec {
  /// @brief constructor
  DenseVec(const Vec& vin) : v(vin) {}

  /// @brief constructor
  DenseVec(const DenseVec& vin)
  {
    VecDuplicate(vin.v, &v);
    VecCopy(vin.v, v);
  }

  /// @brief constructor from size
  DenseVec(size_t size) { VecCreateSeq(PETSC_COMM_SELF, static_cast<int>(size), &v); }

  /// @brief constructor from size
  DenseVec(int size) { VecCreateSeq(PETSC_COMM_SELF, size, &v); }

  /// @brief constructor standard vector
  DenseVec(const std::vector<double> vin)
  {
    const auto sz = vin.size();
    std::vector<int> allints(sz);
    for (size_t i = 0; i < sz; ++i) {
      allints[i] = static_cast<int>(i);
    }
    int sz_int = static_cast<int>(sz);
    VecCreateSeq(PETSC_COMM_SELF, sz_int, &v);
    VecSetValues(v, sz_int, &allints[0], &vin[0], INSERT_VALUES);
  }

  /// @brief assignment
  DenseVec& operator=(const DenseVec& vin)
  {
    VecCopy(vin.v, v);
    return *this;
  }

  /// @brief assignment from scalar
  DenseVec& operator=(const double val)
  {
    VecSet(v, val);
    return *this;
  }

  /// @brief destructor
  ~DenseVec()
  {
    if (v) VecDestroy(&v);
  }

  /// @brief negate
  DenseVec operator-() const
  {
    Vec minus;
    VecDuplicate(v, &minus);
    VecCopy(v, minus);
    VecScale(minus, -1.0);
    return minus;
  }

  /// @brief scale
  DenseVec& operator*=(double scale)
  {
    VecScale(v, scale);
    return *this;
  }

  /// @brief size
  int size() const
  {
    int isize;
    VecGetSize(v, &isize);
    return isize;
  }

  /// @brief index into
  double operator[](int i) const
  {
    double val;
    VecGetValues(v, 1, &i, &val);
    return val;
  }

  /// @brief index into
  double operator[](size_t i) const { return (*this)[int(i)]; }

  /// @brief set value
  void setValue(int i, double val) { VecSetValues(v, 1, &i, &val, INSERT_VALUES); }

  /// @brief set value
  void setValue(size_t i, double val) { setValue(int(i), val); }

  /// @brief add scaled vector
  void add(double val, const DenseVec& w) { VecAXPY(v, val, w.v); }

  /// @brief convert to standard vector
  std::vector<double> getValues() const
  {
    size_t sz = static_cast<size_t>(size());
    std::vector<double> vout(sz);
    std::vector<int> allints(sz);
    for (size_t i = 0; i < sz; ++i) {
      allints[i] = static_cast<int>(i);
    }
    int sz_int = static_cast<int>(sz);
    VecGetValues(v, sz_int, &allints[0], &vout[0]);
    return vout;
  }

  /// @brief print utility
  void print(std::string first = "") const
  {
    if (first.size()) {
      std::cout << first << ": ";
    }
    VecView(v, PETSC_VIEWER_STDOUT_SELF);
  }

  /// petsc vector
  Vec v;
};

/// @brief matrix vector multiply
DenseVec DenseMat::operator*(const DenseVec& v) const
{
  Vec out;
  auto [rows, cols] = size();
  SLIC_ERROR_IF(cols != v.size(), "Column size of dense matrix and length of multiplied vector do not match");
  VecCreateSeq(PETSC_COMM_SELF, rows, &out);
  MatMult(A, v.v, out);
  return out;
}

/// @brief matrix linear solve
DenseVec DenseMat::solve(const DenseVec& v) const
{
  Vec out;
  VecDuplicate(v.v, &out);
  MatLUFactor(A, NULL, NULL, NULL);  // not efficient if done a lot
  MatSolve(A, v.v, out);
  return out;
}

/// @brief multiply matrix by P-transpose on left, P on right
DenseMat DenseMat::PtAP(const DenseMat& P) const
{
  Mat pAp;
  MatPtAP(A, P.A, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &pAp);
  return pAp;
}

/// @brief vector dot product
double dot(const DenseVec& a, const DenseVec& b)
{
  double d;
  VecDot(a.v, b.v, &d);
  return d;
}

/// @brief add a scalar to a vector
DenseVec operator+(const DenseVec& a, double b)
{
  Vec c;
  VecDuplicate(a.v, &c);
  VecSet(c, b);
  VecAXPY(c, 1.0, a.v);
  return c;
}

DenseVec operator+(double b, const DenseVec& a) { return a + b; }

/// @brief component-wise multiplication of vectors
DenseVec operator*(const DenseVec& a, const DenseVec& b)
{
  Vec c;
  VecDuplicate(a.v, &c);
  VecPointwiseMult(c, a.v, b.v);
  return c;
}

/// @brief component-wise vector divide
DenseVec operator/(const DenseVec& a, const DenseVec& b)
{
  Vec c;
  VecDuplicate(a.v, &c);
  VecPointwiseDivide(c, a.v, b.v);
  return c;
}

/// @brief component-wise vector absolute value
DenseVec abs(const DenseVec& a)
{
  Vec absa;
  VecDuplicate(a.v, &absa);
  VecCopy(a.v, absa);
  VecAbs(absa);
  return absa;
}

/// @brief sum values in a vector
double sum(const DenseVec& a)
{
  double s;
  VecSum(a.v, &s);
  return s;
}

/// @brief l2-norm of vector
double norm(const DenseVec& a)
{
  double n;
  VecNorm(a.v, NORM_2, &n);
  return n;
}

/// @brief computes the eigenvectors and eigenvalues of a dense symmetric matrix
auto eigh(const DenseMat& Adense)
{
  auto [isize, jsize] = Adense.size();
  SLIC_ERROR_IF(isize != jsize, "Eig must be called for symmetric matrices");

  const Mat& A = Adense.A;

  EPS eps;
  EPSCreate(PETSC_COMM_SELF, &eps);
  EPSSetOperators(eps, A, NULL);
  EPSSetProblemType(eps, EPS_HEP);
  EPSSetWhichEigenpairs(eps, EPS_SMALLEST_REAL);
  EPSSetDimensions(eps, isize, PETSC_DETERMINE, PETSC_DETERMINE);
  EPSSetFromOptions(eps);

  EPSSolve(eps);

  EPSType type;
  EPSGetType(eps, &type);
  EPSGetDimensions(eps, &jsize, NULL, NULL);

  DenseVec eigenvalues(isize);
  std::vector<DenseVec> eigenvectors;
  for (int i = 0; i < isize; ++i) {
    eigenvectors.emplace_back(isize);
    double eigenvalue;
    EPSGetEigenpair(eps, i, &eigenvalue, PETSC_NULLPTR, eigenvectors[static_cast<size_t>(i)].v, PETSC_NULLPTR);
    eigenvalues.setValue(i, eigenvalue);
  }

  EPSDestroy(&eps);
  return std::make_pair(std::move(eigenvalues), std::move(eigenvectors));
}

#endif  // SMITH_USE_SLEPC
