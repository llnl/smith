#pragma once

#include "serac/numerics/functional/family.hpp"
#include "serac/numerics/refactor/containers/ndarray.hpp"
#include "serac/numerics/refactor/finite_element.hpp"
#include "serac/physics/state/finite_element_state.hpp"
#include "serac/physics/state/finite_element_dual.hpp"

namespace refactor {

using Field = serac::FiniteElementState;

inline bool is_value(DerivedQuantity op) {
  return op == DerivedQuantity::VALUE;
}

inline bool is_derivative(DerivedQuantity op) {
  return op == DerivedQuantity::DERIVATIVE;
}

struct FieldOp {
  const DerivedQuantity op;
  const Field & field;

  FieldOp(const Field & f) : op(DerivedQuantity::VALUE), field(f){};
  FieldOp(const DerivedQuantity & o, const Field & f) : op(o), field(f){};
};

struct BasisFunctionOp {
  const Modifier mod;
  const DerivedQuantity op;
  const BasisFunction function;

  BasisFunctionOp(const BasisFunction & f) : mod(Modifier::NONE), op(DerivedQuantity::VALUE), function(f){};
  BasisFunctionOp(const DerivedQuantity & o, const BasisFunction & f) : mod(Modifier::NONE), op(o), function(f){};
  BasisFunctionOp(const Modifier & m, const DerivedQuantity & o, const BasisFunction & f) : mod(m), op(o), function(f){};
};

////////////////////////////////////////////////////////////////////////////////

FieldOp grad(const Field & f);
FieldOp curl(const Field & f);
FieldOp div(const Field & f);

BasisFunctionOp grad(const BasisFunction & phi);
BasisFunctionOp curl(const BasisFunction & phi);
BasisFunctionOp div(const BasisFunction & phi);

////////////////////////////////////////////////////////////////////////////////

// for integrating sparse matrices (e.g. mass, stiffness)
template < typename T1, typename T2, typename T3 >
struct WeightedIntegrand {
  const T1 test;
  const T2 & qdata; 
  const T3 trial;
};

// for integrating residual vectors
template < typename T1, typename T2 >
struct WeightedIntegrand< T1, T2, void > {
  const T1 test; 
  const T2 & qdata;
};

template < typename T >
struct DiagonalOnly : public T {};

template < typename T >
DiagonalOnly(T) -> DiagonalOnly<T>;

template < typename T, uint32_t n >
auto operator*(const nd::array<T,n> & data, const BasisFunction & phi) {
  return WeightedIntegrand< BasisFunctionOp, nd::array<T,n>, void >{phi, data};
}

template < typename T, uint32_t n >
auto operator*(const nd::array<T,n> & data, const BasisFunctionOp & dphi) {
  return WeightedIntegrand< BasisFunctionOp, nd::array<T,n>, void >{dphi, data};
}

template < typename T, uint32_t n >
auto dot(const nd::array<T,n> & data, const BasisFunctionOp & dphi) {
  return WeightedIntegrand< BasisFunctionOp, nd::array<T,n>, void >{dphi, data};
}

template < typename T, uint32_t n >
auto diagonal(WeightedIntegrand< BasisFunctionOp, nd::array<T,n>, BasisFunctionOp > integrand) {
  SERAC_ASSERT(
    integrand.test.function == integrand.trial.function,
    "diag(...) requires same test and trial space"
  );

  return DiagonalOnly{integrand};
}

////////////////////////////////////////////////////////////////////////////////

template < typename T, uint32_t n >
auto dot(BasisFunctionOp psi, const nd::array<T,n> & data, BasisFunctionOp phi) {
  return WeightedIntegrand< BasisFunctionOp, nd::array<T,n>, BasisFunctionOp >{phi, data, psi};
}

////////////////////////////////////////////////////////////////////////////////

double dot(const Residual & r, const Field & u);
double dot(const Field & u, const Residual & r);

////////////////////////////////////////////////////////////////////////////////

nd::array<double,3> evaluate(const FieldOp && input, Domain & domain, const MeshQuadratureRule &);

}
