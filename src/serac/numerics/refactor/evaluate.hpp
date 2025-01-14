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

////////////////////////////////////////////////////////////////////////////////

FieldOp grad(const Field & f);
FieldOp curl(const Field & f);
FieldOp div(const Field & f);

////////////////////////////////////////////////////////////////////////////////

double dot(const Residual & r, const Field & u);
double dot(const Field & u, const Residual & r);

////////////////////////////////////////////////////////////////////////////////

nd::array<double,3> evaluate(const FieldOp && input, Domain & domain, const MeshQuadratureRule &);

}
