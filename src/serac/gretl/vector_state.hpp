#pragma once

#include "state.hpp"

namespace gretl {
using Vector = std::vector<double>;
using VectorState = State<Vector>;

VectorState testing_update(const VectorState& a);

VectorState copy(const VectorState& a);

VectorState operator+(const VectorState& a, const VectorState& b);
VectorState operator*(const VectorState& a, double b);
VectorState operator*(double b, const VectorState& a);

State<double> inner_product(const VectorState& a, const VectorState& b);

namespace vec {
struct zero_clone {
  Vector operator()(const Vector& from);
};
}  // namespace vec

template <typename T>
size_t get_same_size(const std::vector<const std::vector<T>*>& vs)
{
  size_t size = vs[0]->size();
  for (int n = 1; n < vs.size(); ++n) {
    assert(size == vs[n]->size());
  }
  return size;
}
}  // namespace gretl