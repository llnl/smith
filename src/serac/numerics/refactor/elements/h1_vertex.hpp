#pragma once

#include "refactor/connection.hpp" 

namespace refactor {

template <>
struct FiniteElement<Geometry::Vertex, Family::H1> {
  using source_type = double;
  using flux_type = double;

  __host__ __device__ uint32_t num_nodes() const { return 1; }

  uint32_t p;
};

} // namespace refactor
