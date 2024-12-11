#pragma once

#include "refactor/interpolation.hpp"

namespace refactor {

template <>
struct FiniteElement < Geometry::Vertex, Family::Hcurl >{
  __host__ __device__ uint32_t num_nodes() const { return 0; }

  uint32_t p;
};

} // namespace refactor
