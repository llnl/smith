#pragma once

#include "refactor/interpolation.hpp"

namespace refactor {

template <>
struct FiniteElement < Geometry::Vertex, Family::Hcurl >{

  __host__ __device__ uint32_t num_nodes() const { return 0; }
  void nodes(nd::view<double, 2> xi) {}
  void directions(nd::view<double, 2> xi) {}

  __host__ __device__ uint32_t num_interior_nodes() { return 0; }
  void interior_nodes(nd::view<double, 2> xi) {}
  void interior_directions(nd::view<double, 2> xi) {}

  __host__ __device__ void indices(const GeometryInfo & offsets, const Connection * edge, uint32_t * ids) {}

  uint32_t p;

};

} // namespace refactor
