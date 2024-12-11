#pragma once

#include "refactor/geometry.hpp"
#include "refactor/quadrature.hpp"

namespace refactor {

enum class Family { 
  H1, 
  Hcurl, 
  Hdiv, 
  DG 
};

__host__ __device__ constexpr bool is_scalar_valued(Family f) {
  return (f == Family::H1);
}

__host__ __device__ constexpr bool is_vector_valued(Family f) {
  return (f == Family::Hcurl);
}

struct FunctionSpace {
  Family family;
  uint32_t degree;
  uint32_t components;
  FunctionSpace() : family{}, degree{}, components{} {}
  FunctionSpace(Family f, uint32_t d = 2, uint32_t c = 1) : family{f}, degree{d}, components{c}{}

  bool operator==(const FunctionSpace & other) const {
    return (components == other.components) && 
           (degree == other.degree) &&
           (family == other.family);
  }
};

GeometryInfo nodes_per_geom(FunctionSpace space);
GeometryInfo interior_nodes_per_geom(FunctionSpace space);

GeometryInfo dofs_per_geom(FunctionSpace space);
GeometryInfo interior_dofs_per_geom(FunctionSpace space);

enum class TransformationType {
  PhysicalToParent,
  TransposePhysicalToParent,
};

template < Geometry g, Family f >
struct FiniteElement;

template < Geometry geom, Family family >
auto shape_function_derivatives(FiniteElement< geom, family > element,
                                const nd::view<const double, 2> xi) {
  if constexpr (family == Family::H1) {
    return element.evaluate_shape_function_gradients(xi);
  } 

  if constexpr (family == Family::Hcurl) {
    return element.evaluate_shape_function_curls(xi);
  } 
}

template < Geometry geom, Family family >
auto weighted_shape_function_derivatives(FiniteElement< geom, family > element,
                                         const nd::view<const double, 2> xi,
                                         const nd::view<const double, 1> weights) {
  if constexpr (family == Family::H1) {
    return element.evaluate_weighted_shape_function_gradients(xi, weights);
  } 

  if constexpr (family == Family::Hcurl) {
    return element.evaluate_weighted_shape_function_curls(xi, weights);
  } 
}

__host__ __device__ constexpr fm::mat2 face_transformation(int8_t id) {
  constexpr fm::mat2 matrices[5] = {
    {{{0, 1}, {1, 0}}},
    {{{-1, 0}, {-1, 1}}},
    {{{1, -1}, {0, -1}}},
    {{{-1, -1}, {0, 1}}}, 
    {{{1, 0}, {-1, -1}}}
  };

  return matrices[id-1];
}

__host__ __device__ constexpr int8_t face_transformation_id(TransformationType type, int orientation) {
  if (type == TransformationType::PhysicalToParent) {
    constexpr int8_t LUT[3] = {1, 2, 3};
    return LUT[orientation];
  } else {
    constexpr int8_t LUT[3] = {1, 4, 5};
    return LUT[orientation];
  }
}

}

#include "refactor/elements/h1_vertex.hpp"
#include "refactor/elements/h1_edge.hpp"
#include "refactor/elements/h1_triangle.hpp"
#include "refactor/elements/h1_quadrilateral.hpp"
#include "refactor/elements/h1_tetrahedron.hpp"
#include "refactor/elements/h1_hexahedron.hpp"

#include "refactor/elements/hcurl_vertex.hpp"
#include "refactor/elements/hcurl_edge.hpp"
#include "refactor/elements/hcurl_triangle.hpp"
#include "refactor/elements/hcurl_quadrilateral.hpp"
#include "refactor/elements/hcurl_tetrahedron.hpp"
#include "refactor/elements/hcurl_hexahedron.hpp"

#include "refactor/elements/dg_vertex.hpp"
#include "refactor/elements/dg_edge.hpp"
#include "refactor/elements/dg_triangle.hpp"
#include "refactor/elements/dg_quadrilateral.hpp"
#include "refactor/elements/dg_tetrahedron.hpp"
#include "refactor/elements/dg_hexahedron.hpp"
