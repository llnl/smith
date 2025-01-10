#pragma once

#include "serac/numerics/functional/family.hpp"
#include "serac/numerics/functional/tensor.hpp"
#include "serac/numerics/refactor/geometry.hpp"
#include "serac/numerics/refactor/quadrature.hpp"
#include "serac/numerics/refactor/containers/ndarray.hpp"

namespace refactor {

using serac::Family;

enum class Modifier { NONE, DIAGONAL, SYM };
enum class DerivedQuantity { VALUE, DERIVATIVE };

SERAC_HOST_DEVICE constexpr bool is_scalar_valued(Family f) {
  return (f == Family::H1) || (f == Family::L2);
}

SERAC_HOST_DEVICE constexpr bool is_vector_valued(Family f) {
  return (f == Family::HCURL) || (f == Family::HDIV);
}

enum class TransformationType {
  PhysicalToParent,
  TransposePhysicalToParent,
};

template < mfem::Geometry::Type g, Family f >
struct FiniteElement;

template < mfem::Geometry::Type geom, Family family >
auto shape_function_derivatives(FiniteElement< geom, family > element,
                                const nd::view<const double, 2> xi) {
  if constexpr (family == Family::H1) {
    return element.evaluate_shape_function_gradients(xi);
  } 

  if constexpr (family == Family::HCURL) {
    return element.evaluate_shape_function_curls(xi);
  } 
}

template < mfem::Geometry::Type geom, Family family >
auto weighted_shape_function_derivatives(FiniteElement< geom, family > element,
                                         const nd::view<const double, 2> xi,
                                         const nd::view<const double, 1> weights) {
  if constexpr (family == Family::H1) {
    return element.evaluate_weighted_shape_function_gradients(xi, weights);
  } 

  if constexpr (family == Family::HCURL) {
    return element.evaluate_weighted_shape_function_curls(xi, weights);
  } 
}

SERAC_HOST_DEVICE constexpr serac::mat2 face_transformation(int8_t id) {
  constexpr serac::mat2 matrices[5] = {
    {{{0, 1}, {1, 0}}},
    {{{-1, 0}, {-1, 1}}},
    {{{1, -1}, {0, -1}}},
    {{{-1, -1}, {0, 1}}}, 
    {{{1, 0}, {-1, -1}}}
  };

  return matrices[id-1];
}

// sam: these orientations are for refactor's face orientation convention
SERAC_HOST_DEVICE constexpr int8_t face_transformation_id(TransformationType type, int orientation) {
  if (type == TransformationType::PhysicalToParent) {
    constexpr int8_t LUT[3] = {1, 2, 3};
    return LUT[orientation];
  } else {
    constexpr int8_t LUT[3] = {1, 4, 5};
    return LUT[orientation];
  }
}

}

#include "serac/numerics/refactor/elements/h1_vertex.hpp"
#include "serac/numerics/refactor/elements/h1_edge.hpp"
#include "serac/numerics/refactor/elements/h1_triangle.hpp"
#include "serac/numerics/refactor/elements/h1_quadrilateral.hpp"
#include "serac/numerics/refactor/elements/h1_tetrahedron.hpp"
#include "serac/numerics/refactor/elements/h1_hexahedron.hpp"

#include "serac/numerics/refactor/elements/hcurl_vertex.hpp"
#include "serac/numerics/refactor/elements/hcurl_edge.hpp"
#include "serac/numerics/refactor/elements/hcurl_triangle.hpp"
#include "serac/numerics/refactor/elements/hcurl_quadrilateral.hpp"
#include "serac/numerics/refactor/elements/hcurl_tetrahedron.hpp"
#include "serac/numerics/refactor/elements/hcurl_hexahedron.hpp"
