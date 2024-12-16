#pragma once

#include "serac/numerics/functional/tensor.hpp"
#include "serac/numerics/refactor/geometry.hpp"
#include "serac/numerics/refactor/quadrature.hpp"

#include "serac/physics/state/finite_element_state.hpp"

namespace refactor {

using Field = serac::FiniteElementState;

enum class Family { 
  H1, 
  Hcurl, 
  Hdiv, 
  DG 
};

enum class Modifier { NONE, DIAGONAL, SYM };
enum class DerivedQuantity { VALUE, DERIVATIVE };

SERAC_HOST_DEVICE constexpr bool is_scalar_valued(Family f) {
  return (f == Family::H1);
}

SERAC_HOST_DEVICE constexpr bool is_vector_valued(Family f) {
  return (f == Family::Hcurl);
}

struct FunctionSpace {
  Family family;
  uint32_t degree;
  uint32_t components;
  FunctionSpace() : family{}, degree{}, components{} {}
  FunctionSpace(Family f, uint32_t d = 2, uint32_t c = 1) : family{f}, degree{d}, components{c}{}
  FunctionSpace(Field f) {

    const auto & pgf = f.gridFunction();
    switch(pgf.FESpace()->FEColl()->GetContType()) {
      case mfem::FiniteElementCollection::CONTINUOUS:
        family = Family::H1;
        break;
      case mfem::FiniteElementCollection::TANGENTIAL:
        family = Family::Hcurl;
        break;
      case mfem::FiniteElementCollection::NORMAL:
        family = Family::Hdiv;
        break;
      case mfem::FiniteElementCollection::DISCONTINUOUS:
        family = Family::DG;
        break;
    }

    degree = uint32_t(pgf.FESpace()->FEColl()->GetOrder());

    components = uint32_t(pgf.VectorDim());

  }

  bool operator==(const FunctionSpace & other) const {
    return (components == other.components) && 
           (degree == other.degree) &&
           (family == other.family);
  }
};

struct BasisFunction {
  FunctionSpace space;
  BasisFunction(Field f) : space(f) {}
  BasisFunction(FunctionSpace s) : space(s) {}
  bool operator==(const BasisFunction & other) const {
    return (space.components == other.space.components) && 
           (space.degree == other.space.degree) &&
           (space.family == other.space.family);
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

template < mfem::Geometry::Type g, Family f >
struct FiniteElement;

template < mfem::Geometry::Type geom, Family family >
auto shape_function_derivatives(FiniteElement< geom, family > element,
                                const nd::view<const double, 2> xi) {
  if constexpr (family == Family::H1) {
    return element.evaluate_shape_function_gradients(xi);
  } 

  if constexpr (family == Family::Hcurl) {
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

  if constexpr (family == Family::Hcurl) {
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
