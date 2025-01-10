#pragma once

#include "serac/numerics/refactor/geometry.hpp"
#include "serac/numerics/functional/domain.hpp"
#include "serac/numerics/refactor/containers/ndarray.hpp"

namespace refactor {

struct ElementQuadratureRule {
  bool compact;
  nd::array<double,2> points;
  nd::array<double,1> weights;
};

enum class QuadratureRuleType { UniformStructured, UniformUnstructured, Nonuniform };

struct MeshQuadratureRule : public GeometryData< ElementQuadratureRule >{
  MeshQuadratureRule(uint32_t q, bool compact = true);
  QuadratureRuleType type;

  GeometryInfo num_qpts(const serac::Domain & domain) const; 
};

void gauss_legendre_segment_rule(uint32_t q, double * qpts, double * qwts);
void gauss_legendre_triangle_rule(uint32_t q, double * qpts, double * qwts);
void gauss_legendre_tetrahedron_rule(uint32_t q, double * qpts, double * qwts);

GeometryInfo qpts_per_geom(const MeshQuadratureRule & qrule);
GeometryInfo qpts_per_geom(const MeshQuadratureRule & qrule, int dim);

namespace impl {

template < mfem::Geometry::Type geom >
uint32_t qpe(uint32_t q) {
  if (geom == mfem::Geometry::SQUARE) { return q * q; }
  if (geom == mfem::Geometry::CUBE) { return q * q * q; }
  return q;
}

template < mfem::Geometry::Type geom >
double integration_weight(uint32_t i, const nd::view<const double, 1> w) {
  uint32_t Q = w.shape[0];
  if constexpr (geom == mfem::Geometry::SQUARE) { 
    uint32_t ix = i % Q;
    uint32_t iy = i / Q;
    return w(ix) * w(iy);
  }
  if constexpr (geom == mfem::Geometry::CUBE) { 
    uint32_t ix = i % Q;
    uint32_t iy = (i % (Q * Q)) / Q;
    uint32_t iz = i / (Q * Q);
    return w(ix) * w(iy) * w(iz);
  }
  return w(i);
}

}

}
