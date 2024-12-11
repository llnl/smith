#pragma once

#include "refactor/connection.hpp" 

namespace refactor {

template <>
struct FiniteElement<Geometry::Tetrahedron, Family::H1> {

  using source_type = vec1;
  using flux_type = vec3;

  using value_type = vec1;
  using grad_type = vec3;

  static constexpr int dim = 3;

  __host__ __device__ constexpr uint32_t num_nodes() const { return Tetrahedron::number(p + 1); }

  void nodes(nd::view< double, 2 > xi) const {
    if (p == 1) {
      xi(0, 0) = 0.0; xi(0, 1) = 0.0; xi(0, 2) = 0.0;
      xi(1, 0) = 1.0; xi(1, 1) = 0.0; xi(1, 2) = 0.0;
      xi(2, 0) = 0.0; xi(2, 1) = 1.0; xi(2, 2) = 0.0;

      xi(3, 0) = 0.0; xi(3, 1) = 0.0; xi(3, 2) = 1.0;
    }
    if (p == 2) {
      xi(0, 0) = 0.0; xi(0, 1) = 0.0; xi(0, 2) = 0.0;
      xi(1, 0) = 0.5; xi(1, 1) = 0.0; xi(1, 2) = 0.0;
      xi(2, 0) = 1.0; xi(2, 1) = 0.0; xi(2, 2) = 0.0;
      xi(3, 0) = 0.0; xi(3, 1) = 0.5; xi(3, 2) = 0.0;
      xi(4, 0) = 0.5; xi(4, 1) = 0.5; xi(4, 2) = 0.0;
      xi(5, 0) = 0.0; xi(5, 1) = 1.0; xi(5, 2) = 0.0;
                                         
      xi(6, 0) = 0.0; xi(6, 1) = 0.0; xi(6, 2) = 0.5;
      xi(7, 0) = 0.5; xi(7, 1) = 0.0; xi(7, 2) = 0.5;
      xi(8, 0) = 0.0; xi(8, 1) = 0.5; xi(8, 2) = 0.5;
                                         
      xi(9, 0) = 0.0; xi(9, 1) = 0.0; xi(9, 2) = 1.0;
    }
    if (p == 3) {
      constexpr double s1 = 0.2763932022500210;
      constexpr double s2 = 0.7236067977499790;
      constexpr double s3 = 0.3333333333333333;

      xi( 0, 0) = 0.0; xi( 0, 1) = 0.0; xi( 0, 2) = 0.0;
      xi( 1, 0) =  s1; xi( 1, 1) = 0.0; xi( 1, 2) = 0.0;
      xi( 2, 0) =  s2; xi( 2, 1) = 0.0; xi( 2, 2) = 0.0;
      xi( 3, 0) = 1.0; xi( 3, 1) = 0.0; xi( 3, 2) = 0.0;
      xi( 4, 0) = 0.0; xi( 4, 1) =  s1; xi( 4, 2) = 0.0;
      xi( 5, 0) =  s3; xi( 5, 1) =  s3; xi( 5, 2) = 0.0;
      xi( 6, 0) =  s2; xi( 6, 1) =  s1; xi( 6, 2) = 0.0;
      xi( 7, 0) = 0.0; xi( 7, 1) =  s2; xi( 7, 2) = 0.0;
      xi( 8, 0) =  s1; xi( 8, 1) =  s2; xi( 8, 2) = 0.0;
      xi( 9, 0) = 0.0; xi( 9, 1) = 1.0; xi( 9, 2) = 0.0;
                                           
      xi(10, 0) = 0.0; xi(10, 1) = 0.0; xi(10, 2) =  s1;
      xi(11, 0) =  s3; xi(11, 1) = 0.0; xi(11, 2) =  s3;
      xi(12, 0) =  s2; xi(12, 1) = 0.0; xi(12, 2) =  s1;
      xi(13, 0) = 0.0; xi(13, 1) =  s3; xi(13, 2) =  s3;
      xi(14, 0) =  s3; xi(14, 1) =  s3; xi(14, 2) =  s3;
      xi(15, 0) = 0.0; xi(15, 1) =  s2; xi(15, 2) =  s1;
                                           
      xi(16, 0) = 0.0; xi(16, 1) = 0.0; xi(16, 2) =  s2;
      xi(17, 0) =  s1; xi(17, 1) = 0.0; xi(17, 2) =  s2;
      xi(18, 0) = 0.0; xi(18, 1) =  s1; xi(18, 2) =  s2;
                                           
      xi(19, 0) = 0.0; xi(19, 1) = 0.0; xi(19, 2) = 1.0;
    }
  }

  // interior nodes only show up for tets when p >= 4, but this only supports p <= 3
  __host__ __device__ uint32_t num_interior_nodes() const { return 0; }

  void interior_nodes(nd::view< double, 2 > /* xi */) const {}

  __host__ __device__ void indices(const GeometryInfo & offsets, const Connection * tet, uint32_t * indices) const {
    
    const Connection * vertex = tet;
    const Connection * edge = tet + Tetrahedron::edge_offset;
    const Connection * face = tet + Tetrahedron::tri_offset;

    if (p == 1) {
      indices[0] = offsets.vert + vertex[0].index;
      indices[1] = offsets.vert + vertex[1].index;
      indices[2] = offsets.vert + vertex[2].index;
      indices[3] = offsets.vert + vertex[3].index;
      return;
    }

    if (p == 2) {
      indices[0] = offsets.vert + vertex[0].index;
      indices[1] = offsets.edge +   edge[0].index;
      indices[2] = offsets.vert + vertex[1].index;
      indices[3] = offsets.edge +   edge[2].index;
      indices[4] = offsets.edge +   edge[1].index;
      indices[5] = offsets.vert + vertex[2].index;
      indices[6] = offsets.edge +   edge[3].index;
      indices[7] = offsets.edge +   edge[4].index;
      indices[8] = offsets.edge +   edge[5].index;
      indices[9] = offsets.vert + vertex[3].index;
    }

    if (p == 3) {
      indices[ 0] = offsets.vert +     vertex[0].index;
      indices[ 1] = offsets.edge + 2 *   edge[0].index + 0;
      indices[ 2] = offsets.edge + 2 *   edge[0].index + 1;
      indices[ 3] = offsets.vert +     vertex[1].index;
      indices[ 4] = offsets.edge + 2 *   edge[2].index + 1;
      indices[ 5] = offsets.tri  +       face[0].index + 0;
      indices[ 6] = offsets.edge + 2 *   edge[1].index + 0;
      indices[ 7] = offsets.edge + 2 *   edge[2].index + 0;
      indices[ 8] = offsets.edge + 2 *   edge[1].index + 1;
      indices[ 9] = offsets.vert +     vertex[2].index + 0;

      indices[10] = offsets.edge + 2 *   edge[3].index + 0;
      indices[11] = offsets.tri  +       face[1].index;
      indices[12] = offsets.edge + 2 *   edge[4].index + 0;
      indices[13] = offsets.tri  +       face[3].index;
      indices[14] = offsets.tri  +       face[2].index;
      indices[15] = offsets.edge + 2 *   edge[5].index + 0;

      indices[16] = offsets.edge + 2 *   edge[3].index + 1;
      indices[17] = offsets.edge + 2 *   edge[4].index + 1;
      indices[18] = offsets.edge + 2 *   edge[5].index + 1;

      indices[19] = offsets.vert +     vertex[3].index;

      if (flip(edge[0])) { fm::swap(indices[ 1], indices[ 2]); }
      if (flip(edge[1])) { fm::swap(indices[ 6], indices[ 8]); }
      if (flip(edge[2])) { fm::swap(indices[ 7], indices[ 4]); }
      if (flip(edge[3])) { fm::swap(indices[10], indices[16]); }
      if (flip(edge[4])) { fm::swap(indices[12], indices[17]); }
      if (flip(edge[5])) { fm::swap(indices[15], indices[18]); }
      return;
    }

  }

  constexpr double shape_function(vec3 xi, uint32_t i) const {
    // expressions generated symbolically by mathematica
    if (p == 1) {
      if (i == 0) return 1-xi[0]-xi[1]-xi[2];
      if (i == 1) return xi[0];
      if (i == 2) return xi[1];
      if (i == 3) return xi[2];
    }
    if (p == 2) {
      if (i == 0) return (-1+xi[0]+xi[1]+xi[2])*(-1+2*xi[0]+2*xi[1]+2*xi[2]);
      if (i == 1) return -4*xi[0]*(-1+xi[0]+xi[1]+xi[2]);
      if (i == 2) return xi[0]*(-1+2*xi[0]);
      if (i == 3) return -4*xi[1]*(-1+xi[0]+xi[1]+xi[2]);
      if (i == 4) return 4*xi[0]*xi[1];
      if (i == 5) return xi[1]*(-1+2*xi[1]);
      if (i == 6) return -4*xi[2]*(-1+xi[0]+xi[1]+xi[2]);
      if (i == 7) return 4*xi[0]*xi[2];
      if (i == 8) return 4*xi[1]*xi[2];
      if (i == 9) return xi[2]*(-1+2*xi[2]);
    }
    if (p == 3) {
      double sqrt5 = 2.23606797749978981;
      if (i ==  0) return -((-1+xi[0]+xi[1]+xi[2])*(1+5*xi[0]*xi[0]+5*xi[1]*xi[1]+5*(-1+xi[2])*xi[2]+xi[1]*(-5+11*xi[2])+xi[0]*(-5+11*xi[1]+11*xi[2])));
      if (i ==  1) return (5*xi[0]*(-1+xi[0]+xi[1]+xi[2])*(-1-sqrt5+2*sqrt5*xi[0]+(3+sqrt5)*xi[1]+(3+sqrt5)*xi[2]))/2.;
      if (i ==  2) return (-5*xi[0]*(-1+xi[0]+xi[1]+xi[2])*(1-sqrt5+2*sqrt5*xi[0]+(-3+sqrt5)*xi[1]+(-3+sqrt5)*xi[2]))/2.;
      if (i ==  3) return xi[0]*(1+5*xi[0]*xi[0]+xi[1]-xi[1]*xi[1]+xi[2]-xi[1]*xi[2]-xi[2]*xi[2]-xi[0]*(5+xi[1]+xi[2]));
      if (i ==  4) return (5*xi[1]*(-1+xi[0]+xi[1]+xi[2])*(-1-sqrt5+(3+sqrt5)*xi[0]+2*sqrt5*xi[1]+(3+sqrt5)*xi[2]))/2.;
      if (i ==  5) return -27*xi[0]*xi[1]*(-1+xi[0]+xi[1]+xi[2]);
      if (i ==  6) return (5*xi[0]*xi[1]*(-2+(3+sqrt5)*xi[0]-(-3+sqrt5)*xi[1]))/2.;
      if (i ==  7) return (-5*xi[1]*(-1+xi[0]+xi[1]+xi[2])*(1-sqrt5+(-3+sqrt5)*xi[0]+2*sqrt5*xi[1]+(-3+sqrt5)*xi[2]))/2.;
      if (i ==  8) return (-5*xi[0]*xi[1]*(2+(-3+sqrt5)*xi[0]-(3+sqrt5)*xi[1]))/2.;
      if (i ==  9) return xi[1]*(1-xi[0]*xi[0]+5*xi[1]*xi[1]+xi[2]-xi[2]*xi[2]-xi[1]*(5+xi[2])-xi[0]*(-1+xi[1]+xi[2]));
      if (i == 10) return (5*xi[2]*(-1+xi[0]+xi[1]+xi[2])*(-439204-196418*sqrt5+(710647+317811*sqrt5)*xi[0]+(710647+317811*sqrt5)*xi[1]+606965*xi[2]+271443*sqrt5*xi[2]))/(271443+121393*sqrt5);
      if (i == 11) return -27*xi[0]*xi[2]*(-1+xi[0]+xi[1]+xi[2]);
      if (i == 12) return (5*xi[0]*xi[2]*(-5-3*sqrt5+(15+7*sqrt5)*xi[0]+2*sqrt5*xi[2]))/(5+3*sqrt5);
      if (i == 13) return -27*xi[1]*xi[2]*(-1+xi[0]+xi[1]+xi[2]);
      if (i == 14) return 27*xi[0]*xi[1]*xi[2];
      if (i == 15) return (5*xi[1]*xi[2]*(-5-3*sqrt5+(15+7*sqrt5)*xi[1]+2*sqrt5*xi[2]))/(5+3*sqrt5);
      if (i == 16) return (5*xi[2]*(-1+xi[0]+xi[1]+xi[2])*(88555+39603*sqrt5+(54730+24476*sqrt5)*xi[0]+(54730+24476*sqrt5)*xi[1]-5*(64079+28657*sqrt5)*xi[2]))/(143285+64079*sqrt5);
      if (i == 17) return (-5*xi[0]*xi[2]*(2+(-3+sqrt5)*xi[0]-(3+sqrt5)*xi[2]))/2.;
      if (i == 18) return (-5*xi[1]*xi[2]*(2+(-3+sqrt5)*xi[1]-(3+sqrt5)*xi[2]))/2.;
      if (i == 19) return -(xi[2]*(-1+xi[0]*xi[0]+xi[1]*xi[1]+xi[1]*(-1+xi[2])-5*(-1+xi[2])*xi[2]+xi[0]*(-1+xi[1]+xi[2])));
    }

    return {};
  }

  vec3 shape_function_gradient(vec3 xi, uint32_t i) const {
    // expressions generated symbolically by mathematica
    if (p == 1) {
      if (i == 0) return {-1, -1, -1};
      if (i == 1) return {1, 0, 0};
      if (i == 2) return {0, 1, 0};
      if (i == 3) return {0, 0, 1};
    }
    if (p == 2) {
      if (i == 0) return {-3+4*xi[0]+4*xi[1]+4*xi[2], -3+4*xi[0]+4*xi[1]+4*xi[2], -3+4*xi[0]+4*xi[1]+4*xi[2]};
      if (i == 1) return {-4*(-1+2*xi[0]+xi[1]+xi[2]), -4*xi[0], -4*xi[0]};
      if (i == 2) return {-1+4*xi[0], 0, 0};
      if (i == 3) return {-4*xi[1], -4*(-1+xi[0]+2*xi[1]+xi[2]), -4*xi[1]};
      if (i == 4) return {4*xi[1], 4*xi[0], 0};
      if (i == 5) return {0, -1+4*xi[1], 0};
      if (i == 6) return {-4*xi[2], -4*xi[2], -4*(-1+xi[0]+xi[1]+2*xi[2])};
      if (i == 7) return {4*xi[2], 0, 4*xi[0]};
      if (i == 8) return {0, 4*xi[2], 4*xi[1]};
      if (i == 9) return {0, 0, -1+4*xi[2]};
    }
    if (p == 3) {
      double sqrt5 = 2.23606797749978981;
      if (i ==  0) return {-6-15*xi[0]*xi[0]-16*xi[1]*xi[1]+xi[1]*(21-33*xi[2])+(21-16*xi[2])*xi[2]-4*xi[0]*(-5+8*xi[1]+8*xi[2]), -6-16*xi[0]*xi[0]+20*xi[1]+xi[0]*(21-32*xi[1]-33*xi[2])+21*xi[2]-(3*xi[1]+4*xi[2])*(5*xi[1]+4*xi[2]), -6-16*xi[0]*xi[0]+21*xi[1]+xi[0]*(21-33*xi[1]-32*xi[2])+20*xi[2]-(4*xi[1]+3*xi[2])*(4*xi[1]+5*xi[2])};
      if (i ==  1) return {(5*(6*sqrt5*xi[0]*xi[0]+xi[0]*(-2-6*sqrt5+6*(1+sqrt5)*xi[1]+6*(1+sqrt5)*xi[2])+(-1+xi[1]+xi[2])*(-1-sqrt5+(3+sqrt5)*xi[1]+(3+sqrt5)*xi[2])))/2., (5*xi[0]*(-4-2*sqrt5+3*(1+sqrt5)*xi[0]+2*(3+sqrt5)*xi[1]+2*(3+sqrt5)*xi[2]))/2., (5*xi[0]*(-4-2*sqrt5+3*(1+sqrt5)*xi[0]+2*(3+sqrt5)*xi[1]+2*(3+sqrt5)*xi[2]))/2.};
      if (i ==  2) return {-15*sqrt5*xi[0]*xi[0]-(5*(-1+xi[1]+xi[2])*(1-sqrt5+(-3+sqrt5)*xi[1]+(-3+sqrt5)*xi[2]))/2.-5*xi[0]*(1-3*sqrt5+3*(-1+sqrt5)*xi[1]+3*(-1+sqrt5)*xi[2]), (-5*xi[0]*(4-2*sqrt5+3*(-1+sqrt5)*xi[0]+2*(-3+sqrt5)*xi[1]+2*(-3+sqrt5)*xi[2]))/2., (-5*xi[0]*(4-2*sqrt5+3*(-1+sqrt5)*xi[0]+2*(-3+sqrt5)*xi[1]+2*(-3+sqrt5)*xi[2]))/2.};
      if (i ==  3) return {1+15*xi[0]*xi[0]+xi[1]-xi[1]*xi[1]+xi[2]-xi[1]*xi[2]-xi[2]*xi[2]-2*xi[0]*(5+xi[1]+xi[2]), -(xi[0]*(-1+xi[0]+2*xi[1]+xi[2])), -(xi[0]*(-1+xi[0]+xi[1]+2*xi[2]))};
      if (i ==  4) return {(5*xi[1]*(-2*(2+sqrt5)+2*(3+sqrt5)*xi[0]+3*(1+sqrt5)*xi[1]+2*(3+sqrt5)*xi[2]))/2., 15*sqrt5*xi[1]*xi[1]+5*xi[1]*(-1-3*sqrt5+3*(1+sqrt5)*xi[0]+3*(1+sqrt5)*xi[2])+(5*(-1+xi[0]+xi[2])*(-1-sqrt5+(3+sqrt5)*xi[0]+(3+sqrt5)*xi[2]))/2., (5*xi[1]*(-2*(2+sqrt5)+2*(3+sqrt5)*xi[0]+3*(1+sqrt5)*xi[1]+2*(3+sqrt5)*xi[2]))/2.};
      if (i ==  5) return {-27*xi[1]*(-1+2*xi[0]+xi[1]+xi[2]), -27*xi[0]*(-1+xi[0]+2*xi[1]+xi[2]), -27*xi[0]*xi[1]};
      if (i ==  6) return {(-5*xi[1]*(2-2*(3+sqrt5)*xi[0]+(-3+sqrt5)*xi[1]))/2., (5*xi[0]*(-2+(3+sqrt5)*xi[0]-2*(-3+sqrt5)*xi[1]))/2., 0};
      if (i ==  7) return {(-5*xi[1]*(4-2*sqrt5+2*(-3+sqrt5)*xi[0]+3*(-1+sqrt5)*xi[1]+2*(-3+sqrt5)*xi[2]))/2., -15*sqrt5*xi[1]*xi[1]-(5*(-1+xi[0]+xi[2])*(1-sqrt5+(-3+sqrt5)*xi[0]+(-3+sqrt5)*xi[2]))/2.-5*xi[1]*(1-3*sqrt5+3*(-1+sqrt5)*xi[0]+3*(-1+sqrt5)*xi[2]), (-5*xi[1]*(4-2*sqrt5+2*(-3+sqrt5)*xi[0]+3*(-1+sqrt5)*xi[1]+2*(-3+sqrt5)*xi[2]))/2.};
      if (i ==  8) return {(5*xi[1]*(-2-2*(-3+sqrt5)*xi[0]+(3+sqrt5)*xi[1]))/2., (-5*xi[0]*(2+(-3+sqrt5)*xi[0]-2*(3+sqrt5)*xi[1]))/2., 0};
      if (i ==  9) return {-(xi[1]*(-1+2*xi[0]+xi[1]+xi[2])), 1-xi[0]*xi[0]+15*xi[1]*xi[1]+xi[2]-xi[2]*xi[2]-2*xi[1]*(5+xi[2])-xi[0]*(-1+2*xi[1]+xi[2]), -(xi[1]*(-1+xi[0]+xi[1]+2*xi[2]))};
      if (i == 10) return {(5*xi[2]*(-2*(2+sqrt5)+2*(3+sqrt5)*xi[0]+2*(3+sqrt5)*xi[1]+3*(1+sqrt5)*xi[2]))/2., (5*xi[2]*(-2*(2+sqrt5)+2*(3+sqrt5)*xi[0]+2*(3+sqrt5)*xi[1]+3*(1+sqrt5)*xi[2]))/2., (5*(1+sqrt5+(3+sqrt5)*xi[0]*xi[0]-2*(2+sqrt5)*xi[1]+(3+sqrt5)*xi[1]*xi[1]+6*(1+sqrt5)*xi[1]*xi[2]+2*xi[2]*(-1-3*sqrt5+3*sqrt5*xi[2])+2*xi[0]*(-2-sqrt5+(3+sqrt5)*xi[1]+3*(1+sqrt5)*xi[2])))/2.};
      if (i == 11) return {-27*xi[2]*(-1+2*xi[0]+xi[1]+xi[2]), -27*xi[0]*xi[2], -27*xi[0]*(-1+xi[0]+xi[1]+2*xi[2])};
      if (i == 12) return {(-5*xi[2]*(2-2*(3+sqrt5)*xi[0]+(-3+sqrt5)*xi[2]))/2., 0, (5*xi[0]*(-2+(3+sqrt5)*xi[0]-2*(-3+sqrt5)*xi[2]))/2.};
      if (i == 13) return {-27*xi[1]*xi[2], -27*xi[2]*(-1+xi[0]+2*xi[1]+xi[2]), -27*xi[1]*(-1+xi[0]+xi[1]+2*xi[2])};
      if (i == 14) return {27*xi[1]*xi[2], 27*xi[0]*xi[2], 27*xi[0]*xi[1]};
      if (i == 15) return {0, (-5*xi[2]*(2-2*(3+sqrt5)*xi[1]+(-3+sqrt5)*xi[2]))/2., (5*xi[1]*(-2+(3+sqrt5)*xi[1]-2*(-3+sqrt5)*xi[2]))/2.};
      if (i == 16) return {(-5*xi[2]*(4-2*sqrt5+2*(-3+sqrt5)*xi[0]+2*(-3+sqrt5)*xi[1]+3*(-1+sqrt5)*xi[2]))/2., (-5*xi[2]*(4-2*sqrt5+2*(-3+sqrt5)*xi[0]+2*(-3+sqrt5)*xi[1]+3*(-1+sqrt5)*xi[2]))/2., (-5*(-3+sqrt5)*xi[0]*xi[0])/2.-5*xi[0]*(2-sqrt5+(-3+sqrt5)*xi[1]+3*(-1+sqrt5)*xi[2])-(5*(-1+sqrt5+(-3+sqrt5)*xi[1]*xi[1]+2*xi[2]*(1-3*sqrt5+3*sqrt5*xi[2])+xi[1]*(4-2*sqrt5+6*(-1+sqrt5)*xi[2])))/2.};
      if (i == 17) return {(5*xi[2]*(-2-2*(-3+sqrt5)*xi[0]+(3+sqrt5)*xi[2]))/2., 0, (-5*xi[0]*(2+(-3+sqrt5)*xi[0]-2*(3+sqrt5)*xi[2]))/2.};
      if (i == 18) return {0, (5*xi[2]*(-2-2*(-3+sqrt5)*xi[1]+(3+sqrt5)*xi[2]))/2., (-5*xi[1]*(2+(-3+sqrt5)*xi[1]-2*(3+sqrt5)*xi[2]))/2.};
      if (i == 19) return {-(xi[2]*(-1+2*xi[0]+xi[1]+xi[2])), -(xi[2]*(-1+xi[0]+2*xi[1]+xi[2])), 1+xi[0]-xi[0]*xi[0]+xi[1]-xi[0]*xi[1]-xi[1]*xi[1]-2*(5+xi[0]+xi[1])*xi[2]+15*xi[2]*xi[2]};
    }

    return {};
  }

  vec3 shape_function_derivative(vec3 xi, uint32_t i) const {
    return shape_function_gradient(xi, i);
  }

  double interpolate(vec3 xi, double * values) const {
    double interpolated_value = 0.0;
    for (int i = 0; i < num_nodes(); i++) {
      interpolated_value += values[i] * shape_function(xi, i);
    }
    return interpolated_value;
  }

  vec3 gradient(vec3 xi, double * values) const {
    vec3 interpolated_gradient{};
    for (int i = 0; i < num_nodes(); i++) {
      interpolated_gradient += values[i] * shape_function_gradient(xi, i);
    }
    return interpolated_gradient;
  }

  __host__ __device__ uint32_t batch_interpolation_scratch_space(nd::view<const double,2> xi) const {
    return 0;
  }

  nd::array< double, 2 > evaluate_shape_functions(nd::view<const double, 2> xi) const {
    uint32_t q = xi.shape[0];
    nd::array<double, 2> shape_fns({q, num_nodes()});
    for (int i = 0; i < q; i++) {
      GaussLobattoInterpolationTetrahedron(&xi(i, 0), p, &shape_fns(i, 0));
    }
    return shape_fns;
  }

  nd::array< double, 3 > evaluate_shape_function_gradients(nd::view<const double, 2> xi) const {
    uint32_t q = xi.shape[0];
    nd::array<double, 3> shape_fn_grads({q, num_nodes(), dim});
    for (int i = 0; i < q; i++) {
      GaussLobattoInterpolationDerivativeTetrahedron(&xi(i, 0), p, &shape_fn_grads(i, 0, 0));
    }
    return shape_fn_grads;
  }

  void interpolate(nd::view<value_type> values_q, nd::view<const double, 1> values_e, nd::view<const double, 2> shape_fns, double * /*buffer*/) const {
    int nnodes = num_nodes();
    int nqpts = values_q.shape[0];

    for (int q = 0; q < nqpts; q++) {
      double sum = 0.0;
      for (int i = 0; i < nnodes; i++) {
        sum += shape_fns(q, i) * values_e(i);
      }
      values_q(q) = sum;
    }
  }

  void gradient(nd::view<grad_type> gradients_q, nd::view<const double, 1> values_e, nd::view<const double, 3> shape_fn_grads, double * /*buffer*/) const {
    int nnodes = num_nodes();
    int nqpts = gradients_q.shape[0];

    for (int j = 0; j < nqpts; j++) {
      grad_type sum{};
      for (int i = 0; i < nnodes; i++) {
        sum[0] += values_e(i) * shape_fn_grads(j, i, 0);
        sum[1] += values_e(i) * shape_fn_grads(j, i, 1);
        sum[2] += values_e(i) * shape_fn_grads(j, i, 2);
      }
      gradients_q(j) = sum;
    }
    
  }

  nd::array< double, 2 > evaluate_weighted_shape_functions(nd::view<const double,2> xi,
                                                           nd::view<const double,1> weights) const {
    uint32_t nnodes = num_nodes();
    uint32_t q = xi.shape[0];
    nd::array<double, 2> shape_fns({q, nnodes});
    for (int i = 0; i < q; i++) {
      GaussLobattoInterpolationTetrahedron(&xi(i, 0), p, &shape_fns(i, 0));
      for (int j = 0; j < nnodes; j++) {
        shape_fns(i, j) = shape_fns(i, j) * weights(i);
      }
    }
    return shape_fns;
  }

  __host__ __device__ void integrate_source(nd::view<double> residual_e, nd::view<const source_type> source_q, nd::view<const double, 2> shape_fn, double * /*buffer*/) const {
    int nnodes = num_nodes();
    int nqpts = source_q.shape[0];

    for (int i = 0; i < nnodes; i++) {
      double sum = 0.0;
      for (int q = 0; q < nqpts; q++) {
        sum += shape_fn(q, i) * source_q(q);
      }
      residual_e(i) = sum;
    }
  }

  nd::array< double, 3 > evaluate_weighted_shape_function_gradients(nd::view<const double,2> xi,
                                                                    nd::view<const double,1> weights) const {
    uint32_t nnodes = num_nodes();
    uint32_t q = xi.shape[0];
    nd::array<double, 3> shape_fn_grads({q, nnodes, dim});
    for (int i = 0; i < q; i++) {
      GaussLobattoInterpolationDerivativeTetrahedron(&xi(i, 0), p, &shape_fn_grads(i, 0, 0));
      for (int j = 0; j < nnodes; j++) {
        shape_fn_grads(i, j, 0) = shape_fn_grads(i, j, 0) * weights(i);
        shape_fn_grads(i, j, 1) = shape_fn_grads(i, j, 1) * weights(i);
        shape_fn_grads(i, j, 2) = shape_fn_grads(i, j, 2) * weights(i);
      }
    }

    return shape_fn_grads;
  }

  void integrate_flux(nd::view<double> residual_e, nd::view<const flux_type> flux_q, nd::view<const double, 3> shape_fn_grads, double * /* buffer */) const {
    int nnodes = num_nodes();
    int nqpts = flux_q.shape[0];

    for (int i = 0; i < nnodes; i++) {
      double sum = 0.0;
      for (int q = 0; q < nqpts; q++) {
        for (int d = 0; d < dim; d++) {
          sum += shape_fn_grads(q, i, d) * flux_q(q)[d];
        }
      }
      residual_e(i) = sum;
    }
  }

  #ifdef __CUDACC__
  __device__ void cuda_gradient(nd::view<grad_type> gradients_q, nd::view<const double, 1> values_e, nd::view<const double, 3> shape_fn_grads, double * /*buffer*/) const {
    int nnodes = num_nodes();
    int nqpts = gradients_q.shape[0];

    for (int j = threadIdx.x; j < nqpts; j += blockDim.x) {
      grad_type sum{};
      for (int i = 0; i < nnodes; i++) {
        sum[0] += values_e(i) * shape_fn_grads(j, i, 0);
        sum[1] += values_e(i) * shape_fn_grads(j, i, 1);
        sum[2] += values_e(i) * shape_fn_grads(j, i, 2);
      }
      gradients_q(j) = sum;
    }
  }

  __device__ void cuda_integrate_flux(nd::view<double> residual_e, nd::view<const flux_type> flux_q, nd::view<const double, 3> shape_fn_grads, double * /*buffer*/) const {
    int nnodes = num_nodes();
    int nqpts = flux_q.shape[0];

    for (int i = threadIdx.x; i < nnodes; i += blockDim.x) {
      double sum = 0.0;
      for (int q = 0; q < nqpts; q++) {
        for (int d = 0; d < dim; d++) {
          sum += shape_fn_grads(q, i, d) * flux_q(q)[d];
        }
      }
      residual_e(i) = sum;
    }
  }
  #endif

  uint32_t p;

};

} // namespace refactor
