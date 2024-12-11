#pragma once

#include "fm/types/vec.hpp"

namespace refactor {

using namespace fm;

// clang-format off
template <>
struct FiniteElement<Geometry::Quadrilateral, Family::Hcurl> {

  using value_type = vec2;
  using derivative_type = vec1;

  using source_type = vec2;
  using flux_type = vec1;

  static constexpr int dim = 2;

  __host__ __device__ uint32_t num_nodes() const { return 2 * p * (p + 1); }

  void nodes(nd::view<double,2> xi) const {
    if (p == 1) {
      xi(0, 0) = 0.5; xi(0, 1) = 0;
      xi(1, 0) = 0.5; xi(1, 1) = 1.0;
      xi(2, 0) = 0; xi(2, 1) = 0.5;
      xi(3, 0) = 1.0; xi(3, 1) = 0.5;
    }
    if (p == 2) {
      xi(0, 0) = 0.21132486540518711775; xi(0, 1) = 0;
      xi(1, 0) = 0.78867513459481288225; xi(1, 1) = 0;
      xi(2, 0) = 0.21132486540518711775; xi(2, 1) = 0.5;
      xi(3, 0) = 0.78867513459481288225; xi(3, 1) = 0.5;
      xi(4, 0) = 0.21132486540518711775; xi(4, 1) = 1.0;
      xi(5, 0) = 0.78867513459481288225; xi(5, 1) = 1.0;
      xi(6, 0) = 0; xi(6, 1) = 0.21132486540518711775;
      xi(7, 0) = 0; xi(7, 1) = 0.78867513459481288225;
      xi(8, 0) = 0.5; xi(8, 1) = 0.21132486540518711775;
      xi(9, 0) = 0.5; xi(9, 1) = 0.78867513459481288225;
      xi(10, 0) = 1.0; xi(10, 1) = 0.21132486540518711775;
      xi(11, 0) = 1.0; xi(11, 1) = 0.78867513459481288225;
    }
    if (p == 3) {
      xi(0, 0) = 0.11270166537925831148; xi(0, 1) = 0;
      xi(1, 0) = 0.5; xi(1, 1) = 0;
      xi(2, 0) = 0.88729833462074168852; xi(2, 1) = 0;
      xi(3, 0) = 0.11270166537925831148; xi(3, 1) = 0.27639320225002103036;
      xi(4, 0) = 0.5; xi(4, 1) = 0.27639320225002103036;
      xi(5, 0) = 0.88729833462074168852; xi(5, 1) = 0.27639320225002103036;
      xi(6, 0) = 0.11270166537925831148; xi(6, 1) = 0.72360679774997896964;
      xi(7, 0) = 0.5; xi(7, 1) = 0.72360679774997896964;
      xi(8, 0) = 0.88729833462074168852; xi(8, 1) = 0.72360679774997896964;
      xi(9, 0) = 0.11270166537925831148; xi(9, 1) = 1.0;
      xi(10, 0) = 0.5; xi(10, 1) = 1.0;
      xi(11, 0) = 0.88729833462074168852; xi(11, 1) = 1.0;
      xi(12, 0) = 0; xi(12, 1) = 0.11270166537925831148;
      xi(13, 0) = 0; xi(13, 1) = 0.5;
      xi(14, 0) = 0; xi(14, 1) = 0.88729833462074168852;
      xi(15, 0) = 0.27639320225002103036; xi(15, 1) = 0.11270166537925831148;
      xi(16, 0) = 0.27639320225002103036; xi(16, 1) = 0.5;
      xi(17, 0) = 0.27639320225002103036; xi(17, 1) = 0.88729833462074168852;
      xi(18, 0) = 0.72360679774997896964; xi(18, 1) = 0.11270166537925831148;
      xi(19, 0) = 0.72360679774997896964; xi(19, 1) = 0.5;
      xi(20, 0) = 0.72360679774997896964; xi(20, 1) = 0.88729833462074168852;
      xi(21, 0) = 1.0; xi(21, 1) = 0.11270166537925831148;
      xi(22, 0) = 1.0; xi(22, 1) = 0.5;
      xi(23, 0) = 1.0; xi(23, 1) = 0.88729833462074168852;
    }
  }

  void directions(nd::view<double,2> d) const {
    int i = 0;
    for (int k = 0; k < (p * (p + 1)); k++) { d(i, 0) =  1.0; d(i++, 1) =  0.0; }
    for (int k = 0; k < (p * (p + 1)); k++) { d(i, 0) =  0.0; d(i++, 1) =  1.0; }
  }

  __host__ __device__ uint32_t num_interior_nodes() const { return (p > 1) ? 2 * p * (p - 1) : 0; }

  void interior_nodes(nd::view<double,2> xi) const {
    if (p == 2) {
      xi(0, 0) = 0.21132486540518711775; xi(0, 1) = 0.5;
      xi(1, 0) = 0.78867513459481288225; xi(1, 1) = 0.5;
      xi(2, 0) = 0.5; xi(2, 1) = 0.21132486540518711775;
      xi(3, 0) = 0.5; xi(3, 1) = 0.78867513459481288225;
    }
    if (p == 3) {
      xi(0, 0) = 0.11270166537925831148; xi(0, 1) = 0.27639320225002103036;
      xi(1, 0) = 0.5; xi(1, 1) = 0.27639320225002103036;
      xi(2, 0) = 0.88729833462074168852; xi(2, 1) = 0.27639320225002103036;
      xi(3, 0) = 0.11270166537925831148; xi(3, 1) = 0.72360679774997896964;
      xi(4, 0) = 0.5; xi(4, 1) = 0.72360679774997896964;
      xi(5, 0) = 0.88729833462074168852; xi(5, 1) = 0.72360679774997896964;
      xi(6, 0) = 0.27639320225002103036; xi(6, 1) = 0.11270166537925831148;
      xi(7, 0) = 0.27639320225002103036; xi(7, 1) = 0.5;
      xi(8, 0) = 0.27639320225002103036; xi(8, 1) = 0.88729833462074168852;
      xi(9, 0) = 0.72360679774997896964; xi(9, 1) = 0.11270166537925831148;
      xi(10, 0) = 0.72360679774997896964; xi(10, 1) = 0.5;
      xi(11, 0) = 0.72360679774997896964; xi(11, 1) = 0.88729833462074168852;
    }
  }

  void interior_directions(nd::view<double,2> d) const {
    if (p > 1) {
      int i = 0;
      for (int k = 0; k < (p * (p - 1)); k++) { d(i, 0) =  1.0; d(i++, 1) =  0.0; }
      for (int k = 0; k < (p * (p - 1)); k++) { d(i, 0) =  0.0; d(i++, 1) =  1.0; }
    }
  }

  __host__ __device__ void indices(const GeometryInfo & offsets, const Connection * quad, uint32_t * indices) const {
    
    const Connection * edge = quad + Quadrilateral::edge_offset;
    const Connection cell = *(quad + Quadrilateral::cell_offset);

    if (p == 1) {

      // o-----→-----o         o-----1-----o
      // |           |         |           |
      // |           |         |           |
      // ↑           ↑         2           3
      // |           |         |           |
      // |           |         |           |
      // o-----→-----o         o-----0-----o
      indices[0] = offsets.edge + edge[0].index;
      indices[1] = offsets.edge + edge[2].index;
      indices[2] = offsets.edge + edge[3].index;
      indices[3] = offsets.edge + edge[1].index;
      return;

    }

    if (p == 2) {

      // o---→---→---o         o---4---5---o
      // |           |         |           |
      // ↑     ↑     ↑         7     9    11
      // |   →   →   |         |   2   3   |
      // ↑     ↑     ↑         6     8    10
      // |           |         |           |
      // o---→---→---o         o---0---1---o
      indices[0] = offsets.edge + 2 * edge[0].index + 0;
      indices[1] = offsets.edge + 2 * edge[0].index + 1;
      indices[2] = offsets.quad + 4 * cell.index + 0;
      indices[3] = offsets.quad + 4 * cell.index + 1;
      indices[4] = offsets.edge + 2 * edge[2].index + 1;
      indices[5] = offsets.edge + 2 * edge[2].index + 0;

      indices[ 6] = offsets.edge + 2 * edge[3].index + 1;
      indices[ 7] = offsets.edge + 2 * edge[3].index + 0;
      indices[ 8] = offsets.quad + 4 * cell.index + 2;
      indices[ 9] = offsets.quad + 4 * cell.index + 3;
      indices[10] = offsets.edge + 2 * edge[1].index + 0;
      indices[11] = offsets.edge + 2 * edge[1].index + 1;

      if (flip(edge[0])) { fm::swap(indices[ 0], indices[ 1]); }
      if (flip(edge[1])) { fm::swap(indices[10], indices[11]); }
      if (flip(edge[2])) { fm::swap(indices[ 4], indices[ 5]); }
      if (flip(edge[3])) { fm::swap(indices[ 6], indices[ 7]); }

    }

    if (p == 3) {

      // o--→--→--→--o         o--9-10-11--o
      // ↑   ↑   ↑   ↑         14  17 20  23
      // |  →  →  →  |         |  6  7  8  |
      // ↑   ↑   ↑   ↑         13  16 19  22
      // |  →  →  →  |         |  3  4  5  |
      // ↑   ↑   ↑   ↑         12  15 18  21
      // o--→--→--→--o         o--0--1--2--o
      indices[ 0] = offsets.edge + 3 * edge[0].index + 0;
      indices[ 1] = offsets.edge + 3 * edge[0].index + 1;
      indices[ 2] = offsets.edge + 3 * edge[0].index + 2;
      indices[ 3] = offsets.quad + 12 * cell.index + 0;
      indices[ 4] = offsets.quad + 12 * cell.index + 1;
      indices[ 5] = offsets.quad + 12 * cell.index + 2;
      indices[ 6] = offsets.quad + 12 * cell.index + 3;
      indices[ 7] = offsets.quad + 12 * cell.index + 4;
      indices[ 8] = offsets.quad + 12 * cell.index + 5;
      indices[ 9] = offsets.edge + 3 * edge[2].index + 2;
      indices[10] = offsets.edge + 3 * edge[2].index + 1;
      indices[11] = offsets.edge + 3 * edge[2].index + 0;

      indices[12] = offsets.edge + 3 * edge[3].index + 2;
      indices[13] = offsets.edge + 3 * edge[3].index + 1;
      indices[14] = offsets.edge + 3 * edge[3].index + 0;
      indices[15] = offsets.quad + 12 * cell.index +  6;
      indices[16] = offsets.quad + 12 * cell.index +  7;
      indices[17] = offsets.quad + 12 * cell.index +  8;
      indices[18] = offsets.quad + 12 * cell.index +  9;
      indices[19] = offsets.quad + 12 * cell.index + 10;
      indices[20] = offsets.quad + 12 * cell.index + 11;
      indices[21] = offsets.edge + 3 * edge[1].index + 0;
      indices[22] = offsets.edge + 3 * edge[1].index + 1;
      indices[23] = offsets.edge + 3 * edge[1].index + 2;

      if (flip(edge[0])) { fm::swap(indices[ 0], indices[ 2]); }
      if (flip(edge[1])) { fm::swap(indices[21], indices[23]); }
      if (flip(edge[2])) { fm::swap(indices[ 9], indices[11]); }
      if (flip(edge[3])) { fm::swap(indices[12], indices[14]); }

    }

  }

  template < typename T >
  __host__ __device__ void reorient(const TransformationType type, const Connection * quad, T * values) const {

    const Connection * edge = quad + Quadrilateral::edge_offset;

    if (p == 1) {
      if (edge[0].sign() == Sign::Negative) { values[0] *= -1; }
      if (edge[1].sign() == Sign::Negative) { values[3] *= -1; }
      if (edge[2].sign() == Sign::Positive) { values[1] *= -1; }
      if (edge[3].sign() == Sign::Positive) { values[2] *= -1; }
      return;
    }

    if (p == 2) {
      if (edge[0].sign() == Sign::Negative) { values[ 0] *= -1; values[ 1] *= -1; }
      if (edge[1].sign() == Sign::Negative) { values[10] *= -1; values[11] *= -1; }
      if (edge[2].sign() == Sign::Positive) { values[ 4] *= -1; values[ 5] *= -1; }
      if (edge[3].sign() == Sign::Positive) { values[ 6] *= -1; values[ 7] *= -1; }
      return;
    }

    if (p == 3) {
      if (edge[0].sign() == Sign::Negative) { values[ 0] *= -1; values[ 1] *= -1; values[ 2] *= -1;}
      if (edge[1].sign() == Sign::Negative) { values[21] *= -1; values[22] *= -1; values[23] *= -1;}
      if (edge[2].sign() == Sign::Positive) { values[ 9] *= -1; values[10] *= -1; values[11] *= -1;}
      if (edge[3].sign() == Sign::Positive) { values[12] *= -1; values[13] *= -1; values[14] *= -1;}
      return;
    }

  }

  __host__ __device__ void reorient(const TransformationType type, const Connection * quad, int8_t * transformation) {

    const Connection * edge = quad + Quadrilateral::edge_offset;

    uint32_t nnodes = num_nodes();
    for (int k = 0; k < nnodes; k++) {
      transformation[k] = 0;
    }

    if (p == 1) {
      if (edge[0].sign() == Sign::Negative) { transformation[0] = -1; }
      if (edge[1].sign() == Sign::Negative) { transformation[3] = -1; }
      if (edge[2].sign() == Sign::Positive) { transformation[1] = -1; }
      if (edge[3].sign() == Sign::Positive) { transformation[2] = -1; }
      return;
    }

    if (p == 2) {
      if (edge[0].sign() == Sign::Negative) { transformation[ 0] = -1; transformation[ 1] = -1; }
      if (edge[1].sign() == Sign::Negative) { transformation[10] = -1; transformation[11] = -1; }
      if (edge[2].sign() == Sign::Positive) { transformation[ 4] = -1; transformation[ 5] = -1; }
      if (edge[3].sign() == Sign::Positive) { transformation[ 6] = -1; transformation[ 7] = -1; }
      return;
    }

    if (p == 3) {
      if (edge[0].sign() == Sign::Negative) { transformation[ 0] = -1; transformation[ 1] = -1; transformation[ 2] = -1;}
      if (edge[1].sign() == Sign::Negative) { transformation[21] = -1; transformation[22] = -1; transformation[23] = -1;}
      if (edge[2].sign() == Sign::Positive) { transformation[ 9] = -1; transformation[10] = -1; transformation[11] = -1;}
      if (edge[3].sign() == Sign::Positive) { transformation[12] = -1; transformation[13] = -1; transformation[14] = -1;}
      return;
    }

  }

  constexpr vec2 shape_function(vec2 xi, uint32_t i) const {
    if (p == 1) {
      if (i == 0) { return vec2{1 - xi[1],0}; }
      if (i == 1) { return vec2{xi[1],0}; }
      if (i == 2) { return vec2{0,1 - xi[0]}; }
      if (i == 3) { return vec2{0,xi[0]}; }
    }
    if (p == 2) {
      if (i == 0) { return vec2{-1.7320508075688772935274463415*(-0.7886751345948128822545743902 + xi[0])*(-1 + xi[1])*(-1 + 2*xi[1]),0}; }
      if (i == 1) { return vec2{(-0.3660254037844386467637231708 + 1.7320508075688772935274463415*xi[0])*(-1 + xi[1])*(-1 + 2*xi[1]),0}; }
      if (i == 2) { return vec2{6.928203230275509174109785366*(-0.7886751345948128822545743902 + xi[0])*(-1. + xi[1])*xi[1],0}; }
      if (i == 3) { return vec2{-6.92820323027550917410978537*(-0.21132486540518711774542561 + 1.*xi[0])*(-1. + xi[1])*xi[1],0}; }
      if (i == 4) { return vec2{-1.7320508075688772935274463415*(-0.7886751345948128822545743902 + xi[0])*xi[1]*(-1 + 2*xi[1]),0}; }
      if (i == 5) { return vec2{(-0.3660254037844386467637231708 + 1.7320508075688772935274463415*xi[0])*xi[1]*(-1 + 2*xi[1]),0}; }
      if (i == 6) { return vec2{0,-1.7320508075688772935274463415*(-1 + xi[0])*(-1 + 2*xi[0])*(-0.7886751345948128822545743902 + xi[1])}; }
      if (i == 7) { return vec2{0,(-1 + xi[0])*(-1 + 2*xi[0])*(-0.3660254037844386467637231708 + 1.7320508075688772935274463415*xi[1])}; }
      if (i == 8) { return vec2{0,6.928203230275509174109785366*(-1. + xi[0])*xi[0]*(-0.7886751345948128822545743902 + xi[1])}; }
      if (i == 9) { return vec2{0,-6.92820323027550917410978537*(-1. + xi[0])*xi[0]*(-0.21132486540518711774542561 + 1.*xi[1])}; }
      if (i == 10) { return vec2{0,-1.7320508075688772935274463415*xi[0]*(-1 + 2*xi[0])*(-0.7886751345948128822545743902 + xi[1])}; }
      if (i == 11) { return vec2{0,xi[0]*(-1 + 2*xi[0])*(-0.3660254037844386467637231708 + 1.7320508075688772935274463415*xi[1])}; }
    }
    if (p == 3) {
      if (i == 0) { return vec2{(1.4788305577012361475298776 + xi[0]*(-4.6243277820691389617264218 + 3.3333333333333333333333333*xi[0]))*(1 + xi[1]*(-6. + (10. - 5.*xi[1])*xi[1])),0}; }
      if (i == 1) { return vec2{-6.66666666666666666666666667*(-0.88729833462074168851792654 + xi[0])*(-0.11270166537925831148207346 + xi[0])*(1 + xi[1]*(-6. + (10. - 5.*xi[1])*xi[1])),0}; }
      if (i == 2) { return vec2{(0.1878361089654305191367891 + xi[0]*(-2.0423388845975277049402449 + 3.3333333333333333333333333*xi[0]))*(1 + xi[1]*(-6. + (10. - 5.*xi[1])*xi[1])),0}; }
      if (i == 3) { return vec2{11.180339887498948*(1.4788305577012361475298776 + xi[0]*(-4.6243277820691389617264218 + 3.3333333333333333333333333*xi[0]))*(-1. + xi[1])*xi[1]*(-0.723606797749979 + 1.*xi[1]),0}; }
      if (i == 4) { return vec2{-74.53559924999299*(-0.88729833462074168851792654 + xi[0])*(-0.11270166537925831148207346 + xi[0])*(-1. + xi[1])*xi[1]*(-0.723606797749979 + 1.*xi[1]),0}; }
      if (i == 5) { return vec2{11.180339887498948*(0.1878361089654305191367891 + xi[0]*(-2.0423388845975277049402449 + 3.3333333333333333333333333*xi[0]))*(-1. + xi[1])*xi[1]*(-0.723606797749979 + 1.*xi[1]),0}; }
      if (i == 6) { return vec2{-11.180339887498948482*(1.4788305577012361475298776 + xi[0]*(-4.6243277820691389617264218 + 3.3333333333333333333333333*xi[0]))*(-1. + xi[1])*(-0.2763932022500210304 + xi[1])*xi[1],0}; }
      if (i == 7) { return vec2{74.53559924999298988*(-0.88729833462074168851792654 + xi[0])*(-0.11270166537925831148207346 + xi[0])*(-1. + xi[1])*(-0.2763932022500210304 + xi[1])*xi[1],0}; }
      if (i == 8) { return vec2{-11.180339887498948482*(0.1878361089654305191367891 + xi[0]*(-2.0423388845975277049402449 + 3.3333333333333333333333333*xi[0]))*(-1. + xi[1])*(-0.2763932022500210304 + xi[1])*xi[1],0}; }
      if (i == 9) { return vec2{(1.4788305577012361475298776 + xi[0]*(-4.6243277820691389617264218 + 3.3333333333333333333333333*xi[0]))*xi[1]*(1. + xi[1]*(-5. + 5.*xi[1])),0}; }
      if (i == 10) { return vec2{-6.66666666666666666666666667*(-0.88729833462074168851792654 + xi[0])*(-0.11270166537925831148207346 + xi[0])*xi[1]*(1. + xi[1]*(-5. + 5.*xi[1])),0}; }
      if (i == 11) { return vec2{(0.1878361089654305191367891 + xi[0]*(-2.0423388845975277049402449 + 3.3333333333333333333333333*xi[0]))*xi[1]*(1. + xi[1]*(-5. + 5.*xi[1])),0}; }
      if (i == 12) { return vec2{0,(1 + xi[0]*(-6. + (10. - 5.*xi[0])*xi[0]))*(1.4788305577012361475298776 + xi[1]*(-4.6243277820691389617264218 + 3.3333333333333333333333333*xi[1]))}; }
      if (i == 13) { return vec2{0,-6.66666666666666666666666667*(1 + xi[0]*(-6. + (10. - 5.*xi[0])*xi[0]))*(-0.88729833462074168851792654 + xi[1])*(-0.11270166537925831148207346 + xi[1])}; }
      if (i == 14) { return vec2{0,(1 + xi[0]*(-6. + (10. - 5.*xi[0])*xi[0]))*(0.1878361089654305191367891 + xi[1]*(-2.0423388845975277049402449 + 3.3333333333333333333333333*xi[1]))}; }
      if (i == 15) { return vec2{0,11.180339887498948*(-1. + xi[0])*xi[0]*(-0.723606797749979 + 1.*xi[0])*(1.4788305577012361475298776 + xi[1]*(-4.6243277820691389617264218 + 3.3333333333333333333333333*xi[1]))}; }
      if (i == 16) { return vec2{0,-74.53559924999299*(-1. + xi[0])*xi[0]*(-0.723606797749979 + 1.*xi[0])*(-0.88729833462074168851792654 + xi[1])*(-0.11270166537925831148207346 + xi[1])}; }
      if (i == 17) { return vec2{0,11.180339887498948*(-1. + xi[0])*xi[0]*(-0.723606797749979 + 1.*xi[0])*(0.1878361089654305191367891 + xi[1]*(-2.0423388845975277049402449 + 3.3333333333333333333333333*xi[1]))}; }
      if (i == 18) { return vec2{0,-11.180339887498948482*(-1. + xi[0])*(-0.2763932022500210304 + xi[0])*xi[0]*(1.4788305577012361475298776 + xi[1]*(-4.6243277820691389617264218 + 3.3333333333333333333333333*xi[1]))}; }
      if (i == 19) { return vec2{0,74.53559924999298988*(-1. + xi[0])*(-0.2763932022500210304 + xi[0])*xi[0]*(-0.88729833462074168851792654 + xi[1])*(-0.11270166537925831148207346 + xi[1])}; }
      if (i == 20) { return vec2{0,-11.180339887498948482*(-1. + xi[0])*(-0.2763932022500210304 + xi[0])*xi[0]*(0.1878361089654305191367891 + xi[1]*(-2.0423388845975277049402449 + 3.3333333333333333333333333*xi[1]))}; }
      if (i == 21) { return vec2{0,xi[0]*(1. + xi[0]*(-5. + 5.*xi[0]))*(1.4788305577012361475298776 + xi[1]*(-4.6243277820691389617264218 + 3.3333333333333333333333333*xi[1]))}; }
      if (i == 22) { return vec2{0,-6.66666666666666666666666667*xi[0]*(1. + xi[0]*(-5. + 5.*xi[0]))*(-0.88729833462074168851792654 + xi[1])*(-0.11270166537925831148207346 + xi[1])}; }
      if (i == 23) { return vec2{0,xi[0]*(1. + xi[0]*(-5. + 5.*xi[0]))*(0.1878361089654305191367891 + xi[1]*(-2.0423388845975277049402449 + 3.3333333333333333333333333*xi[1]))}; }
    }

    return {};
  }

  constexpr vec2 reoriented_shape_function(vec2 xi, uint32_t i, int8_t transformation) const {
    if (transformation == -1) {
      return -shape_function(xi, i);
    } else {
      return  shape_function(xi, i);
    }
  }

  vec<1> shape_function_curl(vec2 xi, uint32_t i) const {
    // expressions generated symbolically by mathematica
    if (p == 1) {
      if (i == 0) { return 1; }
      if (i == 1) { return -1; }
      if (i == 2) { return -1; }
      if (i == 3) { return 1; }
    }
    if (p == 2) {
      if (i == 0) { return 3.464101615137754587054892683*(-0.7886751345948128822545743902 + xi[0])*(-1 + xi[1]) + 1.7320508075688772935274463415*(-0.7886751345948128822545743902 + xi[0])*(-1 + 2*xi[1]); }
      if (i == 1) { return -2*(-0.3660254037844386467637231708 + 1.7320508075688772935274463415*xi[0])*(-1 + xi[1]) - (-0.3660254037844386467637231708 + 1.7320508075688772935274463415*xi[0])*(-1 + 2*xi[1]); }
      if (i == 2) { return -6.928203230275509174109785366*(-0.7886751345948128822545743902 + xi[0])*(-1. + xi[1]) - 6.928203230275509174109785366*(-0.7886751345948128822545743902 + xi[0])*xi[1]; }
      if (i == 3) { return 6.92820323027550917410978537*(-0.21132486540518711774542561 + 1.*xi[0])*(-1. + xi[1]) + 6.92820323027550917410978537*(-0.21132486540518711774542561 + 1.*xi[0])*xi[1]; }
      if (i == 4) { return 3.464101615137754587054892683*(-0.7886751345948128822545743902 + xi[0])*xi[1] + 1.7320508075688772935274463415*(-0.7886751345948128822545743902 + xi[0])*(-1 + 2*xi[1]); }
      if (i == 5) { return -2*(-0.3660254037844386467637231708 + 1.7320508075688772935274463415*xi[0])*xi[1] - (-0.3660254037844386467637231708 + 1.7320508075688772935274463415*xi[0])*(-1 + 2*xi[1]); }
      if (i == 6) { return -3.464101615137754587054892683*(-1 + xi[0])*(-0.7886751345948128822545743902 + xi[1]) - 1.7320508075688772935274463415*(-1 + 2*xi[0])*(-0.7886751345948128822545743902 + xi[1]); }
      if (i == 7) { return 2*(-1 + xi[0])*(-0.3660254037844386467637231708 + 1.7320508075688772935274463415*xi[1]) + (-1 + 2*xi[0])*(-0.3660254037844386467637231708 + 1.7320508075688772935274463415*xi[1]); }
      if (i == 8) { return 6.928203230275509174109785366*(-1. + xi[0])*(-0.7886751345948128822545743902 + xi[1]) + 6.928203230275509174109785366*xi[0]*(-0.7886751345948128822545743902 + xi[1]); }
      if (i == 9) { return -6.92820323027550917410978537*(-1. + xi[0])*(-0.21132486540518711774542561 + 1.*xi[1]) - 6.92820323027550917410978537*xi[0]*(-0.21132486540518711774542561 + 1.*xi[1]); }
      if (i == 10) { return -3.464101615137754587054892683*xi[0]*(-0.7886751345948128822545743902 + xi[1]) - 1.7320508075688772935274463415*(-1 + 2*xi[0])*(-0.7886751345948128822545743902 + xi[1]); }
      if (i == 11) { return 2*xi[0]*(-0.3660254037844386467637231708 + 1.7320508075688772935274463415*xi[1]) + (-1 + 2*xi[0])*(-0.3660254037844386467637231708 + 1.7320508075688772935274463415*xi[1]); }
    }
    if (p == 3) {
      if (i == 0) { return -((1.4788305577012361475298776 + xi[0]*(-4.6243277820691389617264218 + 3.3333333333333333333333333*xi[0]))*(-6. + (10. - 10.*xi[1])*xi[1] + (10. - 5.*xi[1])*xi[1])); }
      if (i == 1) { return 6.66666666666666666666666667*(-0.88729833462074168851792654 + xi[0])*(-0.11270166537925831148207346 + xi[0])*(-6. + (10. - 10.*xi[1])*xi[1] + (10. - 5.*xi[1])*xi[1]); }
      if (i == 2) { return -((0.1878361089654305191367891 + xi[0]*(-2.0423388845975277049402449 + 3.3333333333333333333333333*xi[0]))*(-6. + (10. - 10.*xi[1])*xi[1] + (10. - 5.*xi[1])*xi[1])); }
      if (i == 3) { return -11.180339887498948*(1.4788305577012361475298776 + xi[0]*(-4.6243277820691389617264218 + 3.3333333333333333333333333*xi[0]))*(-1. + xi[1])*xi[1] - 11.180339887498948*(1.4788305577012361475298776 + xi[0]*(-4.6243277820691389617264218 + 3.3333333333333333333333333*xi[0]))*(-1. + xi[1])*(-0.723606797749979 + 1.*xi[1]) - 11.180339887498948*(1.4788305577012361475298776 + xi[0]*(-4.6243277820691389617264218 + 3.3333333333333333333333333*xi[0]))*xi[1]*(-0.723606797749979 + 1.*xi[1]); }
      if (i == 4) { return 74.53559924999299*(-0.88729833462074168851792654 + xi[0])*(-0.11270166537925831148207346 + xi[0])*(-1. + xi[1])*xi[1] + 74.53559924999299*(-0.88729833462074168851792654 + xi[0])*(-0.11270166537925831148207346 + xi[0])*(-1. + xi[1])*(-0.723606797749979 + 1.*xi[1]) + 74.53559924999299*(-0.88729833462074168851792654 + xi[0])*(-0.11270166537925831148207346 + xi[0])*xi[1]*(-0.723606797749979 + 1.*xi[1]); }
      if (i == 5) { return -11.180339887498948*(0.1878361089654305191367891 + xi[0]*(-2.0423388845975277049402449 + 3.3333333333333333333333333*xi[0]))*(-1. + xi[1])*xi[1] - 11.180339887498948*(0.1878361089654305191367891 + xi[0]*(-2.0423388845975277049402449 + 3.3333333333333333333333333*xi[0]))*(-1. + xi[1])*(-0.723606797749979 + 1.*xi[1]) - 11.180339887498948*(0.1878361089654305191367891 + xi[0]*(-2.0423388845975277049402449 + 3.3333333333333333333333333*xi[0]))*xi[1]*(-0.723606797749979 + 1.*xi[1]); }
      if (i == 6) { return 11.180339887498948482*(1.4788305577012361475298776 + xi[0]*(-4.6243277820691389617264218 + 3.3333333333333333333333333*xi[0]))*(-1. + xi[1])*(-0.2763932022500210304 + xi[1]) + 11.180339887498948482*(1.4788305577012361475298776 + xi[0]*(-4.6243277820691389617264218 + 3.3333333333333333333333333*xi[0]))*(-1. + xi[1])*xi[1] + 11.180339887498948482*(1.4788305577012361475298776 + xi[0]*(-4.6243277820691389617264218 + 3.3333333333333333333333333*xi[0]))*(-0.2763932022500210304 + xi[1])*xi[1]; }
      if (i == 7) { return -74.53559924999298988*(-0.88729833462074168851792654 + xi[0])*(-0.11270166537925831148207346 + xi[0])*(-1. + xi[1])*(-0.2763932022500210304 + xi[1]) - 74.53559924999298988*(-0.88729833462074168851792654 + xi[0])*(-0.11270166537925831148207346 + xi[0])*(-1. + xi[1])*xi[1] - 74.53559924999298988*(-0.88729833462074168851792654 + xi[0])*(-0.11270166537925831148207346 + xi[0])*(-0.2763932022500210304 + xi[1])*xi[1]; }
      if (i == 8) { return 11.180339887498948482*(0.1878361089654305191367891 + xi[0]*(-2.0423388845975277049402449 + 3.3333333333333333333333333*xi[0]))*(-1. + xi[1])*(-0.2763932022500210304 + xi[1]) + 11.180339887498948482*(0.1878361089654305191367891 + xi[0]*(-2.0423388845975277049402449 + 3.3333333333333333333333333*xi[0]))*(-1. + xi[1])*xi[1] + 11.180339887498948482*(0.1878361089654305191367891 + xi[0]*(-2.0423388845975277049402449 + 3.3333333333333333333333333*xi[0]))*(-0.2763932022500210304 + xi[1])*xi[1]; }
      if (i == 9) { return -((1.4788305577012361475298776 + xi[0]*(-4.6243277820691389617264218 + 3.3333333333333333333333333*xi[0]))*xi[1]*(-5. + 10.*xi[1])) - (1.4788305577012361475298776 + xi[0]*(-4.6243277820691389617264218 + 3.3333333333333333333333333*xi[0]))*(1. + xi[1]*(-5. + 5.*xi[1])); }
      if (i == 10) { return 6.66666666666666666666666667*(-0.88729833462074168851792654 + xi[0])*(-0.11270166537925831148207346 + xi[0])*xi[1]*(-5. + 10.*xi[1]) + 6.66666666666666666666666667*(-0.88729833462074168851792654 + xi[0])*(-0.11270166537925831148207346 + xi[0])*(1. + xi[1]*(-5. + 5.*xi[1])); }
      if (i == 11) { return -((0.1878361089654305191367891 + xi[0]*(-2.0423388845975277049402449 + 3.3333333333333333333333333*xi[0]))*xi[1]*(-5. + 10.*xi[1])) - (0.1878361089654305191367891 + xi[0]*(-2.0423388845975277049402449 + 3.3333333333333333333333333*xi[0]))*(1. + xi[1]*(-5. + 5.*xi[1])); }
      if (i == 12) { return (-6. + (10. - 10.*xi[0])*xi[0] + (10. - 5.*xi[0])*xi[0])*(1.4788305577012361475298776 + xi[1]*(-4.6243277820691389617264218 + 3.3333333333333333333333333*xi[1])); }
      if (i == 13) { return -6.66666666666666666666666667*(-6. + (10. - 10.*xi[0])*xi[0] + (10. - 5.*xi[0])*xi[0])*(-0.88729833462074168851792654 + xi[1])*(-0.11270166537925831148207346 + xi[1]); }
      if (i == 14) { return (-6. + (10. - 10.*xi[0])*xi[0] + (10. - 5.*xi[0])*xi[0])*(0.1878361089654305191367891 + xi[1]*(-2.0423388845975277049402449 + 3.3333333333333333333333333*xi[1])); }
      if (i == 15) { return 11.180339887498948*(-1. + xi[0])*xi[0]*(1.4788305577012361475298776 + xi[1]*(-4.6243277820691389617264218 + 3.3333333333333333333333333*xi[1])) + 11.180339887498948*(-1. + xi[0])*(-0.723606797749979 + 1.*xi[0])*(1.4788305577012361475298776 + xi[1]*(-4.6243277820691389617264218 + 3.3333333333333333333333333*xi[1])) + 11.180339887498948*xi[0]*(-0.723606797749979 + 1.*xi[0])*(1.4788305577012361475298776 + xi[1]*(-4.6243277820691389617264218 + 3.3333333333333333333333333*xi[1])); }
      if (i == 16) { return -74.53559924999299*(-1. + xi[0])*xi[0]*(-0.88729833462074168851792654 + xi[1])*(-0.11270166537925831148207346 + xi[1]) - 74.53559924999299*(-1. + xi[0])*(-0.723606797749979 + 1.*xi[0])*(-0.88729833462074168851792654 + xi[1])*(-0.11270166537925831148207346 + xi[1]) - 74.53559924999299*xi[0]*(-0.723606797749979 + 1.*xi[0])*(-0.88729833462074168851792654 + xi[1])*(-0.11270166537925831148207346 + xi[1]); }
      if (i == 17) { return 11.180339887498948*(-1. + xi[0])*xi[0]*(0.1878361089654305191367891 + xi[1]*(-2.0423388845975277049402449 + 3.3333333333333333333333333*xi[1])) + 11.180339887498948*(-1. + xi[0])*(-0.723606797749979 + 1.*xi[0])*(0.1878361089654305191367891 + xi[1]*(-2.0423388845975277049402449 + 3.3333333333333333333333333*xi[1])) + 11.180339887498948*xi[0]*(-0.723606797749979 + 1.*xi[0])*(0.1878361089654305191367891 + xi[1]*(-2.0423388845975277049402449 + 3.3333333333333333333333333*xi[1])); }
      if (i == 18) { return -11.180339887498948482*(-1. + xi[0])*(-0.2763932022500210304 + xi[0])*(1.4788305577012361475298776 + xi[1]*(-4.6243277820691389617264218 + 3.3333333333333333333333333*xi[1])) - 11.180339887498948482*(-1. + xi[0])*xi[0]*(1.4788305577012361475298776 + xi[1]*(-4.6243277820691389617264218 + 3.3333333333333333333333333*xi[1])) - 11.180339887498948482*(-0.2763932022500210304 + xi[0])*xi[0]*(1.4788305577012361475298776 + xi[1]*(-4.6243277820691389617264218 + 3.3333333333333333333333333*xi[1])); }
      if (i == 19) { return 74.53559924999298988*(-1. + xi[0])*(-0.2763932022500210304 + xi[0])*(-0.88729833462074168851792654 + xi[1])*(-0.11270166537925831148207346 + xi[1]) + 74.53559924999298988*(-1. + xi[0])*xi[0]*(-0.88729833462074168851792654 + xi[1])*(-0.11270166537925831148207346 + xi[1]) + 74.53559924999298988*(-0.2763932022500210304 + xi[0])*xi[0]*(-0.88729833462074168851792654 + xi[1])*(-0.11270166537925831148207346 + xi[1]); }
      if (i == 20) { return -11.180339887498948482*(-1. + xi[0])*(-0.2763932022500210304 + xi[0])*(0.1878361089654305191367891 + xi[1]*(-2.0423388845975277049402449 + 3.3333333333333333333333333*xi[1])) - 11.180339887498948482*(-1. + xi[0])*xi[0]*(0.1878361089654305191367891 + xi[1]*(-2.0423388845975277049402449 + 3.3333333333333333333333333*xi[1])) - 11.180339887498948482*(-0.2763932022500210304 + xi[0])*xi[0]*(0.1878361089654305191367891 + xi[1]*(-2.0423388845975277049402449 + 3.3333333333333333333333333*xi[1])); }
      if (i == 21) { return xi[0]*(-5. + 10.*xi[0])*(1.4788305577012361475298776 + xi[1]*(-4.6243277820691389617264218 + 3.3333333333333333333333333*xi[1])) + (1. + xi[0]*(-5. + 5.*xi[0]))*(1.4788305577012361475298776 + xi[1]*(-4.6243277820691389617264218 + 3.3333333333333333333333333*xi[1])); }
      if (i == 22) { return -6.66666666666666666666666667*xi[0]*(-5. + 10.*xi[0])*(-0.88729833462074168851792654 + xi[1])*(-0.11270166537925831148207346 + xi[1]) - 6.66666666666666666666666667*(1. + xi[0]*(-5. + 5.*xi[0]))*(-0.88729833462074168851792654 + xi[1])*(-0.11270166537925831148207346 + xi[1]); }
      if (i == 23) { return xi[0]*(-5. + 10.*xi[0])*(0.1878361089654305191367891 + xi[1]*(-2.0423388845975277049402449 + 3.3333333333333333333333333*xi[1])) + (1. + xi[0]*(-5. + 5.*xi[0]))*(0.1878361089654305191367891 + xi[1]*(-2.0423388845975277049402449 + 3.3333333333333333333333333*xi[1])); }
    }

    return {};
  }

  vec1 reoriented_shape_function_curl(vec2 xi, uint32_t i, int8_t transformation) const {
    if (transformation == -1) {
      return -shape_function_curl(xi, i);
    } else {
      return  shape_function_curl(xi, i);
    }
  }

  vec<1> shape_function_derivative(vec2 xi, uint32_t i) const {
    return shape_function_curl(xi, i);
  }

  vec2 interpolate(vec2 xi, const double * values) const {
    vec2 interpolated_value{};
    for (int i = 0; i < num_nodes(); i++) {
      interpolated_value += values[i] * shape_function(xi, i);
    }
    return interpolated_value;
  }

  vec<1> curl(vec2 xi, const double * values) const {
    double interpolated_curl = 0.0;
    for (int i = 0; i < num_nodes(); i++) {
      interpolated_curl += values[i] * shape_function_curl(xi, i);
    }
    return interpolated_curl;
  }
       
  // TODO: this is set to nonzero for the batched interpolation, 
  //       even though batched curl doesn't use the buffer
  __host__ __device__ uint32_t batch_interpolation_scratch_space(nd::view<const double,2> xi) const {
    uint32_t q = xi.shape[0];
    return q * (p + 1);
  }

  nd::array< double > evaluate_shape_functions(nd::view<const double,2> xi) const {
    uint32_t q = xi.shape[0];
    uint32_t num_entries = 0;
    num_entries += q * p;       // B1
    num_entries += q * (p + 1); // B2
    nd::array<double> buffer({num_entries});
    for (int i = 0; i < q; i++) {
      GaussLegendreInterpolation(xi(i, 0), p, &buffer(p*i));
      GaussLobattoInterpolation(xi(i, 0), p+1, &buffer((p+1)*i + (p*q)));
    }
    return buffer;
  }

  void interpolate(nd::view<value_type> values_q, nd::view<const double> values_e, nd::view<double> shape_fn, double * buffer) const {
    uint32_t n = p + 1;
    uint32_t q = sqrt(values_q.shape[0]);

    // 1D shape function evaluations
    nd::view<double, 2> B1(shape_fn.data(), {q, p}); // legendre shape functions
    nd::view<double, 2> B2(B1.end(),        {q, n}); // lobatto  shape functions

    nd::view<double, 2> A1(buffer, {q, n}); // storage for intermediates

    nd::view<const double, 2> ue(values_e.data(), {n, p});
    nd::view<double, 2> uq((double*)values_q.data(), {q, q}, {2*q, 2});
    
    _contract(A1, B1, ue); //  A1(qx, iy) = sum_{ix} B1(qx, ix) * ue(iy, ix)  
    _contract(uq, B2, A1); //  uq(qy, qx) = sum_{iy} B2(qy, iy) * A1(qx, iy)

    ue = nd::view< const double, 2 >(values_e.data() + (n * p), {n, p});

    // note: column-major strides here, since quadrature points are still 
    // enumerated lexicographically as {y, x} but y-component nodes are {x, y}
    uq = nd::view<double, 2>(((double*)values_q.data())+1, {q, q}, {2, 2*q});
    
    _contract(A1, B1, ue); //  A1(qx, iy) = sum_{ix} B1(qx, ix) * ue(iy, ix)  
    _contract(uq, B2, A1); //  uq(qy, qx) = sum_{iy} B2(qy, iy) * A1(qx, iy)
  }

#if 1
  nd::array< double, 2 > evaluate_shape_function_curls(nd::view<const double, 2> xi) const {
    uint32_t q = xi.shape[0];
    nd::array<double, 2> shape_fns({q * q, num_nodes()});
    for (int i = 0; i < q; i++) {
      for (int j = 0; j < q; j++) {
        vec2 xi_ij = vec2{xi(j, 0), xi(i, 0)};
        for (int k = 0; k < num_nodes(); k++) {
          shape_fns(i * q + j, k) = shape_function_curl(xi_ij, k);
        }
      }
    }
    return shape_fns;
  }

  void curl(nd::view<derivative_type> values_q, nd::view<const double, 1> values_e, nd::view<const double, 2> shape_fn_curls, double * /*buffer*/) const {
    int nnodes = num_nodes();
    int nqpts = values_q.shape[0];

    for (int q = 0; q < nqpts; q++) {
      derivative_type sum = 0.0;
      for (int i = 0; i < nnodes; i++) {
        sum[0] += shape_fn_curls(q, i) * values_e(i);
      }
      values_q(q) = sum;
    }
  }

  nd::array< double, 3 > evaluate_weighted_shape_functions(nd::view<const double, 2> xi, nd::view<const double, 1> weights) {
    uint32_t q = xi.shape[0];
    nd::array<double, 3> shape_fns({q * q, num_nodes(), dim});
    uint32_t qcount = 0;
    for (int j = 0; j < q; j++) {
      for (int i = 0; i < q; i++) {
        vec2 xi_ij = vec2{xi(i, 0), xi(j, 0)};
        for (int l = 0; l < num_nodes(); l++) {
          vec2 phi = shape_function(xi_ij, l);
          shape_fns(qcount, l, 0) = phi[0] * weights[i] * weights[j];
          shape_fns(qcount, l, 1) = phi[1] * weights[i] * weights[j];
        }
        qcount++;
      }
    }
    return shape_fns;
  }

  __host__ __device__ void integrate_source(nd::view<double> residual_e, nd::view<const source_type> source_q, nd::view<const double, 3> shape_fn, double * /*buffer*/) const {
    int nnodes = num_nodes();
    int nqpts = source_q.shape[0];

    for (int i = 0; i < nnodes; i++) {
      double sum = 0.0;
      for (int q = 0; q < nqpts; q++) {
        for (int j = 0; j < dim; j++) {
          sum += shape_fn(q, i, j) * source_q(q)[j];
        }
      }
      residual_e(i) = sum;
    }
  }

  nd::array< double, 2 > evaluate_weighted_shape_function_curls(nd::view<const double, 2> xi, nd::view<const double, 1> weights) {
    uint32_t q = xi.shape[0];
    nd::array<double, 2> shape_fns({q * q, num_nodes()});
    uint32_t qcount = 0;
    for (int j = 0; j < q; j++) {
      for (int i = 0; i < q; i++) {
        vec2 xi_ij = vec2{xi(i, 0), xi(j, 0)};
        for (int l = 0; l < num_nodes(); l++) {
          shape_fns(qcount, l) = shape_function_curl(xi_ij, l) * weights[i] * weights[j];
        }
        qcount++;
      }
    }
    return shape_fns;
  }

  void integrate_flux(nd::view<double> residual_e, nd::view<const flux_type> flux_q, nd::view<const double, 2> shape_fn_curl, double * /*buffer*/) const {
    int nnodes = num_nodes();
    int nqpts = flux_q.shape[0];

    for (int i = 0; i < nnodes; i++) {
      double sum = 0.0;
      for (int q = 0; q < nqpts; q++) {
        sum += shape_fn_curl(q, i) * flux_q(q);
      }
      residual_e(i) = sum;
    }
  }

  #ifdef __CUDACC__
  __device__ void cuda_integrate_flux(nd::view<double> residual_e, nd::view<const flux_type> flux_q, nd::view<const double, 2> shape_fn_curl, double * /*buffer*/) const {
    int nnodes = num_nodes();
    int nqpts = flux_q.shape[0];

    for (int i = threadIdx.x; i < nnodes; i += blockDim.x) {
      double sum = 0.0;
      for (int q = 0; q < nqpts; q++) {
        sum += shape_fn_curl(q, i) * flux_q(q);
      }
      residual_e(i) = sum;
    }
  }
  #endif

#else
  // sum factorization
  nd::array< double > evaluate_shape_function_curls(nd::view<const double,2> xi) const {
    uint32_t q = xi.shape[0];

    auto [B1, B2, G2, A1, num_entries] = scan({q*p, q*(p+1), q*(p+1), q*(p+1)});
    nd::array<double> buffer({num_entries});
    std::cout << num_entries << std::endl;
    for (int i = 0; i < q; i++) {
      GaussLegendreInterpolation(xi(i, 0), p, &buffer(B1 + p * i));
      GaussLobattoInterpolation(xi(i, 0), p+1, &buffer(B2 + (p+1)*i));
      GaussLobattoInterpolation(xi(i, 0), p+1, &buffer(G2 + (p+1)*i));
    }
    return buffer;
  }

  void curl(nd::view<double, 2> values_q, nd::view<const double> values_e, nd::view<double> buffer) const {
    uint32_t n = p + 1;
    uint32_t q = sqrt(values_q.shape[0]);

    // 1D shape function evaluations
    nd::view<double, 2> B1(buffer.data(), {q, p}); // legendre shape functions
    nd::view<double, 2> B2(B1.end(),      {q, n}); // lobatto  shape functions
    nd::view<double, 2> G2(B2.end(),      {q, n}); // lobatto  shape function derivatives
    nd::view<double, 2> A1(G2.end(),      {q, n}); // storage for intermediates

    nd::view<const double, 2> ue(values_e.data(), {n, p});
    nd::view<double, 2> uq(values_q.data(), {q, q}, {2*q, 2u});
    
    _contract(A1, B1, ue); //  A1(qx, iy) = sum_{ix} B1(qx, ix) * ue(iy, ix)  
    _contract(uq, B2, A1); //  uq(qy, qx) = sum_{iy} B2(qy, iy) * A1(qx, iy)

    ue = nd::view< const double, 2 >(values_e.data() + (n * p), {n, p});

    // note: column-major strides here, since quadrature points 
    // are still enumerated lexicographically as {y, x} but y-component nodes are {x, y}
    uq = nd::view<double, 2>(values_q.data() + 1, {q, q}, {2u, 2*q}); 
    
    _contract(A1, B1, ue); //  A1(qx, iy) = sum_{ix} B1(qx, ix) * ue(iy, ix)  
    _contract(uq, B2, A1); //  uq(qy, qx) = sum_{iy} B2(qy, iy) * A1(qx, iy)
  }
#endif

  uint32_t p;

};
// clang-format on

} // namespace refactor
