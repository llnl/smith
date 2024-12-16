#pragma once

#include <array>
#include <cinttypes>

#include "serac/infrastructure/accelerator.hpp"
#include "serac/numerics/refactor/containers/ndarray.hpp"

#include "mfem.hpp"

namespace refactor {

namespace impl {
  template < mfem::Geometry::Type g >
  struct constexpr_geometry{
    constexpr operator mfem::Geometry::Type() { return g; }
  };
}

template < typename T >
void foreach_geometry(T && function) {
  function(impl::constexpr_geometry< mfem::Geometry::SEGMENT >{});
  function(impl::constexpr_geometry< mfem::Geometry::TRIANGLE >{});
  function(impl::constexpr_geometry< mfem::Geometry::SQUARE >{});
  function(impl::constexpr_geometry< mfem::Geometry::TETRAHEDRON >{});
  function(impl::constexpr_geometry< mfem::Geometry::CUBE >{});
}

inline const char * to_string(mfem::Geometry::Type g) {
  switch (g) {
    case mfem::Geometry::POINT:       return "Geometry_Vertex";
    case mfem::Geometry::SEGMENT:     return "Geometry_Edge";
    case mfem::Geometry::TRIANGLE:    return "Geometry_Triangle";
    case mfem::Geometry::SQUARE:      return "Geometry_Quadrilateral";
    case mfem::Geometry::TETRAHEDRON: return "Geometry_Tetrahedron";
    case mfem::Geometry::CUBE:        return "Geometry_Hexahedron";
    default: return "";
  }
  return "";
}

constexpr mfem::Geometry::Type all_geometries[6] = {
  mfem::Geometry::POINT,
  mfem::Geometry::SEGMENT,
  mfem::Geometry::TRIANGLE,
  mfem::Geometry::SQUARE,
  mfem::Geometry::TETRAHEDRON,
  mfem::Geometry::CUBE
};

constexpr nd::range<const mfem::Geometry::Type *> geometries_by_dim[4] = {
  {all_geometries+0, all_geometries+1},
  {all_geometries+1, all_geometries+2},
  {all_geometries+2, all_geometries+4},
  {all_geometries+4, all_geometries+6}
};

//constexpr uint32_t dimension(mfem::Geometry::Type g) {
SERAC_HOST_DEVICE constexpr uint32_t dimension(mfem::Geometry::Type g) {
  switch (g) {
    case mfem::Geometry::POINT:       return 0;
    case mfem::Geometry::SEGMENT:     return 1;
    case mfem::Geometry::TRIANGLE:    return 2;
    case mfem::Geometry::SQUARE:      return 2;
    case mfem::Geometry::TETRAHEDRON: return 3;
    case mfem::Geometry::CUBE:        return 3;
    default: return 1u<<30;
  }
}

template < typename T >
struct GeometryData {
  T vert;
  T edge;
  T tri;
  T quad;
  T tet;
  T hex;

  T & operator[](mfem::Geometry::Type g) {
    switch (g) {
      case mfem::Geometry::POINT:       return vert;
      case mfem::Geometry::SEGMENT:     return edge;
      case mfem::Geometry::TRIANGLE:    return tri;
      case mfem::Geometry::SQUARE:      return quad;
      case mfem::Geometry::TETRAHEDRON: return tet;
      case mfem::Geometry::CUBE:        return hex;
      default: return vert; // (hopefully) unreachable code, to silence compiler warnings
    }
  }

  const T & operator[](mfem::Geometry::Type g) const {
    switch (g) {
      case mfem::Geometry::POINT:       return vert;
      case mfem::Geometry::SEGMENT:     return edge;
      case mfem::Geometry::TRIANGLE:    return tri;
      case mfem::Geometry::SQUARE:      return quad;
      case mfem::Geometry::TETRAHEDRON: return tet;
      case mfem::Geometry::CUBE:        return hex;
      default: return vert; // (hopefully) unreachable code, to silence compiler warnings
    }
  }

};

struct GeometryInfo : public GeometryData<uint32_t> {
  static GeometryInfo from_array(uint32_t * data) {
    return GeometryInfo { data[0], data[1], data[2], data[3], data[4], data[5] };
  }
};

inline void operator+=(GeometryInfo & a, const GeometryInfo & b) {
  a.vert += b.vert;
  a.edge += b.edge;
  a.tri += b.tri;
  a.quad += b.quad;
  a.tet += b.tet;
  a.hex += b.hex;
};

inline GeometryInfo operator*(GeometryInfo a, GeometryInfo b) {
  return GeometryInfo{
    a.vert * b.vert,
    a.edge * b.edge,
    a.tri * b.tri,
    a.quad * b.quad,
    a.tet * b.tet,
    a.hex * b.hex
  };
};

inline GeometryInfo operator*(GeometryInfo a, uint32_t scale) {
  return GeometryInfo{
    a.vert * scale,
    a.edge * scale,
    a.tri * scale,
    a.quad * scale,
    a.tet * scale,
    a.hex * scale
  };
};

inline uint32_t total(GeometryInfo input){
  return input.vert + input.edge + input.tri + input.quad + input.tet + input.hex;
};

inline GeometryInfo scan(const GeometryInfo & input){
  GeometryInfo output;
  output.vert  = 0;
  output.edge  = output.vert + input.vert;
  output.tri   = output.edge + input.edge;
  output.quad  = output.tri  + input.tri;
  output.tet   = output.quad + input.quad;
  output.hex   = output.tet  + input.tet;
  return output;
};

template < uint32_t n >
std::array< uint32_t, n + 1 > scan(const uint32_t (&input)[n]){
  std::array< uint32_t, n + 1 > output{};
  for (uint32_t i = 0; i < n; i++) {
    output[i+1] = output[i] + input[i];
  }
  return output;
};



template < mfem::Geometry::Type g >
struct GeometryType;

template <>
struct GeometryType< mfem::Geometry::POINT > {
  static constexpr mfem::Geometry::Type geometry = mfem::Geometry::POINT;
  static constexpr int offset = 0;
  static constexpr int dim = 0; 
};
using Vertex = GeometryType< mfem::Geometry::POINT >;

template <>
struct GeometryType< mfem::Geometry::SEGMENT > {
  static constexpr mfem::Geometry::Type geometry = mfem::Geometry::SEGMENT;
  static constexpr int offset = 1;

  static constexpr int dim = 1; 
  static constexpr int num_vertices = 2;

  static constexpr int cell_offset = num_vertices;
};
using Edge = GeometryType< mfem::Geometry::SEGMENT >;

template <>
struct GeometryType< mfem::Geometry::TRIANGLE > {
  static constexpr mfem::Geometry::Type geometry = mfem::Geometry::TRIANGLE;
  static constexpr int offset = 2;

  static constexpr int dim = 2; 
  static constexpr int num_vertices = 3;
  static constexpr int num_edges = 3;

  static constexpr int edge_offset = num_vertices;
  static constexpr int cell_offset = num_vertices + num_edges;

  static constexpr int local_edge_ids[3][2] = {{0, 1},{1, 2},{2, 0}};

  SERAC_HOST_DEVICE static constexpr uint32_t number(uint32_t n) { return (n * (n + 1)) / 2; };
};
using Triangle = GeometryType< mfem::Geometry::TRIANGLE >;

template <>
struct GeometryType< mfem::Geometry::SQUARE > {
  static constexpr mfem::Geometry::Type geometry = mfem::Geometry::SQUARE;
  static constexpr int offset = 3;

  static constexpr int dim = 2; 
  static constexpr int num_vertices = 4;
  static constexpr int num_edges = 4;

  static constexpr int edge_offset = num_vertices;
  static constexpr int cell_offset = edge_offset + num_edges;

  static constexpr int local_edge_ids[4][4] = {{0, 1},{1, 2},{2, 3},{3,0}};
};
using Quadrilateral = GeometryType< mfem::Geometry::SQUARE >;

template <>
struct GeometryType< mfem::Geometry::TETRAHEDRON > {
  static constexpr mfem::Geometry::Type geometry = mfem::Geometry::TETRAHEDRON;
  static constexpr int offset = 4;

  static constexpr int dim = 3; 
  static constexpr int num_vertices = 4;
  static constexpr int num_edges = 6;
  static constexpr int num_triangles = 4;
  static constexpr int num_quadrilaterals = 0;

  static constexpr int edge_offset = num_vertices;
  static constexpr int tri_offset = edge_offset + num_edges;
  static constexpr int quad_offset = tri_offset + num_triangles;
  static constexpr int cell_offset = quad_offset + num_quadrilaterals;

  static constexpr double vertices[4][3] = {{0,0,0}, {1,0,0}, {0,1,0}, {0,0,1}};
  static constexpr int local_edge_ids[6][2] = {{0, 1}, {1, 2}, {2, 0}, {0, 3}, {1, 3}, {2, 3}};
  static constexpr int local_triangle_ids[4][3] = {{2, 1, 0}, {0, 1, 3}, {1, 2, 3}, {2, 0, 3}};

  SERAC_HOST_DEVICE static constexpr uint32_t number(uint32_t n) { return (n * (n + 1) * (n + 2)) / 6; }
};
using Tetrahedron = GeometryType< mfem::Geometry::TETRAHEDRON >;

template <>
struct GeometryType< mfem::Geometry::CUBE > {
  static constexpr mfem::Geometry::Type geometry = mfem::Geometry::CUBE;
  static constexpr int offset = 5;

  static constexpr int dim = 3; 
  static constexpr int num_vertices = 8;
  static constexpr int num_edges = 12;
  static constexpr int num_triangles = 0;
  static constexpr int num_quadrilaterals = 6;

  static constexpr int edge_offset = num_vertices;
  static constexpr int tri_offset = edge_offset + num_edges;
  static constexpr int quad_offset = tri_offset + num_triangles;
  static constexpr int cell_offset = quad_offset + num_quadrilaterals;

  //  mathematica code for visualizing these edge/quad numberings
  /*
    localEdgeIds = 1 +  {{0, 1},{1, 2},{3, 2},{0, 3},{0, 4},{1, 5},{2, 6},{3, 7},{4, 5},{5, 6},{7, 6},{4, 7}};
    localQuadIds = 1 + {{1, 0, 3, 2}, {0, 1, 5, 4}, {1, 2, 6, 5}, {2, 3, 7, 6}, {3, 0, 4, 7}, {4, 5, 6, 7}};
    vertices = {{-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1}, {-1, -1, 1}, {1, -1, 1}, {1, 1, 1}, {-1, 1, 1}};
    Graphics3D[{
      Thickness[0.01], JoinForm["Round"], Black, PointSize[0.03], 
      Point /@ vertices, Table[Text[Style[i - 1, Large], 1.2 vertices[[i]]], {i, 1, 8}],
      Red, Arrow[( { {0.9, 0.1}, {0.1, 0.9} } ) . vertices[[#]]] & /@ localEdgeIds,
      Table[Text[Style[i - 1, Large], 1.3 Mean[vertices[[localEdgeIds[[i]]]]]], {i, 1, 12}],
      Blue, Arrow[( {
            {0.7, 0.1, 0.1, 0.1},
            {0.1, 0.7, 0.1, 0.1},
            {0.1, 0.1, 0.7, 0.1},
            {0.1, 0.1, 0.1, 0.7}
           } ) . vertices[[#]]] & /@ localQuadIds,
      Table[Text[Style[i - 1, Large], Mean[vertices[[localQuadIds[[i]]]]]], {i, 1, 6}]
    }, Boxed -> False]
  */
  static constexpr int local_edge_ids[12][2] = {{0, 1},{1, 2},{3, 2},{0, 3},{0, 4},{1, 5},{2, 6},{3, 7},{4, 5},{5, 6},{7, 6},{4, 7}};
  static constexpr int local_quadrilateral_ids[6][4] = {{1, 0, 3, 2},{0, 1, 5, 4},{1, 2, 6, 5},{2, 3, 7, 6},{3, 0, 4, 7},{4, 5, 6, 7}};
};
using Hexahedron = GeometryType< mfem::Geometry::CUBE >;

}
