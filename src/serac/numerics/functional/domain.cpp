// Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
// other Serac Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file domain.hpp
 *
 * @brief many of the functions in this file amount to extracting
 *        element indices from an mfem::Mesh like
 *
 *    | mfem::Geometry | mfem element id | tri id | quad id |
 *    | -------------- | --------------- | ------ | ------- |
 *    | Triangle       | 0               | 0      |         |
 *    | Triangle       | 1               | 1      |         |
 *    | Square         | 2               |        | 0       |
 *    | Triangle       | 3               | 2      |         |
 *    | Square         | 4               |        | 1       |
 *    | Square         | 5               |        | 2       |
 *    | Square         | 6               |        | 3       |
 *
 *  and then evaluating a predicate function to decide whether that
 *  element gets added to a given Domain.
 *
 */

#include "serac/numerics/functional/domain.hpp"

namespace serac {

/**
 * @brief gather vertex coordinates for a list of vertices
 *
 * @param coordinates mfem's 1D list of vertex coordinates
 * @param ids the list of vertex indices to gather
 */
template <int d>
std::vector<tensor<double, d>> gather(const mfem::Vector& coordinates, mfem::Array<int> ids)
{
  int                            num_vertices = coordinates.Size() / d;
  std::vector<tensor<double, d>> x(std::size_t(ids.Size()));
  for (int v = 0; v < ids.Size(); v++) {
    for (int j = 0; j < d; j++) {
      x[uint32_t(v)][j] = coordinates[j * num_vertices + ids[v]];
    }
  }
  return x;
}

template <int d>
static Domain domain_of_vertices(const mfem::Mesh& mesh, std::function<bool(tensor<double, d>)> predicate)
{
  assert(mesh.SpaceDimension() == d);

  Domain output{mesh, 0 /* points are 0-dimensional */};

  // layout is undocumented, but it seems to be
  // [x1, x2, x3, ..., y1, y2, y3 ..., (z1, z2, z3, ...)]
  mfem::Vector vertices;
  mesh.GetVertices(vertices);

  // vertices that satisfy the predicate are added to the domain
  int num_vertices = mesh.GetNV();
  for (int i = 0; i < num_vertices; i++) {
    tensor<double, d> x;
    for (int j = 0; j < d; j++) {
      x[j] = vertices[j * num_vertices + i];
    }

    if (predicate(x)) {
      output.vertex_ids_.push_back(i);
    }
  }

  return output;
}

Domain Domain::ofVertices(const mfem::Mesh& mesh, std::function<bool(vec2)> func)
{
  return domain_of_vertices(mesh, func);
}

Domain Domain::ofVertices(const mfem::Mesh& mesh, std::function<bool(vec3)> func)
{
  return domain_of_vertices(mesh, func);
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

template <int d, typename T>
static Domain domain_of_edges(const mfem::Mesh& mesh, std::function<T> predicate)
{
  assert(mesh.SpaceDimension() == d);

  Domain output{mesh, 1 /* edges are 1-dimensional */};

  // layout is undocumented, but it seems to be
  // [x1, x2, x3, ..., y1, y2, y3 ..., (z1, z2, z3, ...)]
  mfem::Vector vertices;
  mesh.GetVertices(vertices);

  mfem::Array<int> edge_id_to_bdr_id;
  if (d == 2) {
    edge_id_to_bdr_id = mesh.GetFaceToBdrElMap();
  }

  int num_edges = mesh.GetNEdges();
  for (int i = 0; i < num_edges; i++) {
    mfem::Array<int> vertex_ids;
    mesh.GetEdgeVertices(i, vertex_ids);

    auto x = gather<d>(vertices, vertex_ids);

    if constexpr (d == 2) {
      int bdr_id = edge_id_to_bdr_id[i];
      int attr   = (bdr_id > 0) ? mesh.GetBdrAttribute(bdr_id) : -1;
      if (predicate(x, attr)) {
        output.edge_ids_.push_back(i);
        output.mfem_edge_ids_.push_back(i);
      }
    } else {
      if (predicate(x)) {
        output.edge_ids_.push_back(i);
        output.mfem_edge_ids_.push_back(i);
      }
    }
  }

  return output;
}

Domain Domain::ofEdges(const mfem::Mesh& mesh, std::function<bool(std::vector<vec2>, int)> func)
{
  return domain_of_edges<2>(mesh, func);
}

Domain Domain::ofEdges(const mfem::Mesh& mesh, std::function<bool(std::vector<vec3>)> func)
{
  return domain_of_edges<3>(mesh, func);
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

template <int d>
static Domain domain_of_faces(const mfem::Mesh&                                        mesh,
                              std::function<bool(std::vector<tensor<double, d>>, int)> predicate)
{
  assert(mesh.SpaceDimension() == d);

  Domain output{mesh, 2 /* faces are 2-dimensional */};

  // layout is undocumented, but it seems to be
  // [x1, x2, x3, ..., y1, y2, y3 ..., (z1, z2, z3, ...)]
  mfem::Vector vertices;
  mesh.GetVertices(vertices);

  mfem::Array<int> face_id_to_bdr_id;
  if (d == 3) {
    face_id_to_bdr_id = mesh.GetFaceToBdrElMap();
  }

  // faces that satisfy the predicate are added to the domain
  int num_faces;
  if (d == 2) {
    num_faces = mesh.GetNE();
  } else {
    num_faces = mesh.GetNumFaces();
  }

  int tri_id  = 0;
  int quad_id = 0;

  for (int i = 0; i < num_faces; i++) {
    mfem::Array<int> vertex_ids;

    if (mesh.Dimension() == 2) {
      mesh.GetElementVertices(i, vertex_ids);
    } else {
      mesh.GetFaceVertices(i, vertex_ids);
    }

    auto x = gather<d>(vertices, vertex_ids);

    int attr;
    if (d == 2) {
      attr = mesh.GetAttribute(i);
    } else {
      int bdr_id = face_id_to_bdr_id[i];
      attr       = (bdr_id >= 0) ? mesh.GetBdrAttribute(bdr_id) : -1;
    }

    if (predicate(x, attr)) {
      if (x.size() == 3) {
        output.tri_ids_.push_back(tri_id);
        output.mfem_tri_ids_.push_back(i);
      }
      if (x.size() == 4) {
        output.quad_ids_.push_back(quad_id);
        output.mfem_quad_ids_.push_back(i);
      }
    }

    if (x.size() == 3) {
      tri_id++;
    }
    if (x.size() == 4) {
      quad_id++;
    }
  }

  return output;
}

Domain Domain::ofFaces(const mfem::Mesh& mesh, std::function<bool(std::vector<vec2>, int)> func)
{
  return domain_of_faces(mesh, func);
}

Domain Domain::ofFaces(const mfem::Mesh& mesh, std::function<bool(std::vector<vec3>, int)> func)
{
  return domain_of_faces(mesh, func);
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

template <int d>
static Domain domain_of_elems(const mfem::Mesh&                                        mesh,
                              std::function<bool(std::vector<tensor<double, d>>, int)> predicate)
{
  assert(mesh.SpaceDimension() == d);

  Domain output{mesh, mesh.SpaceDimension() /* elems can be 2 or 3 dimensional */};

  // layout is undocumented, but it seems to be
  // [x1, x2, x3, ..., y1, y2, y3 ..., (z1, z2, z3, ...)]
  mfem::Vector vertices;
  mesh.GetVertices(vertices);

  int tri_id  = 0;
  int quad_id = 0;
  int tet_id  = 0;
  int hex_id  = 0;

  // elements that satisfy the predicate are added to the domain
  int num_elems = mesh.GetNE();
  for (int i = 0; i < num_elems; i++) {
    mfem::Array<int> vertex_ids;
    mesh.GetElementVertices(i, vertex_ids);

    auto x = gather<d>(vertices, vertex_ids);

    bool add = predicate(x, mesh.GetAttribute(i));

    switch (x.size()) {
      case 3:
        if (add) {
          output.tri_ids_.push_back(tri_id);
          output.mfem_tri_ids_.push_back(i);
        }
        tri_id++;
        break;
      case 4:
        if constexpr (d == 2) {
          if (add) {
            output.quad_ids_.push_back(quad_id);
            output.mfem_quad_ids_.push_back(i);
          }
          quad_id++;
        }
        if constexpr (d == 3) {
          if (add) {
            output.tet_ids_.push_back(tet_id);
            output.mfem_tet_ids_.push_back(i);
          }
          tet_id++;
        }
        break;
      case 8:
        if (add) {
          output.hex_ids_.push_back(hex_id);
          output.mfem_hex_ids_.push_back(i);
        }
        hex_id++;
        break;
      default:
        SLIC_ERROR("unsupported element type");
        break;
    }
  }

  return output;
}

Domain Domain::ofElements(const mfem::Mesh& mesh, std::function<bool(std::vector<vec2>, int)> func)
{
  return domain_of_elems<2>(mesh, func);
}

Domain Domain::ofElements(const mfem::Mesh& mesh, std::function<bool(std::vector<vec3>, int)> func)
{
  return domain_of_elems<3>(mesh, func);
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

template <int d>
static Domain domain_of_boundary_elems(const mfem::Mesh&                                        mesh,
                                       std::function<bool(std::vector<tensor<double, d>>, int)> predicate)
{
  assert(mesh.SpaceDimension() == d);

  Domain output{mesh, d - 1, Domain::Type::BoundaryElements};

  mfem::Array<int> face_id_to_bdr_id = mesh.GetFaceToBdrElMap();

  // layout is undocumented, but it seems to be
  // [x1, x2, x3, ..., y1, y2, y3 ..., (z1, z2, z3, ...)]
  mfem::Vector vertices;
  mesh.GetVertices(vertices);

  int edge_id = 0;
  int tri_id  = 0;
  int quad_id = 0;

  // faces that satisfy the predicate are added to the domain
  for (int f = 0; f < mesh.GetNumFaces(); f++) {
    // discard faces with the wrong type
    if (mesh.GetFaceInformation(f).IsInterior()) continue;

    auto geom = mesh.GetFaceGeometry(f);

    mfem::Array<int> vertex_ids;
    mesh.GetFaceVertices(f, vertex_ids);

    auto x = gather<d>(vertices, vertex_ids);

    int bdr_id = face_id_to_bdr_id[f];
    int attr   = (bdr_id >= 0) ? mesh.GetBdrAttribute(bdr_id) : -1;

    bool add = predicate(x, attr);

    switch (geom) {
      case mfem::Geometry::SEGMENT:
        if (add) {
          output.edge_ids_.push_back(edge_id);
          output.mfem_edge_ids_.push_back(f);
        }
        edge_id++;
        break;
      case mfem::Geometry::TRIANGLE:
        if (add) {
          output.tri_ids_.push_back(tri_id);
          output.mfem_tri_ids_.push_back(f);
        }
        tri_id++;
        break;
      case mfem::Geometry::SQUARE:
        if (add) {
          output.quad_ids_.push_back(quad_id);
          output.mfem_quad_ids_.push_back(f);
        }
        quad_id++;
        break;
      default:
        SLIC_ERROR("unsupported element type");
        break;
    }
  }

  return output;
}

Domain Domain::ofBoundaryElements(const mfem::Mesh& mesh, std::function<bool(std::vector<vec2>, int)> func)
{
  return domain_of_boundary_elems<2>(mesh, func);
}

Domain Domain::ofBoundaryElements(const mfem::Mesh& mesh, std::function<bool(std::vector<vec3>, int)> func)
{
  return domain_of_boundary_elems<3>(mesh, func);
}

mfem::Array<int> Domain::dof_list(mfem::FiniteElementSpace* fes) const
{
  std::set<int>    dof_ids;
  mfem::Array<int> elem_dofs;

  std::function<void(int i, mfem::Array<int>&)> GetDofs;
  if (type_ == Type::Elements) {
    GetDofs = [&](int i, mfem::Array<int>& vdofs) { return fes->GetElementDofs(i, vdofs); };
  }

  if (type_ == Type::BoundaryElements) {
    GetDofs = [&](int i, mfem::Array<int>& vdofs) { return fes->GetFaceDofs(i, vdofs); };
  }

  if (dim_ == 0) {
    // sam: what to do with vertex sets?
  }

  if (dim_ == 1) {
    for (auto elem_id : mfem_edge_ids_) {
      GetDofs(elem_id, elem_dofs);
      for (int i = 0; i < elem_dofs.Size(); i++) {
        dof_ids.insert(elem_dofs[i]);
      }
    }
  }

  if (dim_ == 2) {
    for (auto elem_id : mfem_tri_ids_) {
      GetDofs(elem_id, elem_dofs);
      for (int i = 0; i < elem_dofs.Size(); i++) {
        dof_ids.insert(elem_dofs[i]);
      }
    }

    for (auto elem_id : mfem_quad_ids_) {
      GetDofs(elem_id, elem_dofs);
      for (int i = 0; i < elem_dofs.Size(); i++) {
        dof_ids.insert(elem_dofs[i]);
      }
    }
  }

  if (dim_ == 3) {
    for (auto elem_id : mfem_tet_ids_) {
      GetDofs(elem_id, elem_dofs);
      for (int i = 0; i < elem_dofs.Size(); i++) {
        dof_ids.insert(elem_dofs[i]);
      }
    }

    for (auto elem_id : mfem_hex_ids_) {
      GetDofs(elem_id, elem_dofs);
      for (int i = 0; i < elem_dofs.Size(); i++) {
        dof_ids.insert(elem_dofs[i]);
      }
    }
  }

  mfem::Array<int> uniq_dof_ids(int(dof_ids.size()));
  int              i = 0;
  for (auto id : dof_ids) {
    uniq_dof_ids[i++] = id;
  }

  return uniq_dof_ids;
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

/// @cond
using c_iter = std::vector<int>::const_iterator;
using b_iter = std::back_insert_iterator<std::vector<int>>;
using set_op = std::function<b_iter(c_iter, c_iter, c_iter, c_iter, b_iter)>;

set_op union_op        = std::set_union<c_iter, c_iter, b_iter>;
set_op intersection_op = std::set_intersection<c_iter, c_iter, b_iter>;
set_op difference_op   = std::set_difference<c_iter, c_iter, b_iter>;
/// @endcond

/// @brief return a std::vector that is the result of applying (a op b)
std::vector<int> set_operation(set_op op, const std::vector<int>& a, const std::vector<int>& b)
{
  std::vector<int> output;
  op(a.begin(), a.end(), b.begin(), b.end(), back_inserter(output));
  return output;
}

/// @brief return a Domain that is the result of applying (a op b)
Domain set_operation(set_op op, const Domain& a, const Domain& b)
{
  assert(&a.mesh_ == &b.mesh_);
  assert(a.dim_ == b.dim_);

  Domain output{a.mesh_, a.dim_};

  if (output.dim_ == 0) {
    output.vertex_ids_ = set_operation(op, a.vertex_ids_, b.vertex_ids_);
  }

  if (output.dim_ == 1) {
    output.edge_ids_ = set_operation(op, a.edge_ids_, b.edge_ids_);
  }

  if (output.dim_ == 2) {
    output.tri_ids_  = set_operation(op, a.tri_ids_, b.tri_ids_);
    output.quad_ids_ = set_operation(op, a.quad_ids_, b.quad_ids_);
  }

  if (output.dim_ == 3) {
    output.tet_ids_ = set_operation(op, a.tet_ids_, b.tet_ids_);
    output.hex_ids_ = set_operation(op, a.hex_ids_, b.hex_ids_);
  }

  return output;
}

Domain operator|(const Domain& a, const Domain& b) { return set_operation(union_op, a, b); }
Domain operator&(const Domain& a, const Domain& b) { return set_operation(intersection_op, a, b); }
Domain operator-(const Domain& a, const Domain& b) { return set_operation(difference_op, a, b); }

}  // namespace serac
