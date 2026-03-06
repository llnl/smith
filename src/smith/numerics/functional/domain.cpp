// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

/**
 * @file domain.hpp
 *
 * @brief many of the functions in this file amount to extracting
 *        element indices from an mesh_t like
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

#include "smith/numerics/functional/domain.hpp"

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <tuple>

#include "smith/numerics/functional/element_restriction.hpp"

namespace smith {

/**
 * @brief gather vertex coordinates for a list of vertices
 *
 * @param coordinates mfem's 1D list of vertex coordinates
 * @param ids the list of vertex indices to gather
 */
template <int d>
std::vector<tensor<double, d>> gather(const mfem::Vector& coordinates, mfem::Array<int> ids)
{
  int num_vertices = coordinates.Size() / d;
  std::vector<tensor<double, d>> x(std::size_t(ids.Size()));
  for (int v = 0; v < ids.Size(); v++) {
    for (int j = 0; j < d; j++) {
      x[uint32_t(v)][j] = coordinates[j * num_vertices + ids[v]];
    }
  }
  return x;
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

template <int d, typename T>
static Domain domain_of_edges(const mesh_t& mesh, std::function<T> predicate)
{
  assert(mesh.SpaceDimension() == d);

  Domain output{mesh, 1 /* edges are 1-dimensional */, Domain::Type::Elements};

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
      int attr = (bdr_id >= 0) ? mesh.GetBdrAttribute(bdr_id) : -1;
      if (predicate(x, attr)) {
        output.addElement(i, i, mfem::Geometry::SEGMENT);
      }
    } else {
      if (predicate(x)) {
        output.addElement(i, i, mfem::Geometry::SEGMENT);
      }
    }
  }

  return output;
}

Domain Domain::ofEdges(const mesh_t& mesh, std::function<bool(std::vector<vec2>, int)> func)
{
  return domain_of_edges<2>(mesh, func);
}

Domain Domain::ofEdges(const mesh_t& mesh, std::function<bool(std::vector<vec3>)> func)
{
  return domain_of_edges<3>(mesh, func);
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

template <int d>
static Domain domain_of_faces(const mesh_t& mesh, std::function<bool(std::vector<tensor<double, d>>, int)> predicate)
{
  assert(mesh.SpaceDimension() == d);

  Domain output{mesh, 2 /* faces are 2-dimensional */, Domain::Type::Elements};

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

  int tri_id = 0;
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
      attr = (bdr_id >= 0) ? mesh.GetBdrAttribute(bdr_id) : -1;
    }

    if (predicate(x, attr)) {
      if (x.size() == 3) {
        output.addElement(tri_id, i, mfem::Geometry::TRIANGLE);
      }
      if (x.size() == 4) {
        output.addElement(quad_id, i, mfem::Geometry::SQUARE);
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

Domain Domain::ofFaces(const mesh_t& mesh, std::function<bool(std::vector<vec2>, int)> func)
{
  return domain_of_faces(mesh, func);
}

Domain Domain::ofFaces(const mesh_t& mesh, std::function<bool(std::vector<vec3>, int)> func)
{
  return domain_of_faces(mesh, func);
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

template <int d>
static Domain domain_of_elems(const mesh_t& mesh, std::function<bool(std::vector<tensor<double, d>>, int)> predicate)
{
  assert(mesh.SpaceDimension() == d);

  Domain output{mesh, mesh.SpaceDimension() /* elems can be 2 or 3 dimensional */, Domain::Type::Elements};

  // layout is undocumented, but it seems to be
  // [x1, x2, x3, ..., y1, y2, y3 ..., (z1, z2, z3, ...)]
  mfem::Vector vertices;
  mesh.GetVertices(vertices);

  int tri_id = 0;
  int quad_id = 0;
  int tet_id = 0;
  int hex_id = 0;

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
          output.addElement(tri_id, i, mfem::Geometry::TRIANGLE);
        }
        tri_id++;
        break;
      case 4:
        if constexpr (d == 2) {
          if (add) {
            output.addElement(quad_id, i, mfem::Geometry::SQUARE);
          }
          quad_id++;
        }
        if constexpr (d == 3) {
          if (add) {
            output.addElement(tet_id, i, mfem::Geometry::TETRAHEDRON);
          }
          tet_id++;
        }
        break;
      case 8:
        if (add) {
          output.addElement(hex_id, i, mfem::Geometry::CUBE);
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

Domain Domain::ofElements(const mesh_t& mesh, std::function<bool(std::vector<vec2>, int)> func)
{
  return domain_of_elems<2>(mesh, func);
}

Domain Domain::ofElements(const mesh_t& mesh, std::function<bool(std::vector<vec3>, int)> func)
{
  return domain_of_elems<3>(mesh, func);
}

void Domain::addElement(int geom_id, int elem_id, mfem::Geometry::Type element_geometry)
{
  if (element_geometry == mfem::Geometry::SEGMENT) {
    edge_ids_.push_back(geom_id);
    mfem_edge_ids_.push_back(elem_id);
  } else if (element_geometry == mfem::Geometry::TRIANGLE) {
    tri_ids_.push_back(geom_id);
    mfem_tri_ids_.push_back(elem_id);
  } else if (element_geometry == mfem::Geometry::SQUARE) {
    quad_ids_.push_back(geom_id);
    mfem_quad_ids_.push_back(elem_id);
  } else if (element_geometry == mfem::Geometry::TETRAHEDRON) {
    tet_ids_.push_back(geom_id);
    mfem_tet_ids_.push_back(elem_id);
  } else if (element_geometry == mfem::Geometry::CUBE) {
    hex_ids_.push_back(geom_id);
    mfem_hex_ids_.push_back(elem_id);
  } else {
    SLIC_ERROR("unsupported element type");
  }
}

void Domain::addElements(const std::vector<int>& geom_ids, const std::vector<int>& elem_ids,
                         mfem::Geometry::Type element_geometry)
{
  SLIC_ERROR_IF(geom_ids.size() != elem_ids.size(),
                "To add elements, you must specify a geom_id AND an elem_id for each element");

  for (std::vector<int>::size_type i = 0; i < geom_ids.size(); ++i) {
    addElement(geom_ids[i], elem_ids[i], element_geometry);
  }
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

template <int d>
static Domain domain_of_boundary_elems(const mesh_t& mesh,
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
  int tri_id = 0;
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
    int attr = (bdr_id >= 0) ? mesh.GetBdrAttribute(bdr_id) : -1;

    bool add = predicate(x, attr);

    switch (geom) {
      case mfem::Geometry::SEGMENT:
        if (add) {
          output.addElement(edge_id, f, geom);
        }
        edge_id++;
        break;
      case mfem::Geometry::TRIANGLE:
        if (add) {
          output.addElement(tri_id, f, geom);
        }
        tri_id++;
        break;
      case mfem::Geometry::SQUARE:
        if (add) {
          output.addElement(quad_id, f, geom);
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

Domain Domain::ofBoundaryElements(const mesh_t& mesh, std::function<bool(std::vector<vec2>, int)> func)
{
  return domain_of_boundary_elems<2>(mesh, func);
}

Domain Domain::ofBoundaryElements(const mesh_t& mesh, std::function<bool(std::vector<vec3>, int)> func)
{
  return domain_of_boundary_elems<3>(mesh, func);
}

/**
 * @brief  Get local dofs that are part of a domain, but are owned by a neighboring MPI rank
 *
 *  This is necessary for situations like this:
 *  Mesh before parallel partition:
 *      3 *--------* 2
 *        |       /|
 *        |      / |
 *        |     /  |
 *        |    /   |     <-- Edge we want in Domain
 *        |   /    |
 *        |  /     |
 *        | /      |
 *        |/       |
 * Node 0 *--------* 1
 *
 * Possible mesh after partition into two ranks:
 *
 *    RANK 0           RANK 1
 *
 * 3 *--------* 2         2 o
 *   |       /             /|
 *   |      /             / |
 *   |     /             /  |
 *   |    /             /   |  <-- Edge we want in Domain
 *   |   /             /    |
 *   |  /             /     |
 *   | /             /      |
 *   |/             /       |
 * 0 *           0 o--------* 1
 *
 *   *: locally owned node
 *   o: node owned by a neighbor rank
 *
 * We create a domain containing the right vertical edge, and then ask for its local dofs.
 * The dof list returned for Rank 1 will be correct, containing the local indices for nodes
 * 1 and 2. However, the dof list on rank 0 will not be correct without parallel communication.
 * It will see that it doesn't own the edge in question, so when it then goes to fetch the
 * local dofs on the domain, it will be an empty list.
 *
 * This function corrects for that, flagging the dofs we want on the domain on each rank
 * (using the local_dof_ids list), and then exchanging this info with
 * neighboring ranks, so that rank 0 will be told that its local dof for node 2 should be
 * added to the list of dofs on the domain.
 *
 * Before findDomainDofsOnNeighborRanks():
 * dof list on Rank 0: {}
 * dof list on Rank 1: {1, 2}
 *
 * After:
 * dof list on Rank 0: {2}
 * dof list on Rank 1: {1, 2}
 *
 * Note: the sets will actually contain the ldof indices corresponding to the global (tdof)
 * indices in the sets above.
 *
 * This function operates on the local_dof_ids data in place.
 */
void findDomainDofsOnNeighborRanks(const smith::fes_t* fes, mfem::Array<int>& local_dof_ids)
{
  auto par_fes = dynamic_cast<const mfem::ParFiniteElementSpace*>(fes);
  // There's no work to do unless the finite element space really is parallel
  if (par_fes) {
    // As far as I can tell, the parallel communication in mfem only works with
    // vector field dof indexing. So we need to get the parallel-correct scalar
    // dof ids, we do the following:
    // (1) transform scalar ldof ids to vector ldof ids,
    // (2) transform the vector ldof ids into a boolean "marker" ldof field
    // (3) do our parallel sync, which applies an OR logic operator to the
    //     boolean fields from all ranks at each dof
    // (4) get the ldof indices of the TRUE values of the parallel-correct
    //     boolean ldof field
    // (5) transform the parallel-correct vector ldof ids back to scalar dof ids.
    fes->DofsToVDofs(0, local_dof_ids);

    mfem::Array<int> local_dof_markers;
    mfem::FiniteElementSpace::ListToMarker(local_dof_ids, par_fes->GetVSize(), local_dof_markers, 1);

    par_fes->Synchronize(local_dof_markers);

    mfem::FiniteElementSpace::MarkerToList(local_dof_markers, local_dof_ids);

    for (int i = 0; i < local_dof_ids.Size(); i++) {
      local_dof_ids[i] = par_fes->VDofToDof(local_dof_ids[i]);
    }
  }
}

mfem::Array<int> Domain::dof_list(const smith::fes_t* fes) const
{
  std::set<int> dof_ids;
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
  int i = 0;
  for (auto id : dof_ids) {
    uniq_dof_ids[i++] = id;
  }

  findDomainDofsOnNeighborRanks(fes, uniq_dof_ids);

  return uniq_dof_ids;
}

void Domain::insert_restriction(const smith::fes_t* fes, FunctionSpace space)
{
  // if we don't already have a BlockElementRestriction for this FunctionSpace, make one
  if (restriction_operators.count(space) == 0) {
    restriction_operators[space] = BlockElementRestriction(fes, *this);
  }
}

const BlockElementRestriction& Domain::get_restriction(FunctionSpace space) { return restriction_operators.at(space); };

void Domain::insert_shared_interior_face_list()
{
  // Weights only need to be computed for Domain of InteriorFaces type
  SLIC_ERROR_ROOT_IF(type_ != Domain::Type::InteriorFaces, "This method is only for interior face domains");

  // make a list if we don't already have one
  if (shared_interior_face_ids_.empty()) {
    if (dim_ == 1) {
      int i = 0;
      for (int f : mfem_edge_ids_) {
        mfem::Mesh::FaceInformation info = mesh_.GetFaceInformation(f);

        if (info.IsShared()) {
          shared_interior_face_ids_.push_back(i);
        }

        ++i;
      }
    } else if (dim_ == 2) {
      int i = 0;
      for (int f : mfem_tri_ids_) {
        mfem::Mesh::FaceInformation info = mesh_.GetFaceInformation(f);

        if (info.IsShared()) {
          shared_interior_face_ids_.push_back(i);
        }

        ++i;
      }

      for (int f : mfem_quad_ids_) {
        mfem::Mesh::FaceInformation info = mesh_.GetFaceInformation(f);

        if (info.IsShared()) {
          shared_interior_face_ids_.push_back(i);
        }

        ++i;
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

Domain EntireDomain(const mesh_t& mesh)
{
  switch (mesh.SpaceDimension()) {
    case 2:
      return Domain::ofElements(mesh, [](std::vector<vec2>, int) { return true; });
      break;
    case 3:
      return Domain::ofElements(mesh, [](std::vector<vec3>, int) { return true; });
      break;
    default:
      SLIC_ERROR("In valid spatial dimension. Domains may only be created on 2D or 3D meshes.");
      exit(-1);
  }
}

Domain EntireBoundary(const mesh_t& mesh)
{
  switch (mesh.SpaceDimension()) {
    case 2:
      return Domain::ofBoundaryElements(mesh, [](std::vector<vec2>, int) { return true; });
      break;
    case 3:
      return Domain::ofBoundaryElements(mesh, [](std::vector<vec3>, int) { return true; });
      break;
    default:
      SLIC_ERROR("In valid spatial dimension. Domains may only be created on 2D or 3D meshes.");
      exit(-1);
  }
}

/// @brief constructs a domain from all the interior face elements in a mesh
Domain InteriorFaces(const mesh_t& mesh)
{
  Domain output{mesh, mesh.SpaceDimension() - 1, Domain::Type::InteriorFaces};

  int edge_id = 0;
  int tri_id = 0;
  int quad_id = 0;

  for (int f = 0; f < mesh.GetNumFaces(); f++) {
    // discard faces with the wrong type
    if (!mesh.GetFaceInformation(f).IsInterior()) continue;

    auto geom = mesh.GetFaceGeometry(f);

    switch (geom) {
      case mfem::Geometry::SEGMENT:
        output.edge_ids_.push_back(edge_id++);
        output.mfem_edge_ids_.push_back(f);
        break;
      case mfem::Geometry::TRIANGLE:
        output.tri_ids_.push_back(tri_id++);
        output.mfem_tri_ids_.push_back(f);
        break;
      case mfem::Geometry::SQUARE:
        output.quad_ids_.push_back(quad_id++);
        output.mfem_quad_ids_.push_back(f);
        break;
      default:
        SLIC_ERROR("unsupported element type");
        break;
    }
  }

  output.insert_shared_interior_face_list();

  return output;
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

/// @cond
using int2 = std::tuple<int, int>;
enum SET_OPERATION
{
  UNION,
  INTERSECTION,
  DIFFERENCE
};
/// @endcond

/// @brief combine a pair of arrays of ints into a single array of `int2`, see also: unzip()
void zip(std::vector<int2>& ab, const std::vector<int>& a, const std::vector<int>& b)
{
  ab.resize(a.size());
  for (uint32_t i = 0; i < a.size(); i++) {
    ab[i] = {a[i], b[i]};
  }
}

/// @brief split an array of `int2` into a pair of arrays of ints, see also: zip()
void unzip(const std::vector<int2>& ab, std::vector<int>& a, std::vector<int>& b)
{
  a.resize(ab.size());
  b.resize(ab.size());
  for (uint32_t i = 0; i < ab.size(); i++) {
    auto ab_i = ab[i];
    a[i] = std::get<0>(ab_i);
    b[i] = std::get<1>(ab_i);
  }
}

/// @brief return a std::vector that is the result of applying (a op b)
template <typename T>
std::vector<T> set_operation(SET_OPERATION op, const std::vector<T>& a, const std::vector<T>& b)
{
  using c_iter = typename std::vector<T>::const_iterator;
  using b_iter = std::back_insert_iterator<std::vector<T>>;
  using set_op = std::function<b_iter(c_iter, c_iter, c_iter, c_iter, b_iter)>;

  set_op combine;
  if (op == SET_OPERATION::UNION) {
    combine = std::set_union<c_iter, c_iter, b_iter>;
  }
  if (op == SET_OPERATION::INTERSECTION) {
    combine = std::set_intersection<c_iter, c_iter, b_iter>;
  }
  if (op == SET_OPERATION::DIFFERENCE) {
    combine = std::set_difference<c_iter, c_iter, b_iter>;
  }

  std::vector<T> combined;
  combine(a.begin(), a.end(), b.begin(), b.end(), back_inserter(combined));
  return combined;
}

/// @brief return a Domain that is the result of applying (a op b)
Domain set_operation(SET_OPERATION op, const Domain& a, const Domain& b)
{
  assert(&a.mesh_ == &b.mesh_);
  assert(a.dim_ == b.dim_);
  assert(a.type_ == b.type_);

  Domain combined{a.mesh_, a.dim_, a.type_};

  using Ids = std::vector<int>;
  auto apply_set_op = [&op](const Ids& x, const Ids& y) { return set_operation(op, x, y); };

  auto fill_combined_lists = [apply_set_op, &combined](const Ids& a_ids, const Ids& a_mfem_ids, const Ids& b_ids,
                                                       const Ids& b_mfem_ids, mfem::Geometry::Type g) {
    auto combined_ids = apply_set_op(a_ids, b_ids);
    auto combined_mfem_ids = apply_set_op(a_mfem_ids, b_mfem_ids);
    combined.addElements(combined_ids, combined_mfem_ids, g);
  };

  if (combined.dim_ == 1) {
    fill_combined_lists(a.edge_ids_, a.mfem_edge_ids_, b.edge_ids_, b.mfem_edge_ids_, mfem::Geometry::SEGMENT);
  }

  if (combined.dim_ == 2) {
    fill_combined_lists(a.tri_ids_, a.mfem_tri_ids_, b.tri_ids_, b.mfem_tri_ids_, mfem::Geometry::TRIANGLE);
    fill_combined_lists(a.quad_ids_, a.mfem_quad_ids_, b.quad_ids_, b.mfem_quad_ids_, mfem::Geometry::SQUARE);
  }

  if (combined.dim_ == 3) {
    fill_combined_lists(a.tet_ids_, a.mfem_tet_ids_, b.tet_ids_, b.mfem_tet_ids_, mfem::Geometry::TETRAHEDRON);
    fill_combined_lists(a.hex_ids_, a.mfem_hex_ids_, b.hex_ids_, b.mfem_hex_ids_, mfem::Geometry::CUBE);
  }

  return combined;
}

Domain operator|(const Domain& a, const Domain& b) { return set_operation(SET_OPERATION::UNION, a, b); }
Domain operator&(const Domain& a, const Domain& b) { return set_operation(SET_OPERATION::INTERSECTION, a, b); }
Domain operator-(const Domain& a, const Domain& b) { return set_operation(SET_OPERATION::DIFFERENCE, a, b); }

}  // namespace smith
