#include "serac/numerics/functional/element_restriction.hpp"

#include "mfem.hpp"

#include "serac/numerics/functional/geometry.hpp"

std::vector<std::vector<int> > lexicographic_permutations(int p)
{
  // p == 0 is admissible for L2 spaces, but lexicographic permutations
  // aren't needed in that corner case
  if (p == 0) {
    return {};
  }

  std::vector<std::vector<int> > output(mfem::Geometry::Type::NUM_GEOMETRIES);

  {
    auto             P = mfem::H1_SegmentElement(p).GetLexicographicOrdering();
    std::vector<int> native_to_lex(uint32_t(P.Size()));
    for (int i = 0; i < P.Size(); i++) {
      native_to_lex[uint32_t(i)] = P[i];
    }
    output[mfem::Geometry::Type::SEGMENT] = native_to_lex;
  }

  {
    auto             P = mfem::H1_TriangleElement(p).GetLexicographicOrdering();
    std::vector<int> native_to_lex(uint32_t(P.Size()));
    for (int i = 0; i < P.Size(); i++) {
      native_to_lex[uint32_t(i)] = P[i];
    }
    output[mfem::Geometry::Type::TRIANGLE] = native_to_lex;
  }

  {
    auto             P = mfem::H1_QuadrilateralElement(p).GetLexicographicOrdering();
    std::vector<int> native_to_lex(uint32_t(P.Size()));
    for (int i = 0; i < P.Size(); i++) {
      native_to_lex[uint32_t(i)] = P[i];
    }
    output[mfem::Geometry::Type::SQUARE] = native_to_lex;
  }

  {
    auto             P = mfem::H1_TetrahedronElement(p).GetLexicographicOrdering();
    std::vector<int> native_to_lex(uint32_t(P.Size()));
    for (int i = 0; i < P.Size(); i++) {
      native_to_lex[uint32_t(i)] = P[i];
    }
    output[mfem::Geometry::Type::TETRAHEDRON] = native_to_lex;
  }

  {
    auto             P = mfem::H1_HexahedronElement(p).GetLexicographicOrdering();
    std::vector<int> native_to_lex(uint32_t(P.Size()));
    for (int i = 0; i < P.Size(); i++) {
      native_to_lex[uint32_t(i)] = P[i];
    }
    output[mfem::Geometry::Type::CUBE] = native_to_lex;
  }

  // other geometries are not defined, as they are not currently used

  return output;
}

Array2D<int> face_permutations(mfem::Geometry::Type geom, int p)
{
  if (geom == mfem::Geometry::Type::SEGMENT) {
    Array2D<int> output(2, p + 1);
    for (int i = 0; i <= p; i++) {
      output(0, i) = i;
      output(1, i) = p - i;
    }
    return output;
  }

  if (geom == mfem::Geometry::Type::TRIANGLE) {
    // v = {{0, 0}, {1, 0}, {0, 1}};
    // f = Transpose[{{0, 1, 2}, {1, 0, 2}, {2, 0, 1}, {2, 1, 0}, {1, 2, 0}, {0, 2, 1}} + 1];
    // p v[[f[[1]]]] +  (v[[f[[2]]]] - v[[f[[1]]]]) i + (v[[f[[3]]]] - v[[f[[1]]]]) j
    //
    // {{i, j}, {p-i-j, j}, {j, p-i-j}, {i, p-i-j}, {p-i-j, i}, {j, i}}
    Array2D<int> output(6, (p + 1) * (p + 2) / 2);
    auto         tri_id = [p](int x, int y) { return x + ((3 + 2 * p - y) * y) / 2; };
    for (int j = 0; j <= p; j++) {
      for (int i = 0; i <= p - j; i++) {
        int id                          = tri_id(i, j);
        output(0, tri_id(i, j))         = id;
        output(1, tri_id(p - i - j, j)) = id;
        output(2, tri_id(j, p - i - j)) = id;
        output(3, tri_id(i, p - i - j)) = id;
        output(4, tri_id(p - i - j, i)) = id;
        output(5, tri_id(j, i))         = id;
      }
    }
    return output;
  }

  if (geom == mfem::Geometry::Type::SQUARE) {
    // v = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
    // f = Transpose[{{0, 1, 2, 3}, {0, 3, 2, 1}, {1, 2, 3, 0}, {1, 0, 3, 2},
    //                {2, 3, 0, 1}, {2, 1, 0, 3}, {3, 0, 1, 2}, {3, 2, 1, 0}} + 1];
    // p v[[f[[1]]]] +  (v[[f[[2]]]] - v[[f[[1]]]]) i + (v[[f[[4]]]] - v[[f[[1]]]]) j
    //
    // {{i,j}, {j,i}, {p-j,i}, {p-i,j}, {p-i, p-j}, {p-j, p-i}, {j, p-i}, {i, p-j}}
    Array2D<int> output(8, (p + 1) * (p + 1));
    auto         quad_id = [p](int x, int y) { return ((p + 1) * y) + x; };
    for (int j = 0; j <= p; j++) {
      for (int i = 0; i <= p; i++) {
        int id                           = quad_id(i, j);
        output(0, quad_id(i, j))         = id;
        output(1, quad_id(j, i))         = id;
        output(2, quad_id(p - j, i))     = id;
        output(3, quad_id(p - i, j))     = id;
        output(4, quad_id(p - i, p - j)) = id;
        output(5, quad_id(p - j, p - i)) = id;
        output(6, quad_id(j, p - i))     = id;
        output(7, quad_id(i, p - j))     = id;
      }
    }
    return output;
  }

  std::cout << "face_permutation(): unsupported geometry type" << std::endl;
  exit(1);
}

std::vector<Array2D<int> > geom_local_face_dofs(int p)
{
  // FullSimplify[InterpolatingPolynomial[{
  //   {{0, 2}, (p + 1) + p},
  //   {{0, 1}, p + 1}, {{1, 1}, p + 2},
  //   {{0, 0}, 0}, {{1, 0}, 1}, {{2, 0}, 2}
  // }, {x, y}]]
  //
  // x + 1/2 (3 + 2 p - y) y
  auto tri_id = [p](int x, int y) { return x + ((3 + 2 * p - y) * y) / 2; };

  // FullSimplify[InterpolatingPolynomial[{
  //  {{0, 3}, ((p - 1) (p) + (p) (p + 1) + (p + 1) (p + 2))/2},
  //  {{0, 2}, ((p) (p + 1) + (p + 1) (p + 2))/2}, {{1, 2},  p - 1 + ((p) (p + 1) + (p + 1) (p + 2))/2},
  //  {{0, 1}, (p + 1) (p + 2)/2}, {{1, 1}, p + (p + 1) (p + 2)/2}, {{2, 1}, 2 p - 1 + (p + 1) (p + 2)/2},
  //  {{0, 0}, 0}, {{1, 0}, p + 1}, {{2, 0}, 2 p + 1}, {{3, 0}, 3 p}
  // }, {y, z}]] + x
  //
  // x + (z (11 + p (12 + 3 p) - 6 y + z (z - 6 - 3 p)) - 3 y (y - 2 p - 3))/6
  auto tet_id = [p](int x, int y, int z) {
    return x + (z * (11 + p * (12 + 3 * p) - 6 * y + z * (z - 6 - 3 * p)) - 3 * y * (y - 2 * p - 3)) / 6;
  };

  auto quad_id = [p](int x, int y) { return ((p + 1) * y) + x; };

  auto hex_id = [p](int x, int y, int z) { return (p + 1) * ((p + 1) * z + y) + x; };

  std::vector<Array2D<int> > output(mfem::Geometry::Type::NUM_GEOMETRIES);

  Array2D<int> tris(3, p + 1);
  for (int k = 0; k <= p; k++) {
    tris(0, k) = tri_id(k, 0);
    tris(1, k) = tri_id(p - k, k);
    tris(2, k) = tri_id(0, p - k);
  }
  output[mfem::Geometry::Type::TRIANGLE] = tris;

  Array2D<int> quads(4, p + 1);
  for (int k = 0; k <= p; k++) {
    quads(0, k) = quad_id(k, 0);
    quads(1, k) = quad_id(p, k);
    quads(2, k) = quad_id(p - k, p);
    quads(3, k) = quad_id(0, p - k);
  }
  output[mfem::Geometry::Type::SQUARE] = quads;

  // v = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  // f = Transpose[{{1, 2, 3}, {0, 3, 2}, {0, 1, 3}, {0, 2, 1}} + 1];
  // p v[[f[[1]]]] +  (v[[f[[2]]]] - v[[f[[1]]]]) j + (v[[f[[3]]]] - v[[f[[1]]]]) k
  //
  // {{p-j-k, j, k}, {0, k, j}, {j, 0, k}, {k, j, 0}}
  Array2D<int> tets(4, (p + 1) * (p + 2) / 2);
  for (int k = 0; k <= p; k++) {
    for (int j = 0; j <= p - k; j++) {
      int id      = tri_id(j, k);
      tets(0, id) = tet_id(p - j - k, j, k);
      tets(1, id) = tet_id(0, k, j);
      tets(2, id) = tet_id(j, 0, k);
      tets(3, id) = tet_id(k, j, 0);
    }
  }
  output[mfem::Geometry::Type::TETRAHEDRON] = tets;

  // v = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
  //      {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
  // f = Transpose[{{3, 2, 1, 0}, {0, 1, 5, 4}, {1, 2, 6, 5},
  //               {2, 3, 7, 6}, {3, 0, 4, 7}, {4, 5, 6, 7}} + 1];
  // p v[[f[[1]]]] +  (v[[f[[2]]]] - v[[f[[1]]]]) j + (v[[f[[4]]]] - v[[f[[1]]]]) k
  //
  // {{j, p-k, 0}, {j, 0, k}, {p, j, k}, {p-j, p, k}, {0, p-j, k}, {j, k, p}}
  Array2D<int> hexes(6, (p + 1) * (p + 1));
  for (int k = 0; k <= p; k++) {
    for (int j = 0; j <= p; j++) {
      int id       = quad_id(j, k);
      hexes(0, id) = hex_id(j, p - k, 0);
      hexes(1, id) = hex_id(j, 0, k);
      hexes(2, id) = hex_id(p, j, k);
      hexes(3, id) = hex_id(p - j, p, k);
      hexes(4, id) = hex_id(0, p - j, k);
      hexes(5, id) = hex_id(j, k, p);
    }
  }
  output[mfem::Geometry::Type::CUBE] = hexes;

  return output;
}

#ifdef SERAC_USE_UMPIRE
axom::Array<DoF, 2, axom::MemorySpace::Host>
#else
axom::Array<DoF, 2>
#endif
GetElementRestriction(const mfem::FiniteElementSpace* fes, mfem::Geometry::Type geom)
{
  std::vector<DoF> elem_dofs{};
  mfem::Mesh*      mesh = fes->GetMesh();

  // note: this assumes that all the elements are the same polynomial order
  int                            p        = fes->GetElementOrder(0);
  std::vector<std::vector<int> > lex_perm = lexicographic_permutations(p);

  uint64_t n = 0;

  for (int elem = 0; elem < fes->GetNE(); elem++) {
    // discard elements with the wrong geometry
    if (mesh->GetElementGeometry(elem) != geom) continue;

    mfem::Array<int> dofs;

    [[maybe_unused]] auto* dof_transformation = fes->GetElementDofs(elem, dofs);

    // mfem returns the H1 dofs in "native" order, so we need
    // to apply the native-to-lexicographic permutation
    if (isH1(*fes)) {
      for (int k = 0; k < dofs.Size(); k++) {
        elem_dofs.push_back({uint64_t(dofs[lex_perm[uint32_t(geom)][uint32_t(k)]])});
      }
    }

    // the dofs mfem returns for Hcurl include information about
    // dof orientation, but not for triangle faces on 3D elements.
    // So, we need to manually
    if (isHcurl(*fes)) {
      // TODO
      // TODO
      // TODO
      uint64_t sign        = 1;
      uint64_t orientation = 0;
      for (int k = 0; k < dofs.Size(); k++) {
        elem_dofs.push_back({uint64_t(dofs[k]), sign, orientation});
      }
    }

    // mfem returns DG dofs in lexicographic order already
    // so no permutation is required here
    if (isDG(*fes)) {
      for (int k = 0; k < dofs.Size(); k++) {
        elem_dofs.push_back({uint64_t(dofs[k])});
      }
    }

    n++;
  }

  if (n == 0) {
#ifdef SERAC_USE_UMPIRE
    return axom::Array<DoF, 2, axom::MemorySpace::Host>{};
#else
    return axom::Array<DoF, 2>{};
#endif
  } else {
    uint64_t dofs_per_elem = elem_dofs.size() / n;
#ifdef SERAC_USE_UMPIRE
    axom::Array<DoF, 2, axom::MemorySpace::Host> output(n, dofs_per_elem);
#else
    axom::Array<DoF, 2> output(n, dofs_per_elem);
#endif
    std::memcpy(output.data(), elem_dofs.data(), sizeof(DoF) * n * dofs_per_elem);
    return output;
  }
}

#ifdef SERAC_USE_UMPIRE
axom::Array<DoF, 2, axom::MemorySpace::Host>
#else
axom::Array<DoF, 2>
#endif
GetFaceDofs(const mfem::FiniteElementSpace* fes, mfem::Geometry::Type face_geom, FaceType type)
{
  std::vector<DoF> face_dofs;
  mfem::Mesh*      mesh         = fes->GetMesh();
  mfem::Table*     face_to_elem = mesh->GetFaceToElementTable();

  // note: this assumes that all the elements are the same polynomial order
  int                            p               = fes->GetElementOrder(0);
  Array2D<int>                   face_perm       = face_permutations(face_geom, p);
  std::vector<Array2D<int> >     local_face_dofs = geom_local_face_dofs(p);
  std::vector<std::vector<int> > lex_perm        = lexicographic_permutations(p);

  uint64_t n = 0;

  for (int f = 0; f < fes->GetNF(); f++) {
    auto faceinfo = mesh->GetFaceInformation(f);

    // discard faces with the wrong geometry or type
    if (mesh->GetFaceGeometry(f) != face_geom) continue;
    if (faceinfo.IsInterior() && type == FaceType::BOUNDARY) continue;
    if (faceinfo.IsBoundary() && type == FaceType::INTERIOR) continue;

    // mfem doesn't provide this connectivity info for DG spaces directly,
    // so we have to get at it indirectly in several steps:
    if (isDG(*fes)) {
      // 1. find the element(s) that this face belongs to
      mfem::Array<int> elem_ids;
      face_to_elem->GetRow(f, elem_ids);

      for (auto elem : elem_ids) {
        // 2a. get the list of faces (and their orientations) that belong to that element ...
        mfem::Array<int> elem_side_ids, orientations;
        if (mesh->Dimension() == 2) {
          mesh->GetElementEdges(elem, elem_side_ids, orientations);

          // mfem returns {-1, 1} for edge orientations,
          // but {0, 1, ... , n} for face orientations.
          // Here, we renumber the edge orientations to
          // {0 (no permutation), 1 (reversed)} so the values can be
          // consistently used as indices into a permutation table
          for (auto& o : orientations) {
            o = (o == -1) ? 1 : 0;
          }

        } else {
          mesh->GetElementFaces(elem, elem_side_ids, orientations);
        }

        // 2b. ... and find `i` such that `elem_side_ids[i] == f`
        int i;
        for (i = 0; i < elem_side_ids.Size(); i++) {
          if (elem_side_ids[i] == f) break;
        }

        // 3. get the dofs for the entire element
        mfem::Array<int> elem_dof_ids;
        fes->GetElementDofs(elem, elem_dof_ids);

        mfem::Geometry::Type elem_geom = mesh->GetElementGeometry(elem);

        // mfem uses different conventions for boundary element orientations in 2D and 3D.
        // In 2D, mfem's official edge orientations on the boundary will always be a mix of
        // CW and CCW, so we have to discard mfem's orientation information in order
        // to get a consistent winding.
        //
        // In 3D, mfem does use a consistently CCW winding for boundary faces (I think).
        int orientation = (mesh->Dimension() == 2 && type == FaceType::BOUNDARY) ? 0 : orientations[i];

        // 4. extract only the dofs that correspond to side `i`
        for (auto k : face_perm(orientation)) {
          face_dofs.push_back(uint64_t(elem_dof_ids[local_face_dofs[uint32_t(elem_geom)](i, k)]));
        }

        // boundary faces only belong to 1 element, so we exit early
        if (type == FaceType::BOUNDARY) break;
      }

      // H1 and Hcurl spaces are more straight-forward, since
      // we can use FiniteElementSpace::GetFaceDofs() directly
    } else {
      mfem::Array<int> dofs;

      fes->GetFaceDofs(f, dofs);

      if (isHcurl(*fes)) {
        for (int k = 0; k < dofs.Size(); k++) {
          face_dofs.push_back(uint64_t(dofs[k]));
        }
      } else {
        for (int k = 0; k < dofs.Size(); k++) {
          face_dofs.push_back(uint64_t(dofs[lex_perm[uint32_t(face_geom)][uint32_t(k)]]));
        }
      }
    }

    n++;
  }

  delete face_to_elem;

  if (n == 0) {
#ifdef SERAC_USE_UMPIRE
    return axom::Array<DoF, 2, axom::MemorySpace::Host>{};
#else
    return axom::Array<DoF, 2>{};
#endif
  } else {
    uint64_t dofs_per_face = face_dofs.size() / n;
#ifdef SERAC_USE_UMPIRE
    axom::Array<DoF, 2, axom::MemorySpace::Host> output(n, dofs_per_face);
#else
    axom::Array<DoF, 2> output(n, dofs_per_face);
#endif
    std::memcpy(output.data(), face_dofs.data(), sizeof(DoF) * n * dofs_per_face);
    return output;
  }
}

namespace serac {

ElementRestriction::ElementRestriction(const mfem::FiniteElementSpace* fes, mfem::Geometry::Type elem_geom)
{
  dof_info = GetElementRestriction(fes, elem_geom);

  ordering = fes->GetOrdering();

  lsize          = uint64_t(fes->GetVSize());
  components     = uint64_t(fes->GetVDim());
  num_nodes      = lsize / components;
  num_elements   = uint64_t(dof_info.shape()[0]);
  nodes_per_elem = uint64_t(dof_info.shape()[1]);
  esize          = num_elements * nodes_per_elem * components;
}

ElementRestriction::ElementRestriction(const mfem::FiniteElementSpace* fes, mfem::Geometry::Type face_geom,
                                       FaceType type)
{
  dof_info = GetFaceDofs(fes, face_geom, type);

  ordering = fes->GetOrdering();

  lsize          = uint64_t(fes->GetVSize());
  components     = uint64_t(fes->GetVDim());
  num_nodes      = lsize / components;
  num_elements   = uint64_t(dof_info.shape()[0]);
  nodes_per_elem = uint64_t(dof_info.shape()[1]);
  esize          = num_elements * nodes_per_elem * components;
}

uint64_t ElementRestriction::ESize() const { return esize; }

uint64_t ElementRestriction::LSize() const { return lsize; }

DoF ElementRestriction::GetVDof(DoF node, uint64_t component) const
{
  if (ordering == mfem::Ordering::Type::byNODES) {
    return DoF{component * num_nodes + node.index(), (node.sign() == 1) ? 0ull : 1ull, node.orientation()};
  } else {
    return DoF{node.index() * components + component, (node.sign() == 1) ? 0ull : 1ull, node.orientation()};
  }
}

void ElementRestriction::GetElementVDofs(int i, std::vector<DoF>& vdofs) const
{
  for (uint64_t c = 0; c < components; c++) {
    for (uint64_t j = 0; j < nodes_per_elem; j++) {
      vdofs[c * nodes_per_elem + j] = GetVDof(dof_info(i, j), c);
    }
  }
}

void ElementRestriction::Gather(const mfem::Vector& L_vector, mfem::Vector& E_vector) const
{
  for (uint64_t i = 0; i < num_elements; i++) {
    for (uint64_t c = 0; c < components; c++) {
      for (uint64_t j = 0; j < nodes_per_elem; j++) {
        uint64_t E_id       = (i * components + c) * nodes_per_elem + j;
        uint64_t L_id       = GetVDof(dof_info(i, j), c).index();
        E_vector[int(E_id)] = L_vector[int(L_id)];
      }
    }
  }
}

void ElementRestriction::ScatterAdd(const mfem::Vector& E_vector, mfem::Vector& L_vector) const
{
  for (uint64_t i = 0; i < num_elements; i++) {
    for (uint64_t c = 0; c < components; c++) {
      for (uint64_t j = 0; j < nodes_per_elem; j++) {
        uint64_t E_id = (i * components + c) * nodes_per_elem + j;
        uint64_t L_id = GetVDof(dof_info(i, j), c).index();
        L_vector[int(L_id)] += E_vector[int(E_id)];
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////

BlockElementRestriction::BlockElementRestriction(const mfem::FiniteElementSpace* fes)
{
  int dim = fes->GetMesh()->Dimension();

  if (dim == 2) {
    for (auto geom : {mfem::Geometry::TRIANGLE, mfem::Geometry::SQUARE}) {
      restrictions[geom] = ElementRestriction(fes, geom);
    }
  }

  if (dim == 3) {
    for (auto geom : {mfem::Geometry::TETRAHEDRON, mfem::Geometry::CUBE}) {
      restrictions[geom] = ElementRestriction(fes, geom);
    }
  }
}

BlockElementRestriction::BlockElementRestriction(const mfem::FiniteElementSpace* fes, FaceType type)
{
  int dim = fes->GetMesh()->Dimension();

  if (dim == 2) {
    restrictions[mfem::Geometry::SEGMENT] = ElementRestriction(fes, mfem::Geometry::SEGMENT, type);
  }

  if (dim == 3) {
    for (auto geom : {mfem::Geometry::TRIANGLE, mfem::Geometry::SQUARE}) {
      restrictions[geom] = ElementRestriction(fes, geom, type);
    }
  }
}

uint64_t BlockElementRestriction::ESize() const { return (*restrictions.begin()).second.ESize(); }

uint64_t BlockElementRestriction::LSize() const { return (*restrictions.begin()).second.LSize(); }

mfem::Array<int> BlockElementRestriction::bOffsets() const
{
  mfem::Array<int> offsets(mfem::Geometry::NUM_GEOMETRIES + 1);

  offsets[0] = 0;
  for (int i = 0; i < mfem::Geometry::NUM_GEOMETRIES; i++) {
    auto g = mfem::Geometry::Type(i);
    if (restrictions.count(g) > 0) {
      offsets[g + 1] = offsets[g] + int(restrictions.at(g).ESize());
    } else {
      offsets[g + 1] = offsets[g];
    }
    // std::cout << g << " " << offsets[g+1] << std::endl;
  }
  return offsets;
};

void BlockElementRestriction::Gather(const mfem::Vector& L_vector, mfem::BlockVector& E_block_vector) const
{
  for (auto [geom, restriction] : restrictions) {
    restriction.Gather(L_vector, E_block_vector.GetBlock(geom));
  }
}

void BlockElementRestriction::ScatterAdd(const mfem::BlockVector& E_block_vector, mfem::Vector& L_vector) const
{
  for (auto [geom, restriction] : restrictions) {
    restriction.ScatterAdd(E_block_vector.GetBlock(geom), L_vector);
  }
}

}  // namespace serac
