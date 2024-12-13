#include "common.hpp"

#include "refactor/domain.hpp"
#include "refactor/assert.hpp"
#include "refactor/threadpool.hpp"

#include "misc/timer.hpp"

#include "parallel_hashmap/phmap.h"

namespace refactor {

uint32_t elements_per_block(mfem::Geometry::Type geom, Family family, int p) {

  uint32_t gid = uint32_t(geom);
  uint32_t fid = uint32_t(family);

  uint32_t values[6][2][4] = {

    // vertex
    {{{}}},

    // edge
    {
      {0, 32, 16, 16}, // H1
      {0, 1, 1, 1}, // Hcurl
    },

    // triangle
    {
      {0, 32, 16, 16}, // H1
      {0, 1, 1, 1}, // Hcurl
    },

    // quadrilateral
    {
      {0, 32, 14, 8}, // H1
      {0, 1, 1, 1}, // Hcurl
    },

    // tetrahedron
    {
      {0, 12, 8, 8}, // H1
      {0,  1, 1, 1}, // Hcurl
    },

    // hexahedron
    {
      {0, 8, 4, 2}, // H1
      {0, 1, 1, 1}, // Hcurl
    },
  };

  return values[gid][fid][p];

}

template < mfem::Geometry::Type geom, Family test_family, Family trial_family >
void sparsity_pattern(std::vector< phmap::flat_hash_set< int > > & column_ids,
                      FunctionSpace trial_space,
                      FunctionSpace test_space,
                      GeometryInfo trial_offsets,
                      GeometryInfo test_offsets,
                      nd::view<const int> elements,
                      nd::view<const Connection, 2> connectivity) {

  FiniteElement< geom, test_family > test_el{test_space.degree};
  FiniteElement< geom, trial_family > trial_el{trial_space.degree};

  // allocate storage for an element's nodal forces
  constexpr uint32_t gdim = dimension(geom);

  uint32_t num_elements = elements.shape[0];
  uint32_t test_components = test_space.components;
  uint32_t trial_components = trial_space.components;
  uint32_t nodes_per_test_element = test_el.num_nodes();
  uint32_t nodes_per_trial_element = trial_el.num_nodes();

  constexpr int nmutex = 1024;
  std::vector< std::mutex > mutexes(nmutex);

  threadpool::parallel_for(num_elements, [&](uint32_t e){
    nd::array<uint32_t> test_ids({nodes_per_test_element});
    nd::array<uint32_t> trial_ids({nodes_per_trial_element});

    // get the ids of nodes for that element
    test_el.indices(test_offsets, connectivity(elements(e)).data(), test_ids.data());
    trial_el.indices(trial_offsets, connectivity(elements(e)).data(), trial_ids.data());

    for (int i = 0; i < test_ids.shape[0]; i++) {
      for (int ci = 0; ci < test_components; ci++) {
        int row_id = test_ids[i] * test_components + ci;

        int which = row_id % nmutex;
        mutexes[which].lock();
        auto & row = column_ids[row_id];
        for (int j = 0; j < trial_ids.shape[0]; j++) {
          for (int cj = 0; cj < trial_components; cj++) {
            row.insert(trial_ids[j] * trial_components + cj);
          }
        }
        mutexes[which].unlock();
      }
    }
  });

}

refactor::sparse_matrix blank_sparse_matrix(BasisFunctionOp test, BasisFunctionOp trial, const Domain &domain) {

  MTR_SCOPE("integrate_spmat", "blank_sparse_matrix");

  auto phi = test.function.space;
  auto psi = trial.function.space;

  uint32_t test_components = phi.components;
  uint32_t trial_components = psi.components;
  uint32_t sdim = domain.mesh.spatial_dimension;
  uint32_t gdim = domain.mesh.geometry_dimension;

  auto dofs_per_psi = dofs_per_geom(psi);
  auto dofs_per_phi = dofs_per_geom(phi);

  GeometryInfo counts = domain.mesh.geometry_counts();
  GeometryInfo test_offsets = scan(interior_nodes_per_geom(phi) * counts);
  GeometryInfo trial_offsets = scan(interior_nodes_per_geom(psi) * counts);
  auto nrows = total(interior_dofs_per_geom(phi) * counts);
  auto ncols = total(interior_dofs_per_geom(psi) * counts);

  std::vector< phmap::flat_hash_set<int> > column_ids(nrows); 
  foreach_geometry([&](auto geom){
    nd::view<const int> elements = domain.active_elements[geom];
    if (gdim == dimension(geom) && elements.size() > 0) {
      nd::view<const Connection, 2> connectivity = domain.mesh[geom];
      foreach_constexpr< Family::H1, Family::Hcurl >([&](auto test_family) {
        foreach_constexpr< Family::H1, Family::Hcurl >([&](auto trial_family) {
          if (test_family == phi.family && trial_family == psi.family) {
            sparsity_pattern<geom, test_family, trial_family>(column_ids, psi, phi, trial_offsets, test_offsets, elements, connectivity);
          }
        });
      });
    }
  });

  refactor::sparse_matrix A;
  A.nrows = nrows;
  A.ncols = ncols;
  A.nnz = 0;
  A.row_ptr.resize(nrows + 1);
  A.row_ptr[0] = 0;

  for (int i = 0; i < nrows; i++) {
    int nz_per_row = column_ids[i].size();
    A.nnz += nz_per_row;
    A.row_ptr[i+1] = A.row_ptr[i] + nz_per_row;
  }
  A.col_ind.resize(A.nnz);
  A.values.resize(A.nnz);

  threadpool::parallel_for(nrows, [&](int i){
    int offset = A.row_ptr[i];
    for (int col : column_ids[i]) {
      A.col_ind[offset++] = col;
    }
    std::sort(&A.col_ind[A.row_ptr[i]], &A.col_ind[A.row_ptr[i+1]]);
  });

  return A;
}

} // namespace refactor
