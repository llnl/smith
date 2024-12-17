#include "common.hpp"

namespace refactor {

namespace impl {

template < mfem::Geometry::Type geom, Family test_family, DerivedQuantity test_op, Family trial_family, DerivedQuantity trial_op>
void batched_integrate_spmat(nd::view<double> values,
                             nd::view<const int> row_ptr,
                             nd::view<const int> col_ind,
                             nd::view<const double, 5> qdata,
                             FunctionSpace trial_space,
                             FunctionSpace test_space,
                             GeometryInfo trial_offsets,
                             GeometryInfo test_offsets,
                             const Field & X,
                             const DomainType type,
                             nd::view<const Connection, 2> connectivity,
                             const nd::view<const int> elements,
                             const nd::view<const double, 2> xi,
                             const nd::view<const double, 1> weights) {

  constexpr uint32_t gdim = dimension(geom);
  constexpr uint32_t test_qshape = qshape(test_family, test_op, gdim);
  constexpr uint32_t trial_qshape = qshape(trial_family, trial_op, gdim);

  using test_qtype = vec<test_qshape>;
  using trial_qtype = vec<trial_qshape>;
  using mat_t = mat<test_qshape, trial_qshape>;

  using test_Atype = decltype(piola_transformation<test_family, test_op>(mat<gdim,gdim>{}));
  using trial_Atype = decltype(weighted_piola_transformation<trial_family, trial_op>(mat<gdim,gdim>{}));

  FiniteElement< geom, Family::H1 > X_el{get_degree(X)};
  FiniteElement< geom, test_family > test_el{test_space.degree};
  FiniteElement< geom, trial_family > trial_el{trial_space.degree};

  uint32_t num_elements = elements.size();
  uint32_t test_components = test_space.components;
  uint32_t trial_components = trial_space.components;
  uint32_t test_nodes_per_element = test_el.num_nodes();
  uint32_t trial_nodes_per_element = trial_el.num_nodes();
  uint32_t qpts_per_element = impl::qpe<geom>(xi.shape[0]);

  // precalculate test functions for the provided quadrature rule
  auto psi_wt = [&](){
    if constexpr(trial_op == DerivedQuantity::VALUE) {
      return trial_el.evaluate_weighted_shape_functions(xi, weights);
    }  

    if constexpr(trial_op == DerivedQuantity::DERIVATIVE && trial_family == Family::Hcurl) {
      return trial_el.evaluate_weighted_shape_function_curls(xi, weights);
    }  

    if constexpr(trial_op == DerivedQuantity::DERIVATIVE && is_scalar_valued(trial_family)) {
      return trial_el.evaluate_weighted_shape_function_gradients(xi, weights);
    }  
  }();

  uint32_t X_components = get_num_components(X);
  uint32_t X_nodes_per_element = X_el.num_nodes();

  auto X_shape_fn_grads = X_el.evaluate_shape_function_gradients(xi);

  nd::array<test_Atype> testA_q;
  nd::array<trial_Atype> trialA_q;
  nd::array<vec<gdim>, 2> dX_dxi_q;
  nd::array<uint32_t> X_ids;
  nd::array<double> X_e;
  nd::array<double> X_scratch;

  //       When do we need to calculate dX/dxi?
  // 
  //     ❌ : don't need to          ✅ : need to 
  // +---------------------+------+-------+------+----+
  // |                     |  H1  | Hcurl | Hdiv | DG |
  // +---------------------+------+-------+------+----+
  // | isoparametric value |  ❌  |  ❌   |  ❌  | ❌ |
  // +---------------------+------+-------+------+----+
  // | isoparametric deriv |  ❌  |  ❌   |  ❌  | ❌ |
  // +---------------------+------+-------+------+----+
  // |       spatial value |  ✅  |  ✅   |  ✅  | ✅ |
  // +---------------------+------+-------+------+----+
  // |       spatial deriv |  ✅  |  ✅   |  ✅  | ✅ |
  // +---------------------+------+-------+------+----+
  const bool need_to_compute_dX_dxi = (type == DomainType::SPATIAL);

  if (need_to_compute_dX_dxi) {
    testA_q.resize({qpts_per_element});
    trialA_q.resize({qpts_per_element});
    dX_dxi_q.resize({X_components, qpts_per_element});
    X_ids.resize(X_nodes_per_element);
    X_e.resize(X_nodes_per_element);
    X_scratch.resize({X_el.batch_interpolation_scratch_space(xi)});
  }

  // for each element of this mfem::Geometry::Type in the domain
  for (uint32_t e = 0; e < num_elements; e++) {

    if (need_to_compute_dX_dxi) {

      // figure out which nodal values belong to this element 
      X_el.indices(X.offsets, connectivity(elements(e)).data(), X_ids.data());

      for (uint32_t c = 0; c < X_components; c++) {
        for (uint32_t j = 0; j < X_nodes_per_element; j++) {
          X_e(j) = X.data(X_ids(j), c);
        }
        X_el.gradient(dX_dxi_q(c), X_e, X_shape_fn_grads, X_scratch.data());
      }

      for (uint32_t q = 0; q < qpts_per_element; q++) {
        mat<gdim,gdim> dX_dxi;
        for (uint32_t c = 0; c < gdim; c++) {
          dX_dxi[c] = dX_dxi_q(c, q);
        } 
        testA_q[q] = piola_transformation<test_family, test_op>(dX_dxi);
        trialA_q[q] = weighted_piola_transformation<trial_family, trial_op>(dX_dxi);
      }
    }

    nd::array< double > trial_scratch({trial_el.batch_interpolation_scratch_space(xi)});

    nd::array< trial_qtype > hat_f({qpts_per_element});

    nd::array<uint32_t> test_ids({test_nodes_per_element});
    test_el.indices(test_offsets, connectivity(elements(e)).data(), test_ids.data());

    nd::array<uint32_t> trial_ids({trial_nodes_per_element});
    trial_el.indices(trial_offsets, connectivity(elements(e)).data(), trial_ids.data());

    nd::array< int8_t > transformation({test_nodes_per_element});
    if constexpr (is_vector_valued(test_family)) {
      test_el.reorient(TransformationType::TransposePhysicalToParent, &connectivity(elements(e), 0), transformation.data()); 
    }

    uint32_t qoffset = e * qpts_per_element;

    for (uint32_t i = 0; i < test_components; i++) {
      for (uint32_t j = 0; j < trial_components; j++) {
        for (uint32_t I = 0; I < test_nodes_per_element; I++) {
          uint32_t row_id = test_ids[I] * test_components + i;

          for (uint32_t q = 0; q < qpts_per_element; q++) {
            uint32_t qid = qoffset + q;

            auto xi_q = quadrature_point<geom>(q, xi);

            mat_t C{};
            for (uint32_t k = 0; k < test_qshape; k++) {
              for (uint32_t m = 0; m < trial_qshape; m++) {
                C(k, m) = qdata(qid, i, k, j, m);
              }
            }

            test_qtype phi_I;

            if constexpr (is_scalar_valued(test_family) && test_op == DerivedQuantity::VALUE) {
              phi_I = test_el.shape_function(xi_q, I);
            }

            if constexpr (is_scalar_valued(test_family) && test_op == DerivedQuantity::DERIVATIVE) {
              phi_I = test_el.shape_function_gradient(xi_q, I);
            }

            if constexpr (test_family == Family::Hcurl && test_op == DerivedQuantity::VALUE) {
              phi_I = test_el.reoriented_shape_function(xi_q, I, transformation[I]);
            }

            if constexpr (test_family == Family::Hcurl && test_op == DerivedQuantity::DERIVATIVE) {
              phi_I = test_el.reoriented_shape_function_curl(xi_q, I, transformation[I]);
            }

            if (need_to_compute_dX_dxi) {
              phi_I = serac::dot(phi_I, testA_q[q]);
            }

            hat_f[q] = serac::dot(phi_I, C);

            if (need_to_compute_dX_dxi) {
              hat_f[q] = serac::dot(trialA_q[q], hat_f[q]);
            }

          }

          nd::array<double> r_e({trial_el.num_nodes()});

          if constexpr (trial_op == DerivedQuantity::VALUE) {
            trial_el.integrate_source(r_e, hat_f, psi_wt, trial_scratch.data());
          } 

          if constexpr (trial_op == DerivedQuantity::DERIVATIVE) {
            trial_el.integrate_flux(r_e, hat_f, psi_wt, trial_scratch.data());
          }

          if constexpr (is_vector_valued(trial_family)) {
            trial_el.reorient(TransformationType::TransposePhysicalToParent, &connectivity(elements(e), 0), r_e.data()); 
          }

          int row_start = row_ptr[row_id];
          int row_end = row_ptr[row_id+1];
          for (uint32_t J = 0; J < trial_nodes_per_element; J++) {
            int col_id = static_cast<int>(trial_ids[J] * trial_components + j);

            // find the position of the nonzero entry of this row with the right column
            int position = std::lower_bound(&col_ind[row_start], &col_ind[row_end], col_id) - &col_ind[0];

            values[position] += r_e(J);
          }
        }
      }
    }

  }

}

template <DerivedQuantity test_op, DerivedQuantity trial_op, uint32_t n>
std::function< void(refactor::sparse_matrix&) > integrate_sparse_matrix(BasisFunction test, const nd::array<double, n> & qdata, BasisFunction trial, const Domain &domain, const DomainType type) {

  return [&, type, test, trial](refactor::sparse_matrix & A) {

    auto phi = test.space;
    auto psi = trial.space;

    uint32_t test_components = phi.components;
    uint32_t trial_components = psi.components;
    uint32_t sdim = spatial_dimension(domain);
    uint32_t gdim = geometry_dimension(domain);

    stack::array<uint32_t, 5> shape5D = {
      qdata.shape[0],
      phi.components, qshape(phi.family, test_op, gdim),
      psi.components, qshape(psi.family, trial_op, gdim)
    };

    SLIC_ASSERT_MSG(compatible_shapes(qdata.shape, shape5D), "incompatible array shapes");

    nd::view<const double,5> q5D{qdata.data(), shape5D};

    GeometryInfo counts = domain.mesh.geometry_counts();
    GeometryInfo test_offsets = scan(interior_nodes_per_geom(phi) * counts);
    GeometryInfo trial_offsets = scan(interior_nodes_per_geom(psi) * counts);
    auto nrows = total(interior_dofs_per_geom(phi) * counts);
    auto ncols = total(interior_dofs_per_geom(psi) * counts);

    if (A.nnz == 0) {
      A = blank_sparse_matrix(test, trial, domain);
    } else {
      zero(A.values);
    }

    uint32_t qoffset = 0;
    foreach_geometry([&](auto geom){
      nd::view<const int> elements = domain.active_elements[geom];
      if (gdim == dimension(geom) && elements.size() > 0) {
        nd::view<const Connection, 2> connectivity = domain.mesh[geom];
        nd::view<const double, 2> xi = domain.rule[geom].points;
        nd::view<const double, 1> weights = domain.rule[geom].weights;

        if constexpr (geom != mfem::Geometry::POINT) {
          stack::array< uint32_t, 5 > qdata_shape{domain.num_qpts[geom], shape5D[1], shape5D[2], shape5D[3], shape5D[4]};
          nd::view<const double, 5> geom_qdata{&q5D(qoffset, 0, 0, 0, 0), qdata_shape};

          foreach_constexpr< Family::H1, Family::Hcurl >([&](auto test_family) {
            foreach_constexpr< Family::H1, Family::Hcurl >([&](auto trial_family) {
              if (test_family == phi.family && trial_family == psi.family) {
                batched_integrate_spmat<geom, test_family, test_op, trial_family, trial_op >(
                  A.values, A.row_ptr, A.col_ind, 
                  geom_qdata, psi, phi, 
                  trial_offsets, test_offsets, domain.mesh.X, type,
                  connectivity, elements, 
                  xi, weights
                );
              }
            });
          });

          qoffset += domain.num_qpts[geom];
        }
      }
    });

    return A;

  };
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template std::function< void(refactor::sparse_matrix&) > integrate_sparse_matrix<DerivedQuantity::VALUE, DerivedQuantity::VALUE, 2>(BasisFunction, const nd::array<double, 2> &, BasisFunction, const Domain &, const DomainType);
template std::function< void(refactor::sparse_matrix&) > integrate_sparse_matrix<DerivedQuantity::VALUE, DerivedQuantity::VALUE, 3>(BasisFunction, const nd::array<double, 3> &, BasisFunction, const Domain &, const DomainType);
template std::function< void(refactor::sparse_matrix&) > integrate_sparse_matrix<DerivedQuantity::VALUE, DerivedQuantity::VALUE, 4>(BasisFunction, const nd::array<double, 4> &, BasisFunction, const Domain &, const DomainType);
template std::function< void(refactor::sparse_matrix&) > integrate_sparse_matrix<DerivedQuantity::VALUE, DerivedQuantity::VALUE, 5>(BasisFunction, const nd::array<double, 5> &, BasisFunction, const Domain &, const DomainType);

template std::function< void(refactor::sparse_matrix&) > integrate_sparse_matrix<DerivedQuantity::DERIVATIVE, DerivedQuantity::VALUE, 2>(BasisFunction, const nd::array<double, 2> &, BasisFunction, const Domain &, const DomainType);
template std::function< void(refactor::sparse_matrix&) > integrate_sparse_matrix<DerivedQuantity::DERIVATIVE, DerivedQuantity::VALUE, 3>(BasisFunction, const nd::array<double, 3> &, BasisFunction, const Domain &, const DomainType);
template std::function< void(refactor::sparse_matrix&) > integrate_sparse_matrix<DerivedQuantity::DERIVATIVE, DerivedQuantity::VALUE, 4>(BasisFunction, const nd::array<double, 4> &, BasisFunction, const Domain &, const DomainType);
template std::function< void(refactor::sparse_matrix&) > integrate_sparse_matrix<DerivedQuantity::DERIVATIVE, DerivedQuantity::VALUE, 5>(BasisFunction, const nd::array<double, 5> &, BasisFunction, const Domain &, const DomainType);

template std::function< void(refactor::sparse_matrix&) > integrate_sparse_matrix<DerivedQuantity::VALUE, DerivedQuantity::DERIVATIVE, 2>(BasisFunction, const nd::array<double, 2> &, BasisFunction, const Domain &, const DomainType);
template std::function< void(refactor::sparse_matrix&) > integrate_sparse_matrix<DerivedQuantity::VALUE, DerivedQuantity::DERIVATIVE, 3>(BasisFunction, const nd::array<double, 3> &, BasisFunction, const Domain &, const DomainType);
template std::function< void(refactor::sparse_matrix&) > integrate_sparse_matrix<DerivedQuantity::VALUE, DerivedQuantity::DERIVATIVE, 4>(BasisFunction, const nd::array<double, 4> &, BasisFunction, const Domain &, const DomainType);
template std::function< void(refactor::sparse_matrix&) > integrate_sparse_matrix<DerivedQuantity::VALUE, DerivedQuantity::DERIVATIVE, 5>(BasisFunction, const nd::array<double, 5> &, BasisFunction, const Domain &, const DomainType);

template std::function< void(refactor::sparse_matrix&) > integrate_sparse_matrix<DerivedQuantity::DERIVATIVE, DerivedQuantity::DERIVATIVE, 2>(BasisFunction, const nd::array<double, 2> &, BasisFunction, const Domain &, const DomainType);
template std::function< void(refactor::sparse_matrix&) > integrate_sparse_matrix<DerivedQuantity::DERIVATIVE, DerivedQuantity::DERIVATIVE, 3>(BasisFunction, const nd::array<double, 3> &, BasisFunction, const Domain &, const DomainType);
template std::function< void(refactor::sparse_matrix&) > integrate_sparse_matrix<DerivedQuantity::DERIVATIVE, DerivedQuantity::DERIVATIVE, 4>(BasisFunction, const nd::array<double, 4> &, BasisFunction, const Domain &, const DomainType);
template std::function< void(refactor::sparse_matrix&) > integrate_sparse_matrix<DerivedQuantity::DERIVATIVE, DerivedQuantity::DERIVATIVE, 5>(BasisFunction, const nd::array<double, 5> &, BasisFunction, const Domain &, const DomainType);

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

} // namespace impl

} // namespace refactor
