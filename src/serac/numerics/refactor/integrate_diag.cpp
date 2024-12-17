#include "common.hpp"

namespace refactor {

namespace impl {

template < mfem::Geometry::Type geom, Family family, DerivedQuantity test_op, DerivedQuantity trial_op>
void batched_integrate_diag(nd::view<double, 2> D,
                            nd::view<const double, 5> qdata,
                            const FunctionSpace space,
                            const GeometryInfo offsets,
                            const Field & X,
                            const DomainType type,
                            const nd::view<const Connection, 2> connectivity,
                            const nd::view<const int> elements,
                            const nd::view<const double, 2> xi,
                            const nd::view<const double, 1> weights,
                            const Domain::AssemblyLUT & table,
                            nd::array<double> & D_e_buffer) {

  MTR_SCOPE("integrate_spmat", "batched_dphi_dpsi_xi");

  constexpr uint32_t gdim = dimension(geom);
  constexpr uint32_t test_qshape = qshape(family, test_op, gdim);
  constexpr uint32_t trial_qshape = qshape(family, trial_op, gdim);

  using test_qtype = vec<test_qshape>;
  using trial_qtype = vec<trial_qshape>;
  using mat_t = mat<test_qshape, trial_qshape>;

  using test_Atype = decltype(piola_transformation<family, test_op>(mat<gdim,gdim>{}));
  using trial_Atype = decltype(weighted_piola_transformation<family, trial_op>(mat<gdim,gdim>{}));

  FiniteElement< geom, Family::H1 > X_el{X.degree};
  FiniteElement< geom, family > el{space.degree};

  uint32_t num_nodes = D.shape[0];
  uint32_t num_elements = elements.size();
  uint32_t num_components = space.components;
  uint32_t nodes_per_element = el.num_nodes();
  uint32_t qpts_per_element = impl::qpe<geom>(xi.shape[0]);

  uint32_t X_components = X.data.shape[1];
  uint32_t X_nodes_per_element = X_el.num_nodes();

  auto X_shape_fn_grads = X_el.evaluate_shape_function_gradients(xi);

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

  stack::array<uint32_t, 2> D_e_shape = {num_elements * nodes_per_element, num_components};
  
  if (D_e_buffer.sz < nd::product(D_e_shape)) {
    D_e_buffer.resize(nd::product(D_e_shape));
  }

  nd::view<double, 2> D_e(D_e_buffer.data(), D_e_shape);

  // for each element of this mfem::Geometry::Type in the domain
  threadpool::parallel_for(num_elements, [&](uint32_t e) {

    nd::array<test_Atype> testA_q;
    nd::array<trial_Atype> trialA_q;

    if (need_to_compute_dX_dxi) {
      nd::array<vec<gdim>, 2> dX_dxi_q({X_components, qpts_per_element});
      nd::array<uint32_t> X_ids({X_nodes_per_element});
      nd::array<double> X_e({X_nodes_per_element});
      nd::array<double> X_scratch({X_el.batch_interpolation_scratch_space(xi)});

      // figure out which nodal values belong to this element 
      X_el.indices(X.offsets, connectivity(elements(e)).data(), X_ids.data());

      for (int c = 0; c < X_components; c++) {
        for (int j = 0; j < X_nodes_per_element; j++) {
          X_e(j) = X.data(X_ids(j), c);
        }
        X_el.gradient(dX_dxi_q(c), X_e, X_shape_fn_grads, X_scratch.data());
      }

      testA_q.resize({qpts_per_element});
      trialA_q.resize({qpts_per_element});
      for (int q = 0; q < qpts_per_element; q++) {
        mat<gdim,gdim> dX_dxi;
        for (int c = 0; c < gdim; c++) {
          dX_dxi[c] = dX_dxi_q(c, q);
        } 
        testA_q[q] = piola_transformation<family, test_op>(dX_dxi);
        trialA_q[q] = weighted_piola_transformation<family, trial_op>(dX_dxi);
      }
    }

    nd::array< int8_t > transformation({nodes_per_element});
    if constexpr (is_vector_valued(family)) {
      el.reorient(TransformationType::TransposePhysicalToParent, &connectivity(elements(e), 0), transformation.data()); 
    }

    uint32_t qoffset = e * qpts_per_element;

    for (uint32_t i = 0; i < num_components; i++) {
      for (uint32_t I = 0; I < nodes_per_element; I++) {

        double sum = 0.0;

        for (uint32_t q = 0; q < qpts_per_element; q++) {
          uint32_t qid = qoffset + q;

          auto xi_q = quadrature_point<geom>(q, xi);
          double wt = integration_weight<geom>(q, weights);

          mat_t C{};
          for (uint32_t k = 0; k < test_qshape; k++) {
            for (uint32_t m = 0; m < trial_qshape; m++) {
              C(k, m) = qdata(qid, i, k, i, m);
            }
          }

          test_qtype phi_I;
          if constexpr (is_scalar_valued(family) && test_op == DerivedQuantity::VALUE) {
            phi_I = el.shape_function(xi_q, I);
          }

          if constexpr (is_scalar_valued(family) && test_op == DerivedQuantity::DERIVATIVE) {
            phi_I = el.shape_function_gradient(xi_q, I);
          }

          if constexpr (family == Family::Hcurl && test_op == DerivedQuantity::VALUE) {
            phi_I = el.reoriented_shape_function(xi_q, I, transformation[I]);
          }

          if constexpr (family == Family::Hcurl && test_op == DerivedQuantity::DERIVATIVE) {
            phi_I = el.reoriented_shape_function_curl(xi_q, I, transformation[I]);
          }

          trial_qtype psi_I;
          if constexpr (is_scalar_valued(family) && trial_op == DerivedQuantity::VALUE) {
            psi_I = el.shape_function(xi_q, I);
          }

          if constexpr (is_scalar_valued(family) && trial_op == DerivedQuantity::DERIVATIVE) {
            psi_I = el.shape_function_gradient(xi_q, I);
          }

          if constexpr (family == Family::Hcurl && trial_op == DerivedQuantity::VALUE) {
            psi_I = el.reoriented_shape_function(xi_q, I, transformation[I]);
          }

          if constexpr (family == Family::Hcurl && trial_op == DerivedQuantity::DERIVATIVE) {
            psi_I = el.reoriented_shape_function_curl(xi_q, I, transformation[I]);
          }

          if (need_to_compute_dX_dxi) {
            phi_I = fm::dot(phi_I, testA_q[q]);
            psi_I = fm::dot(psi_I, trialA_q[q]);
          }

          sum += fm::dot(fm::dot(phi_I, C), psi_I) * wt;

        }

        D_e(e * nodes_per_element + I, i) = sum;

      }
      
    }

  });

  {
    MTR_SCOPE("integrate", "(gather)");
    threadpool::block_parallel_for(num_nodes, [&](uint32_t istart, uint32_t iend) {

      std::vector<double> sums(num_components);

      for (uint32_t i = istart; i < iend; i++) {
        uint32_t begin = table.offsets[i];
        uint32_t end = table.offsets[i+1];

        for (uint32_t c = 0; c < num_components; c++) {
          sums[c] = 0.0;
        }

        for (uint32_t k = begin; k < end; k++) {
          uint32_t id = table.ids[k];
          for (uint32_t c = 0; c < num_components; c++) {
            sums[c] += D_e(id, c);
          }
        }

        for (uint32_t c = 0; c < num_components; c++) {
          D(i, c) += sums[c];
        }
      }

    });
  }

}

template <DerivedQuantity test_op, DerivedQuantity trial_op, uint32_t n>
nd::array<double,2> integrate_sparse_matrix_diagonal(BasisFunction test, const nd::array<double, n> & qdata, BasisFunction trial, const Domain &domain, const DomainType type) {

  MTR_SCOPE("integrate", "diag");

  auto phi = test.space;
  auto psi = trial.space;

  refactor_ASSERT(phi == psi, "must have matching test and trial spaces for diag(...)");

  uint32_t test_components = phi.components;
  uint32_t trial_components = psi.components;
  uint32_t sdim = domain.mesh.spatial_dimension;
  uint32_t gdim = domain.mesh.geometry_dimension;

  stack::array<uint32_t, 5> shape5D = {
    qdata.shape[0],
    phi.components, qshape(phi.family, test_op, gdim),
    psi.components, qshape(psi.family, trial_op, gdim)
  };

  refactor_ASSERT(compatible_shapes(qdata.shape, shape5D), "incompatible array shapes");

  nd::view<const double,5> q5D{qdata.data(), shape5D};

  Residual output(phi, domain.mesh);

  static nd::array< double > D_e_buffer;

  uint32_t qoffset = 0;
  foreach_geometry([&](auto geom){
    nd::view<const int> elements = domain.active_elements[geom];
    if (gdim == dimension(geom) && elements.size() > 0) {
      nd::view<const Connection, 2> connectivity = domain.mesh[geom];
      nd::view<const double, 2> xi = domain.rule[geom].points;
      nd::view<const double, 1> weights = domain.rule[geom].weights;

      stack::array< uint32_t, 5 > qdata_shape{domain.num_qpts[geom], shape5D[1], shape5D[2], shape5D[3], shape5D[4]};
      nd::view<const double, 5> geom_qdata{&q5D(qoffset, 0, 0, 0, 0), qdata_shape};

      const Domain::AssemblyLUT & table = domain.get(geom, phi.family, phi.degree);

      foreach_constexpr< Family::H1, Family::Hcurl >([&](auto family) {
        if (family == phi.family) {
          batched_integrate_diag<geom, family, test_op, trial_op >(
            output.data,
            geom_qdata, psi,
            output.offsets, domain.mesh.X, type,
            connectivity, elements, 
            xi, weights,
            table, D_e_buffer
          );
        }
      });

      qoffset += domain.num_qpts[geom];
    }
  });

  return output.data;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template nd::array<double,2> integrate_sparse_matrix_diagonal<DerivedQuantity::VALUE, DerivedQuantity::VALUE, 2>(BasisFunction, const nd::array<double, 2> &, BasisFunction, const Domain &, const DomainType);
template nd::array<double,2> integrate_sparse_matrix_diagonal<DerivedQuantity::VALUE, DerivedQuantity::VALUE, 3>(BasisFunction, const nd::array<double, 3> &, BasisFunction, const Domain &, const DomainType);
template nd::array<double,2> integrate_sparse_matrix_diagonal<DerivedQuantity::VALUE, DerivedQuantity::VALUE, 4>(BasisFunction, const nd::array<double, 4> &, BasisFunction, const Domain &, const DomainType);
template nd::array<double,2> integrate_sparse_matrix_diagonal<DerivedQuantity::VALUE, DerivedQuantity::VALUE, 5>(BasisFunction, const nd::array<double, 5> &, BasisFunction, const Domain &, const DomainType);

template nd::array<double,2> integrate_sparse_matrix_diagonal<DerivedQuantity::DERIVATIVE, DerivedQuantity::VALUE, 2>(BasisFunction, const nd::array<double, 2> &, BasisFunction, const Domain &, const DomainType);
template nd::array<double,2> integrate_sparse_matrix_diagonal<DerivedQuantity::DERIVATIVE, DerivedQuantity::VALUE, 3>(BasisFunction, const nd::array<double, 3> &, BasisFunction, const Domain &, const DomainType);
template nd::array<double,2> integrate_sparse_matrix_diagonal<DerivedQuantity::DERIVATIVE, DerivedQuantity::VALUE, 4>(BasisFunction, const nd::array<double, 4> &, BasisFunction, const Domain &, const DomainType);
template nd::array<double,2> integrate_sparse_matrix_diagonal<DerivedQuantity::DERIVATIVE, DerivedQuantity::VALUE, 5>(BasisFunction, const nd::array<double, 5> &, BasisFunction, const Domain &, const DomainType);

template nd::array<double,2> integrate_sparse_matrix_diagonal<DerivedQuantity::VALUE, DerivedQuantity::DERIVATIVE, 2>(BasisFunction, const nd::array<double, 2> &, BasisFunction, const Domain &, const DomainType);
template nd::array<double,2> integrate_sparse_matrix_diagonal<DerivedQuantity::VALUE, DerivedQuantity::DERIVATIVE, 3>(BasisFunction, const nd::array<double, 3> &, BasisFunction, const Domain &, const DomainType);
template nd::array<double,2> integrate_sparse_matrix_diagonal<DerivedQuantity::VALUE, DerivedQuantity::DERIVATIVE, 4>(BasisFunction, const nd::array<double, 4> &, BasisFunction, const Domain &, const DomainType);
template nd::array<double,2> integrate_sparse_matrix_diagonal<DerivedQuantity::VALUE, DerivedQuantity::DERIVATIVE, 5>(BasisFunction, const nd::array<double, 5> &, BasisFunction, const Domain &, const DomainType);

template nd::array<double,2> integrate_sparse_matrix_diagonal<DerivedQuantity::DERIVATIVE, DerivedQuantity::DERIVATIVE, 2>(BasisFunction, const nd::array<double, 2> &, BasisFunction, const Domain &, const DomainType);
template nd::array<double,2> integrate_sparse_matrix_diagonal<DerivedQuantity::DERIVATIVE, DerivedQuantity::DERIVATIVE, 3>(BasisFunction, const nd::array<double, 3> &, BasisFunction, const Domain &, const DomainType);
template nd::array<double,2> integrate_sparse_matrix_diagonal<DerivedQuantity::DERIVATIVE, DerivedQuantity::DERIVATIVE, 4>(BasisFunction, const nd::array<double, 4> &, BasisFunction, const Domain &, const DomainType);
template nd::array<double,2> integrate_sparse_matrix_diagonal<DerivedQuantity::DERIVATIVE, DerivedQuantity::DERIVATIVE, 5>(BasisFunction, const nd::array<double, 5> &, BasisFunction, const Domain &, const DomainType);

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

} // namespace impl

} // namespace refactor
