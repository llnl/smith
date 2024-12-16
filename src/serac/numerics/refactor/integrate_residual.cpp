#include "common.hpp"

#include "refactor/domain.hpp"
#include "refactor/assert.hpp"
#include "refactor/threadpool.hpp"


#include "serac/numerics/functional/tensor.hpp"

#include "common.hpp"

#include "misc/for_constexpr.hpp"
#include "misc/timer.hpp"

namespace refactor {

namespace impl {

template < mfem::Geometry::Type geom, Family family, DerivedQuantity op >
void batched_integrate_residual(Residual & r, 
                                const Field & X,
                                const DomainType type,
                                nd::view<const double, 3> f_q,
                                nd::view<const Connection, 2> connectivity,
                                const nd::view<const int> elements,
                                const nd::view<const double, 2> xi,
                                const nd::view<const double, 1> weights,
                                const Domain::AssemblyLUT & table,
                                nd::array<double> & element_residual_buffer) {

  uint32_t num_elements = elements.size();
  if (num_elements == 0) return;

  FiniteElement< geom, family > r_el{r.space.degree};

  using input_t = std::conditional< 
    op == DerivedQuantity::VALUE, 
    typename FiniteElement< geom, family >::source_type,
    typename FiniteElement< geom, family >::flux_type
  >::type;

  nd::view<input_t, 2> input_q((input_t*)&f_q[0], {f_q.shape[0], f_q.shape[1]});

  constexpr uint32_t gdim = dimension(geom);
  uint32_t qpts_per_element = impl::qpe<geom>(xi.shape[0]);

  using A_type = decltype(weighted_piola_transformation<family, op>(mat<gdim,gdim>{}));

  uint32_t num_nodes = r.data.shape[0];
  uint32_t r_components = r.data.shape[1];
  uint32_t r_nodes_per_element = r_el.num_nodes();
  auto r_shape_fns = [&](){
    if constexpr(op == DerivedQuantity::VALUE) {
      return r_el.evaluate_weighted_shape_functions(xi, weights);
    }  

    if constexpr(op == DerivedQuantity::DERIVATIVE && family == Family::Hcurl) {
      return r_el.evaluate_weighted_shape_function_curls(xi, weights);
    }  

    if constexpr(op == DerivedQuantity::DERIVATIVE && is_scalar_valued(family)) {
      return r_el.evaluate_weighted_shape_function_gradients(xi, weights);
    }  
  }();

  FiniteElement< geom, Family::H1 > X_el{X.degree};
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



  stack::array<uint32_t, 2> element_residual_shape = {num_elements * r_nodes_per_element, r_components};
  
  if (element_residual_buffer.sz < nd::product(element_residual_shape)) {
    element_residual_buffer.resize(nd::product(element_residual_shape));
  }

  nd::view<double, 2> element_residuals(element_residual_buffer.data(), element_residual_shape);

  {
    MTR_SCOPE("integrate", "(element residuals)");

    // for each element with this geometry
    threadpool::block_parallel_for(num_elements, [&](uint32_t istart, uint32_t iend) {

      nd::array< A_type > A_q;
      nd::array< vec<gdim>, 2 > dX_dxi_q;
      nd::array<uint32_t> X_ids;
      nd::array<double> X_e;
      nd::array< double > X_scratch;

      if (need_to_compute_dX_dxi) {
        X_ids.resize(X_nodes_per_element);
        X_e.resize(X_nodes_per_element);
        X_scratch.resize({X_el.batch_interpolation_scratch_space(xi)});
        A_q.resize({qpts_per_element});
        dX_dxi_q.resize({X_components, qpts_per_element});
      }

      nd::array<uint32_t> r_ids({r_nodes_per_element});
      nd::array<double> r_e({r_nodes_per_element});
      nd::array< double > r_scratch({r_el.batch_interpolation_scratch_space(xi)});

      for (uint32_t i = istart; i < iend; i++) {

        if (need_to_compute_dX_dxi) {

          // figure out which nodal values belong to this element 
          X_el.indices(X.offsets, connectivity(elements(i)).data(), X_ids.data());

          for (int c = 0; c < X_components; c++) {
            for (int j = 0; j < X_nodes_per_element; j++) {
              X_e(j) = X.data(X_ids(j), c);
            }
            X_el.gradient(dX_dxi_q(c), X_e, X_shape_fn_grads, X_scratch.data());
          }

          for (int q = 0; q < qpts_per_element; q++) {
            mat<gdim,gdim> dX_dxi;
            for (int c = 0; c < gdim; c++) {
              dX_dxi[c] = dX_dxi_q(c, q);
            } 
            A_q[q] = weighted_piola_transformation<family, op>(dX_dxi);
          }

        }

        nd::array<input_t> input_xi_q({qpts_per_element});
        for (int c = 0; c < r_components; c++) {
          if (need_to_compute_dX_dxi) {
            for (int q = 0; q < qpts_per_element; q++) {
              input_xi_q(q) = fm::dot(A_q[q], input_q(i*qpts_per_element+q, c));
            }
          } else {
            for (int q = 0; q < qpts_per_element; q++) {
              input_xi_q(q) = input_q(i*qpts_per_element+q, c);
            }
          }

          if constexpr (op == DerivedQuantity::VALUE) {
            r_el.integrate_source(r_e, input_xi_q, r_shape_fns, r_scratch.data());
          } 

          if constexpr (op == DerivedQuantity::DERIVATIVE) {
            r_el.integrate_flux(r_e, input_xi_q, r_shape_fns, r_scratch.data());
          }

          if constexpr (is_vector_valued(family)) {
            r_el.reorient(TransformationType::TransposePhysicalToParent, &connectivity(elements(i), 0), r_e.data()); 
          }

          for (int j = 0; j < r_nodes_per_element; j++) {
            element_residuals(i * r_nodes_per_element + j, c) = r_e(j);
          }

        }

      }

    });

  }

  {
    MTR_SCOPE("integrate", "(gather)");
    threadpool::block_parallel_for(num_nodes, [&](uint32_t istart, uint32_t iend) {

      std::vector<double> sums(r_components);

      for (uint32_t i = istart; i < iend; i++) {
        uint32_t begin = table.offsets[i];
        uint32_t end = table.offsets[i+1];

        for (uint32_t c = 0; c < r_components; c++) {
          sums[c] = 0.0;
        }

        for (uint32_t k = begin; k < end; k++) {
          uint32_t id = table.ids[k];
          for (uint32_t c = 0; c < r_components; c++) {
            sums[c] += element_residuals(id, c);
          }
        }

        for (uint32_t c = 0; c < r_components; c++) {
          r.data(i, c) += sums[c];
        }
      }
    });
  }
}

template < DerivedQuantity op, uint32_t n >
void integrate_residual(Residual & r, BasisFunction phi, const nd::array<double, n> & f_q, const Domain & domain, const DomainType type) {

  MTR_SCOPE("integrate", "residual");

  zero(r.data);

  uint32_t gdim = domain.mesh.geometry_dimension;

  stack::array<uint32_t, 3> input_dimensions{
    f_q.shape[0],
    phi.space.components,
    qshape(r.space.family, op, gdim)
  };

  refactor_ASSERT(compatible_shapes(f_q.shape, input_dimensions), "incompatible array shapes");

  static nd::array< double > element_residual_buffer;

  nd::view<const double,3> f3D{f_q.data(), input_dimensions};
  uint32_t offset = 0;
  foreach_geometry([&](auto geom){
    nd::view<const int> elements = domain.active_elements[geom];
    if (gdim == dimension(geom) && elements.shape[0] > 0) {
      nd::view<const double, 2> xi = domain.rule[geom].points;
      nd::view<const double, 1> weights = domain.rule[geom].weights;
      nd::view<const Connection, 2> connectivity = domain.mesh[geom];

      nd::view<const double, 3> integrand_geom = {&f3D(offset, 0, 0), {domain.num_qpts[geom], f3D.shape[1], f3D.shape[2]}};

      const Domain::AssemblyLUT & table = domain.get(geom, phi.space.family, phi.space.degree);

      if (phi.space.family == Family::H1) {
        batched_integrate_residual<geom, Family::H1, op>(r, domain.mesh.X, type, integrand_geom, connectivity, elements, xi, weights, table, element_residual_buffer); 
      }

      if (phi.space.family == Family::Hcurl) {
        batched_integrate_residual<geom, Family::Hcurl, op>(r, domain.mesh.X, type, integrand_geom, connectivity, elements, xi, weights, table, element_residual_buffer); 
      }
    }

    offset += domain.num_qpts[geom];
  });

}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template void integrate_residual<DerivedQuantity::VALUE, 1>(Residual&, BasisFunction, const nd::array<double, 1> &, const Domain &, const DomainType);
template void integrate_residual<DerivedQuantity::VALUE, 2>(Residual&, BasisFunction, const nd::array<double, 2> &, const Domain &, const DomainType);
template void integrate_residual<DerivedQuantity::VALUE, 3>(Residual&, BasisFunction, const nd::array<double, 3> &, const Domain &, const DomainType);

template void integrate_residual<DerivedQuantity::DERIVATIVE, 1>(Residual&, BasisFunction, const nd::array<double, 1> &, const Domain &, const DomainType);
template void integrate_residual<DerivedQuantity::DERIVATIVE, 2>(Residual&, BasisFunction, const nd::array<double, 2> &, const Domain &, const DomainType);
template void integrate_residual<DerivedQuantity::DERIVATIVE, 3>(Residual&, BasisFunction, const nd::array<double, 3> &, const Domain &, const DomainType);

} // namespace impl

} // namespace refactor
