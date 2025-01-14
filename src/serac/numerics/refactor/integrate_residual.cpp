#include "serac/numerics/refactor/integrate.hpp"

#include "serac/numerics/refactor/common.hpp"

namespace refactor {

namespace impl {

template < mfem::Geometry::Type geom, Family family, DerivedQuantity op >
void batched_integrate_residual(Residual & r, 
                                const Field & X,
                                const DomainType type,
                                nd::view<const double, 3> f_q,
                                const nd::view<const int> elements,
                                const nd::view<const double, 2> xi,
                                const nd::view<const double, 1> weights,
                                nd::array<double> & element_residual_buffer) {

  uint32_t num_elements = elements.size();
  if (num_elements == 0) return;

  FiniteElement< geom, family > r_el{get_degree(r)};

  using input_t = typename std::conditional< 
    op == DerivedQuantity::VALUE, 
    typename FiniteElement< geom, family >::source_type,
    typename FiniteElement< geom, family >::flux_type
  >::type;

  nd::view<input_t, 2> input_q(reinterpret_cast<input_t*>(&f_q[0]), {f_q.shape[0], f_q.shape[1]});

  uint32_t qpts_per_element = impl::qpe<geom>(xi.shape[0]);

  uint32_t r_components = get_num_components(r);
  uint32_t r_nodes_per_element = r_el.num_nodes();
  auto r_shape_fns = [&](){
    if constexpr(op == DerivedQuantity::VALUE) {
      return r_el.evaluate_weighted_shape_functions(xi, weights);
    }  

    if constexpr(op == DerivedQuantity::DERIVATIVE && family == Family::HCURL) {
      return r_el.evaluate_weighted_shape_function_curls(xi, weights);
    }  

    if constexpr(op == DerivedQuantity::DERIVATIVE && is_scalar_valued(family)) {
      return r_el.evaluate_weighted_shape_function_gradients(xi, weights);
    }  
  }();

  stack::array<uint32_t, 2> element_residual_shape = {num_elements * r_nodes_per_element, r_components};
  
  if (element_residual_buffer.sz < nd::product(element_residual_shape)) {
    element_residual_buffer.resize(nd::product(element_residual_shape));
  }

  nd::view<double, 2> element_residuals(element_residual_buffer.data(), element_residual_shape);

  // for each element with this geometry
  for (uint32_t i = 0; i < num_elements; i++) {

    nd::array<uint32_t> r_ids({r_nodes_per_element});
    nd::array<double> r_e({r_nodes_per_element});
    nd::array< double > r_scratch({r_el.batch_interpolation_scratch_space(xi)});

    nd::array<input_t> input_xi_q({qpts_per_element});
    for (uint32_t c = 0; c < r_components; c++) {

      for (uint32_t q = 0; q < qpts_per_element; q++) {
        input_xi_q(q) = input_q(i*qpts_per_element+q, c);
      }

      if constexpr (op == DerivedQuantity::VALUE) {
        r_el.integrate_source(r_e, input_xi_q, r_shape_fns, r_scratch.data());
      } 

      if constexpr (op == DerivedQuantity::DERIVATIVE) {
        r_el.integrate_flux(r_e, input_xi_q, r_shape_fns, r_scratch.data());
      }

      //if constexpr (is_vector_valued(family)) {
      //  r_el.reorient(TransformationType::TransposePhysicalToParent, &connectivity(elements(i), 0), r_e.data()); 
      //}

      for (uint32_t j = 0; j < r_nodes_per_element; j++) {
        element_residuals(i * r_nodes_per_element + j, c) = r_e(j);
      }

    }

  }

}

template < DerivedQuantity op, uint32_t n >
void integrate_residual(Residual & r, BasisFunction phi, const nd::array<double, n> & f_q, Domain & domain, const MeshQuadratureRule & qrule) {

  r = 0.0;

  uint32_t gdim = geometry_dimension(domain);

  stack::array<uint32_t, 3> input_dimensions{
    f_q.shape[0],
    phi.space.components,
    qshape(get_family(r), op, gdim)
  };

  GeometryInfo num_qpts = qrule.num_qpts(domain);

  SLIC_ASSERT_MSG(compatible_shapes(f_q.shape, input_dimensions), "incompatible array shapes");
  SLIC_ASSERT_MSG(input_dimensions[0] == total(num_qpts), "wrong number of quadrature points");

  static nd::array< double > element_residual_buffer;

  nd::view<const double,3> f3D{f_q.data(), input_dimensions};
  uint32_t offset = 0;
  foreach_geometry([&](auto geom){
    const std::vector<int> & elements = domain.get(geom);
    if (gdim == dimension(geom) && elements.size() > 0) {
      nd::view<const double, 2> xi = qrule[geom].points;
      nd::view<const double, 1> weights = qrule[geom].weights;

      nd::view<const double, 3> integrand_geom = {&f3D(offset, 0, 0), {num_qpts[geom], f3D.shape[1], f3D.shape[2]}};

      if (phi.space.family == Family::H1) {
        batched_integrate_residual<geom, Family::H1, op>(r, integrand_geom, elements, xi, weights, element_residual_buffer); 
      }

      if (phi.space.family == Family::HCURL) {
        batched_integrate_residual<geom, Family::HCURL, op>(r, integrand_geom, elements, xi, weights, element_residual_buffer); 
      }
    }

    offset += num_qpts[geom];
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
