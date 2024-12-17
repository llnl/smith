#include "evaluate.hpp"

#include "common.hpp"

namespace refactor {

inline GeometryData< nd::range<uint32_t> > ranges(GeometryInfo a) {
  GeometryData< nd::range<uint32_t> > output;

  uint32_t total = 0;
  output.vert = nd::range{total, total + a.vert}; total += a.vert;
  output.edge = nd::range{total, total + a.edge}; total += a.edge;
  output.tri  = nd::range{total, total + a.tri};  total += a.tri;
  output.quad = nd::range{total, total + a.quad}; total += a.quad;
  output.tet  = nd::range{total, total + a.tet};  total += a.tet;
  output.hex  = nd::range{total, total + a.hex};  total += a.hex;

  return output;
};

namespace impl {

template < mfem::Geometry::Type geom, Family family, DerivedQuantity op >
void batched_interpolate(nd::view<double, 3> u_q, 
                         const Field & u,  
                         const Field & X,  
                         DomainType type,
                         const nd::view<const int> elements,
                         const nd::view<const double, 2> xi) {

  uint32_t num_elements = elements.size();
  if (num_elements == 0) return;

  FiniteElement< geom, family > u_el{get_degree(u)};

  using output_t = typename std::conditional< 
    op == DerivedQuantity::VALUE, 
    typename FiniteElement< geom, family >::source_type,
    typename FiniteElement< geom, family >::flux_type
  >::type;

  nd::view<output_t, 2> output_q(reinterpret_cast<output_t*>(&u_q[0]), {u_q.shape[0], u_q.shape[1]});

  constexpr uint32_t gdim = dimension(geom);
  constexpr uint32_t curl_components = source_shape(family, gdim);
  uint32_t qpts_per_element = impl::qpe<geom>(xi.shape[0]);

  using A_type = decltype(piola_transformation<family, op>(mat<gdim,gdim>{}));

  uint32_t u_components = get_num_components(u);
  uint32_t u_nodes_per_element = u_el.num_nodes();
  auto u_shape_fns = [&](){
    if constexpr(op == DerivedQuantity::VALUE) {
      return u_el.evaluate_shape_functions(xi);
    }  

    if constexpr(op == DerivedQuantity::DERIVATIVE && family == Family::Hcurl) {
      return u_el.evaluate_shape_function_curls(xi);
    }  

    if constexpr(op == DerivedQuantity::DERIVATIVE && is_scalar_valued(family)) {
      return u_el.evaluate_shape_function_gradients(xi);
    }  
  }();

  FiniteElement< geom, Family::H1 > X_el{get_degree(X)};
  uint32_t X_components = get_num_components(X);
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
  // |       spatial value |  ❌  |  ✅   |  ✅  | ❌ |
  // +---------------------+------+-------+------+----+
  // |       spatial deriv |  ✅  |  ✅   |  ✅  | ✅ |
  // +---------------------+------+-------+------+----+
  const bool need_to_compute_dX_dxi = 
    (type == DomainType::SPATIAL && op == DerivedQuantity::DERIVATIVE) || 
    (type == DomainType::SPATIAL && is_vector_valued(family));

  nd::array< A_type > A_q;
  nd::array<double> X_e;
  nd::array<uint32_t> X_ids;
  nd::array< double > X_scratch;
  nd::array< vec<gdim>, 2 > dX_dxi_q;

  if (need_to_compute_dX_dxi) {
    A_q.resize({qpts_per_element});
    X_e.resize(X_nodes_per_element);
    X_ids.resize(X_nodes_per_element);
    X_scratch.resize({X_el.batch_interpolation_scratch_space(xi)});
    dX_dxi_q.resize({X_components, qpts_per_element});
  }

  // for each element with this geometry
  nd::array<uint32_t> u_ids({u_nodes_per_element});
  nd::array<double> u_e({u_nodes_per_element});
  nd::array< double > u_scratch({u_el.batch_interpolation_scratch_space(xi)});

  for (uint32_t i = 0; i < num_elements; i++) {

    if (need_to_compute_dX_dxi) {

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
        A_q[q] = piola_transformation<family, op>(dX_dxi);
      }

    }

    nd::array<output_t> u_xi_q({qpts_per_element});
    for (uint32_t c = 0; c < u_components; c++) {

      for (uint32_t j = 0; j < u_nodes_per_element; j++) {
        u_e(j) = u.data(u_ids(j), c);
      }

      if constexpr (is_vector_valued(family)) {
        u_el.reorient(TransformationType::PhysicalToParent, &connectivity(elements(i), 0), u_e.data()); 
      }

      // carry out the appropriate kind of interpolation
      // for the requested family and differential operator
      if constexpr (op == DerivedQuantity::VALUE) {
        u_el.interpolate(u_xi_q, u_e, u_shape_fns, u_scratch.data());
      } 

      if constexpr (op == DerivedQuantity::DERIVATIVE && is_scalar_valued(family)) {
        u_el.gradient(u_xi_q, u_e, u_shape_fns, u_scratch.data());
      }

      if constexpr (op == DerivedQuantity::DERIVATIVE && family == Family::Hcurl) {
        u_el.curl(u_xi_q, u_e, u_shape_fns, u_scratch.data());
      } 

      if (need_to_compute_dX_dxi) {
        for (uint32_t q = 0; q < qpts_per_element; q++) {
          output_q(i*qpts_per_element+q, c) = serac::dot(u_xi_q[q], A_q[q]);
        }
      } else {
        for (uint32_t q = 0; q < qpts_per_element; q++) {
          output_q(i*qpts_per_element+q, c) = u_xi_q[q];
        }
      }

    }

  }

}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template < Family family, DerivedQuantity op >
void evaluate(nd::array<double,3> & output, const Field & u, const Domain & domain, const DomainType & type) {

  uint32_t gdim = static_cast<uint32_t>(domain.dim_);
  uint32_t num_components = output.shape[1];

  auto qranges = ranges(domain.num_qpts);

  uint32_t offset = 0;
  foreach_geometry([&](auto geom){
    nd::view<const int> elements = domain.active_elements[geom];
    if (gdim == dimension(geom) && elements.size() > 0) {
      nd::view<const double, 2> xi = domain.rule[geom].points;
      nd::range all{0u, num_components};
      impl::batched_interpolate<geom, family, op>(output(qranges[geom]), u, domain.mesh.X, type, elements, xi); 
    }
  });

}

} // namespace impl

nd::array<double,3> evaluate(const FieldOp && input, const DomainWithType & d) { 

  const Domain & domain = d.domain; 
  const DomainType & type = d.type; 

  Family f = get_family(input.field);

  uint32_t gdim = geometry_dimension(domain);
  uint32_t num_components = get_num_components(input.field);
  stack::array<uint32_t, 3> output_dimensions{
    total(domain.num_qpts),
    num_components,
    qshape(f, input.op, gdim)
  };

  nd::array<double, 3> u_q(output_dimensions);

  uint32_t offset = 0;
  for_constexpr< Family::H1, Family::Hcurl >([&](auto family) {
    if (family == f) {
      if (input.op == DerivedQuantity::VALUE) {
        impl::evaluate<family, DerivedQuantity::VALUE>(u_q, input.field, domain, type);
      }
      if (input.op == DerivedQuantity::DERIVATIVE) {
        impl::evaluate<family, DerivedQuantity::DERIVATIVE>(u_q, input.field, domain, type);
      }
    }
  });

  return u_q;
  
}

} // namespace refactor
