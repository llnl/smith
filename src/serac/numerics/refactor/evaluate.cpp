#if 1
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
                         const double * u,  
                         uint32_t u_degree,
                         uint32_t u_components,
                         const std::vector<int> & elements,
                         const nd::view<const double, 2> xi) {

  uint32_t num_elements = uint32_t(elements.size());
  if (num_elements == 0) return;

  FiniteElement< geom, family > u_el{u_degree};

  using output_t = typename std::conditional< 
    op == DerivedQuantity::VALUE, 
    typename FiniteElement< geom, family >::source_type,
    typename FiniteElement< geom, family >::flux_type
  >::type;

  nd::view<output_t, 2> output_q(reinterpret_cast<output_t*>(&u_q[0]), {u_q.shape[0], u_q.shape[1]});

  uint32_t qpts_per_element = impl::qpe<geom>(xi.shape[0]);

  uint32_t u_nodes_per_element = u_el.num_nodes();
  uint32_t u_dofs_per_element = u_nodes_per_element * u_components;
  auto u_shape_fns = [&](){
    if constexpr(op == DerivedQuantity::VALUE) {
      return u_el.evaluate_shape_functions(xi);
    }  

    if constexpr(op == DerivedQuantity::DERIVATIVE && family == Family::HCURL) {
      return u_el.evaluate_shape_function_curls(xi);
    }  

    if constexpr(op == DerivedQuantity::DERIVATIVE && is_scalar_valued(family)) {
      return u_el.evaluate_shape_function_gradients(xi);
    }  
  }();

  // for each element with this geometry
  nd::array<uint32_t> u_ids({u_nodes_per_element});
  nd::array<double> u_e({u_nodes_per_element});
  nd::array< double > u_scratch({u_el.batch_interpolation_scratch_space(xi)});

  for (uint32_t i = 0; i < num_elements; i++) {

    nd::array<output_t> u_xi_q({qpts_per_element});
    for (uint32_t c = 0; c < u_components; c++) {

      for (uint32_t j = 0; j < u_nodes_per_element; j++) {
        u_e(j) = u[i * u_dofs_per_element + c * u_nodes_per_element + j];
      }

      //if constexpr (is_vector_valued(family)) {
      //  u_el.reorient(TransformationType::PhysicalToParent, &connectivity(elements(i), 0), u_e.data()); 
      //}

      // carry out the appropriate kind of interpolation
      // for the requested family and differential operator
      if constexpr (op == DerivedQuantity::VALUE) {
        u_el.interpolate(u_xi_q, u_e, u_shape_fns, u_scratch.data());
      } 

      if constexpr (op == DerivedQuantity::DERIVATIVE && is_scalar_valued(family)) {
        u_el.gradient(u_xi_q, u_e, u_shape_fns, u_scratch.data());
      }

      if constexpr (op == DerivedQuantity::DERIVATIVE && family == Family::HCURL) {
        u_el.curl(u_xi_q, u_e, u_shape_fns, u_scratch.data());
      } 

      for (uint32_t q = 0; q < qpts_per_element; q++) {
        output_q(i*qpts_per_element+q, c) = u_xi_q[q];
      }

    }

  }

}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template < Family family, DerivedQuantity op >
void evaluate(nd::array<double,3> & output, const Field & u_T, Domain & domain, const MeshQuadratureRule & qrule) {

  uint32_t gdim = static_cast<uint32_t>(domain.dim_);

  auto qranges = ranges(qrule.num_qpts(domain));

  uint32_t u_degree = get_degree(u_T);
  uint32_t u_components = get_num_components(u_T);
  mfem::FiniteElementSpace * u_fes = get_FES(u_T);
  const mfem::Operator * P = u_fes->GetProlongationMatrix();

  static mfem::Vector u_L;
  u_L.SetSize(P->Height(), mfem::Device::GetMemoryType());
  P->Mult(u_T, u_L);

  // we just use this as a key to create/fetch the appropriate restriction operator
  serac::FunctionSpace space{get_family(u_T), int(get_degree(u_T)), int(u_components)}; 

  // insert_restriction is idempotent, so only the first call will do anything expensive
  domain.insert_restriction(u_fes, space);

  const serac::BlockElementRestriction& G = domain.get_restriction(space);

  static mfem::Vector u_E_buffer; // persistent storage buffer to be used by u_E below
  u_E_buffer.SetSize(int(G.ESize()));
  mfem::BlockVector u_E(u_E_buffer, G.bOffsets());

  G.Gather(u_L, u_E);

  foreach_geometry([&](auto geom){
    const std::vector<int> & elements = domain.get(geom);
    if (gdim == dimension(geom) && elements.size() > 0) {
      nd::view<const double, 2> xi = qrule[geom].points;
      impl::batched_interpolate<geom, family, op>(output(qranges[geom]), u_E.GetBlock(geom).ReadWrite(), u_degree, u_components, elements, xi); 
    }
  });

}

} // namespace impl

nd::array<double,3> evaluate(const FieldOp && input, Domain & domain, const MeshQuadratureRule & qrule) { 

  Family f = get_family(input.field);

  uint32_t gdim = geometry_dimension(domain);
  uint32_t num_components = get_num_components(input.field);
  stack::array<uint32_t, 3> output_dimensions{
    total(qrule.num_qpts(domain)),
    num_components,
    qshape(f, input.op, gdim)
  };

  nd::array<double, 3> u_q(output_dimensions);

  foreach_constexpr< Family::H1, Family::HCURL >([&](auto family) {
    if (family == f) {
      if (input.op == DerivedQuantity::VALUE) {
        impl::evaluate<family, DerivedQuantity::VALUE>(u_q, input.field, domain, qrule);
      }
      if (input.op == DerivedQuantity::DERIVATIVE) {
        impl::evaluate<family, DerivedQuantity::DERIVATIVE>(u_q, input.field, domain, qrule);
      }
    }
  });

  return u_q;
  
}

FieldOp grad(const Field & f) { 
  SLIC_ERROR_IF(!is_scalar_valued(get_family(f)), "grad(Field) only supports scalar-valued function spaces");
  return {DerivedQuantity::DERIVATIVE, f}; 
}

FieldOp curl(const Field & f) {
  SLIC_ERROR_IF(get_family(f) != Family::HCURL, "curl(Field) only supports Family::Hcurl");
  return {DerivedQuantity::DERIVATIVE, f};
}

FieldOp div(const Field & f) {
  SLIC_ERROR_IF(get_family(f) != Family::HDIV, "div(Field) only supports Family::Hdiv");
  return {DerivedQuantity::DERIVATIVE, f};
}

BasisFunctionOp grad(const BasisFunction & f) { 
  SLIC_ERROR_IF(!is_scalar_valued(f.space.family), "grad(BasisFunction) only supports scalar-valued function spaces");
  return {DerivedQuantity::DERIVATIVE, f}; 
}

BasisFunctionOp curl(const BasisFunction & f) {
  SLIC_ERROR_IF(f.space.family != Family::HCURL, "curl(BasisFunction) only supports Family::Hcurl");
  return {DerivedQuantity::DERIVATIVE, f};
}

BasisFunctionOp div(const BasisFunction & f) {
  SLIC_ERROR_IF(f.space.family != Family::HDIV, "div(BasisFunction) only supports Family::Hdiv");
  return {DerivedQuantity::DERIVATIVE, f};
}

} // namespace refactor
#endif
