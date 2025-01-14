#pragma once

#include <functional>

#include "serac/numerics/functional/family.hpp"
#include "serac/numerics/refactor/type_traits.hpp"
#include "serac/numerics/refactor/finite_element.hpp"
#include "serac/physics/state/finite_element_state.hpp"
#include "serac/physics/state/finite_element_dual.hpp"

//#include "misc/for_constexpr.hpp"

namespace refactor {

struct BasisFunctionOp {
  const Modifier mod;
  const DerivedQuantity op;
  const BasisFunction function;

  BasisFunctionOp(const BasisFunction & f) : mod(Modifier::NONE), op(DerivedQuantity::VALUE), function(f){};
  BasisFunctionOp(const DerivedQuantity & o, const BasisFunction & f) : mod(Modifier::NONE), op(o), function(f){};
  BasisFunctionOp(const Modifier & m, const DerivedQuantity & o, const BasisFunction & f) : mod(m), op(o), function(f){};
};

BasisFunctionOp grad(const BasisFunction & phi);
BasisFunctionOp curl(const BasisFunction & phi);
BasisFunctionOp div(const BasisFunction & phi);

// for integrating sparse matrices (e.g. mass, stiffness)
template < typename T1, typename T2, typename T3 >
struct WeightedIntegrand {
  const T1 test;
  const T2 & qdata; 
  const T3 trial;
};

// for integrating residual vectors
template < typename T1, typename T2 >
struct WeightedIntegrand< T1, T2, void > {
  const T1 test; 
  const T2 & qdata;
};

inline BasisFunctionOp grad(const BasisFunction & f) { 
  SLIC_ERROR_IF(!is_scalar_valued(f.space.family), "grad(BasisFunction) only supports scalar-valued function spaces");
  return {DerivedQuantity::DERIVATIVE, f}; 
}

inline BasisFunctionOp curl(const BasisFunction & f) {
  SLIC_ERROR_IF(f.space.family != Family::HCURL, "curl(BasisFunction) only supports Family::Hcurl");
  return {DerivedQuantity::DERIVATIVE, f};
}

inline BasisFunctionOp div(const BasisFunction & f) {
  SLIC_ERROR_IF(f.space.family != Family::HDIV, "div(BasisFunction) only supports Family::Hdiv");
  return {DerivedQuantity::DERIVATIVE, f};
}

template < typename T >
struct DiagonalOnly : public T {};

template < typename T >
DiagonalOnly(T) -> DiagonalOnly<T>;

template < typename T, uint32_t n >
auto operator*(const nd::array<T,n> & data, const BasisFunction & phi) {
  return WeightedIntegrand< BasisFunctionOp, nd::array<T,n>, void >{phi, data};
}

template < typename T, uint32_t n >
auto operator*(const nd::array<T,n> & data, const BasisFunctionOp & dphi) {
  return WeightedIntegrand< BasisFunctionOp, nd::array<T,n>, void >{dphi, data};
}

template < typename T, uint32_t n >
auto dot(const nd::array<T,n> & data, const BasisFunctionOp & dphi) {
  return WeightedIntegrand< BasisFunctionOp, nd::array<T,n>, void >{dphi, data};
}

template < typename T, uint32_t n >
auto diagonal(WeightedIntegrand< BasisFunctionOp, nd::array<T,n>, BasisFunctionOp > integrand) {
  SERAC_ASSERT(
    integrand.test.function == integrand.trial.function,
    "diag(...) requires same test and trial space"
  );

  return DiagonalOnly{integrand};
}

////////////////////////////////////////////////////////////////////////////////

template < typename T, uint32_t n >
auto dot(BasisFunctionOp psi, const nd::array<T,n> & data, BasisFunctionOp phi) {
  return WeightedIntegrand< BasisFunctionOp, nd::array<T,n>, BasisFunctionOp >{phi, data, psi};
}

namespace impl {

template < typename T >
struct is_a_weighted_integrand {
  static constexpr bool value = false;
};

template < typename T1, typename T2, typename T3 >
struct is_a_weighted_integrand < WeightedIntegrand< T1, T2, T3 > >{
  static constexpr bool value = true;
};

template < typename T1, typename T2, typename T3 >
struct is_a_weighted_integrand < DiagonalOnly< WeightedIntegrand< T1, T2, T3 > > >{
  static constexpr bool value = true;
};

template < typename return_type, typename T >
uint32_t dimension_of_first_argument(return_type (* /*integrand*/)(T)) {
  return dimension(T{});
}

template < typename return_type, typename T >
uint32_t dimension_of_first_argument(std::function< return_type(T) > /*integrand*/) {
  return dimension(T{});
}

////////////////////////////////////////////////////////////////////////////////

#if 0
template < typename T >
T integrate_ndarray(const nd::array< T > & integrand, const Domain& domain);

////////////////////////////////////////////////////////////////////////////////

template < typename return_type >
return_type integrate_function_pointer(return_type (*f)(vec2), const Domain& domain, const DomainType & type);

template < typename return_type >
return_type integrate_function_pointer(return_type (*f)(vec3), const Domain& domain, const DomainType & type);

////////////////////////////////////////////////////////////////////////////////

template < typename return_type >
return_type integrate_stdfunction(std::function< return_type(vec2) > integrand, const Domain& domain, const DomainType & type);

template < typename return_type >
return_type integrate_stdfunction(std::function< return_type(vec3) > integrand, const Domain& domain, const DomainType & type);
#endif

////////////////////////////////////////////////////////////////////////////////

template < serac::Family family >
void integrate_residual(Residual & r, BasisFunctionOp b, const nd::view<const double, 3> f, Domain & domain);

template < uint32_t n >
Residual integrate_weighted_integrand(
  const WeightedIntegrand< BasisFunctionOp, nd::array<double, n>, void > & integrand, 
  Domain & domain) {

  DerivedQuantity op = integrand.test.op;
  BasisFunction phi = integrand.test.function;
  Family test_family = phi.space.family;
  uint32_t p = phi.space.degree;
  uint32_t components = phi.space.components;
  uint32_t gdim = geometry_dimension(domain);

  Residual r(domain.mesh_, test_family, p, components);

  const nd::array<double, n> & qdata = integrand.qdata;

  stack::array<uint32_t, 3> shape3D{qdata.shape[0], components, qshape(test_family, op, gdim)};

  refactor_ASSERT(compatible_shapes(qdata.shape, shape3D), "incompatible array shapes");

  nd::view<const double, 3> q3D{qdata.data(), shape3D};
  foreach_constexpr< serac::Family::H1, serac::Family::Hcurl >([&](auto family) {
    if (family == test_family) {
      integrate_residual< family >(r, integrand.test, q3D, domain, type);
    }
  });

  return r;
}

////////////////////////////////////////////////////////////////////////////////

#if 0
template <serac::Family test_family, serac::Family trial_family>
std::function< void(refactor::sparse_matrix&) > integrate_sparse_matrix(BasisFunctionOp phi, const nd::view<const double, 5> f, BasisFunctionOp psi, const Domain &domain);

template < uint32_t n >
std::function< void(refactor::sparse_matrix&) > integrate_weighted_integrand(
  const WeightedIntegrand< BasisFunctionOp, nd::array<double, n>, BasisFunctionOp > & integrand, 
  const Domain & domain) {

  BasisFunctionOp phi = integrand.test;
  BasisFunctionOp psi = integrand.trial;
  const nd::array<double, n> & qdata = integrand.qdata;

  DerivedQuantity test_op = phi.op;
  DerivedQuantity trial_op = psi.op;
  FunctionSpace test_space = phi.function.space;
  FunctionSpace trial_space = psi.function.space;
  uint32_t test_components = phi.function.space.components;
  uint32_t trial_components = psi.function.space.components;
  uint32_t sdim = domain.mesh.spatial_dimension;
  uint32_t gdim = domain.mesh.geometry_dimension;

  stack::array<uint32_t, 5> shape5D = {
    qdata.shape[0],
    phi.function.space.components, qshape(test_space.family, test_op, gdim),
    psi.function.space.components, qshape(trial_space.family, trial_op, gdim)
  };

  SLIC_ASSER_MSH(compatible_shapes(qdata.shape, shape5D), "incompatible array shapes");

  std::function< void(refactor::sparse_matrix&) > output;

  nd::view<const double,5> q5D{qdata.data(), shape5D};
  foreach_constexpr< serac::Family::H1, serac::Family::Hcurl >([&](auto test_family) {
    foreach_constexpr< serac::Family::H1, serac::Family::Hcurl >([&](auto trial_family) {
      if (test_family == test_space.family && trial_family == trial_space.family) {
        output = integrate_sparse_matrix< test_family, trial_family >(phi, q5D, psi, domain, type);
      }
    });
  });

  return output;

}

////////////////////////////////////////////////////////////////////////////////

template <Family family>
void integrate_sparse_matrix_diagonal(Residual & r, BasisFunctionOp phi, const nd::view<const double, 5> f, BasisFunctionOp psi, const Domain &domain);

template < uint32_t n >
nd::array<double,2> integrate_weighted_integrand(
  const DiagonalOnly < WeightedIntegrand< BasisFunctionOp, nd::array<double, n>, BasisFunctionOp > > & integrand, 
  const Domain & domain) {

  BasisFunctionOp phi = integrand.test;
  BasisFunctionOp psi = integrand.trial;
  const nd::array<double, n> & qdata = integrand.qdata;

  SLIC_ASSERT_MSG(phi.function.space == psi.function.space, "must have matching test and trial spaces for diag(...)");

  DerivedQuantity test_op = phi.op;
  DerivedQuantity trial_op = psi.op;
  Family test_family = phi.function.space.family;
  Family trial_family = psi.function.space.family;
  uint32_t test_components = phi.function.space.components;
  uint32_t trial_components = psi.function.space.components;
  uint32_t sdim = domain.mesh.spatial_dimension;
  uint32_t gdim = domain.mesh.geometry_dimension;

  stack::array<uint32_t, 5> shape5D = {
    qdata.shape[0],
    phi.function.space.components, qshape(test_family, test_op, gdim),
    psi.function.space.components, qshape(trial_family, trial_op, gdim)
  };

  SLIC_ASSERT_MSG(compatible_shapes(qdata.shape, shape5D), "incompatible array shapes");

  Residual output(phi.function.space, domain.mesh);

  nd::view<const double,5> q5D{qdata.data(), shape5D};
  foreach_constexpr< Family::H1, Family::Hcurl >([&](auto family) {
    if (family == test_family) {
      integrate_sparse_matrix_diagonal< family >(output, phi, q5D, psi, domain, type);
    }
  });

  return output.data;

}
#endif

} // namespace impl

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template < typename integrand_t >
auto integrate(const integrand_t & integrand, Domain & domain) {

  //if constexpr (is_a_1D_ndarray<integrand_t>::value) {
  //  return impl::integrate_ndarray(integrand, d.domain, d.type);
  //} 

  //if constexpr (is_a_function_pointer<integrand_t>::value) {
  //  uint32_t sdim = impl::dimension_of_first_argument(integrand);
  //  SLIC_ASSERT_MSG(d.domain.mesh.X.data.shape[1] != sdim, "integration function has invalid signature");
  //  return impl::integrate_function_pointer(integrand, d.domain, d.type);
  //} 

  //if constexpr (is_a_stdfunction<integrand_t>::value) {
  //  uint32_t sdim = impl::dimension_of_first_argument(integrand);
  //  SLIC_ASSERT_MSG(d.domain.mesh.X.data.shape[1] != sdim, "integration function has invalid signature");
  //  return impl::integrate_stdfunction(integrand, d.domain, d.type);
  //} 

  if constexpr (impl::is_a_weighted_integrand<integrand_t>::value) {
    return impl::integrate_weighted_integrand(integrand, domain);
  }

  static_assert(always_true<integrand_t>::value, "error: unsupported integrand type");

}

}