#pragma once

#include "serac/infrastructure/accelerator.hpp"
#include "serac/numerics/functional/tensor.hpp"
#include "serac/numerics/refactor/finite_element.hpp"

namespace refactor {

uint32_t elements_per_block(mfem::Geometry::Type geom, Family family, int p);

GeometryInfo geometry_counts(const Domain & domain);

template < typename T >
struct array_rank;

template < typename T, uint32_t n >
struct array_rank< nd::view< T, n > >{
  static constexpr uint32_t value = n;
};

template < typename T, uint32_t n >
struct array_rank< nd::array< T, n > >{
  static constexpr uint32_t value = n;
};

SERAC_HOST_DEVICE constexpr uint32_t round_up_to_multiple_of_128(uint32_t n) {
  return ((n + 127) / 128) * 128;
};

constexpr uint32_t source_shape(Family f, uint32_t gdim) {
  switch(f) {
    case Family::QOI:   return 1;
    case Family::H1:    return 1;
    case Family::HCURL: return gdim;
    case Family::HDIV:  return gdim;
    case Family::L2:    return 1;
  }
  return (1u << 31);
}

constexpr uint32_t flux_shape(Family f, uint32_t gdim) {
  switch(f) {
    case Family::QOI:   return 0;
    case Family::H1:    return gdim;
    case Family::HCURL: return (gdim == 2) ? 1 : gdim;
    case Family::HDIV:  return (gdim == 2) ? 1 : gdim;
    case Family::L2:    return gdim;
  }

  return (1u << 31);
}

template < Family f, DerivedQuantity op, int dim >
auto piola_transformation(const mat<dim,dim> & dX_dxi) {
  if constexpr ((f == Family::H1    && op == DerivedQuantity::DERIVATIVE) ||
                (f == Family::L2    && op == DerivedQuantity::DERIVATIVE) ||
                (f == Family::HCURL && op == DerivedQuantity::VALUE)) {
    return inv(dX_dxi);
  }

  if constexpr ((f == Family::HCURL && op == DerivedQuantity::DERIVATIVE) ||
                (f == Family::HDIV  && op == DerivedQuantity::VALUE)) {
    if constexpr (dim == 2) {
      return mat<1,1>{1.0 / det(dX_dxi)};
    } else {
      return transpose(dX_dxi) / det(dX_dxi);
    }
  }

  if constexpr ((f == Family::H1    && op == DerivedQuantity::VALUE) ||
                (f == Family::L2    && op == DerivedQuantity::VALUE) || 
                (f == Family::HDIV  && op == DerivedQuantity::DERIVATIVE)) {
    // this should never be called, but we implement it here regardless
    // to suppress a compiler warning about incompatible return values
    return 1.0;
  }
}

template < Family f, DerivedQuantity op, int dim >
auto weighted_piola_transformation(const mat<dim,dim> & dX_dxi) {
  if constexpr ((f == Family::H1    && op == DerivedQuantity::DERIVATIVE) ||
                (f == Family::L2    && op == DerivedQuantity::DERIVATIVE) ||
                (f == Family::HCURL && op == DerivedQuantity::VALUE)) {
    return adj(dX_dxi);
  }

  if constexpr ((f == Family::HCURL && op == DerivedQuantity::DERIVATIVE) ||
                (f == Family::HDIV  && op == DerivedQuantity::VALUE)) {
    if constexpr (dim == 2) {
      return mat<1,1>{1.0};
    } else {
      return transpose(dX_dxi);
    }
  }

  if constexpr ((f == Family::H1    && op == DerivedQuantity::VALUE) ||
                (f == Family::L2    && op == DerivedQuantity::VALUE) || 
                (f == Family::HDIV  && op == DerivedQuantity::DERIVATIVE)) {
    return det(dX_dxi);
  }
}

// +------------+------+-------+------+----+
// |            |  H1  | Hcurl | Hdiv | DG |
// +------------+------+-------+------+----+
// | value (1D) |   1  |   1   |   1  |  1 |
// +------------+------+-------+------+----+
// | deriv (1D) |   1  |   1   |   1  |  1 |
// +------------+------+-------+------+----+
// | value (2D) |   1  |   2   |   2  |  1 |
// +------------+------+-------+------+----+
// | deriv (2D) |   2  |   1   |   1  |  2 |
// +------------+------+-------+------+----+
// | value (3D) |   1  |   3   |   3  |  1 |
// +------------+------+-------+------+----+
// | deriv (3D) |   3  |   3   |   1  |  3 |
// +------------+------+-------+------+----+
constexpr uint32_t qshape(Family f, DerivedQuantity op, uint32_t gdim) {

  if (op == DerivedQuantity::VALUE) {
    return is_vector_valued(f) ? gdim : 1;
  } 

  if (op == DerivedQuantity::DERIVATIVE) {
    if (f == Family::H1 || f == Family::L2) { return gdim; }
    if (f == Family::HCURL) { return (gdim == 2) ? 1 : gdim; }
    if (f == Family::HDIV) { return 1; }
  }

  return (1u<<31);

}

template < typename T, uint32_t n >
stack::array<T, n> remove_ones(const stack::array<T, n> & x) { 
  uint32_t rank = 0;
  stack::array<T, n> copy{};
  for (uint32_t i = 0; i < n; i++) {
    if (x[i] > 1) copy[rank++] = x[i];
  }
  return copy;
}

template < typename T, uint32_t m, uint32_t n >
bool compatible_shapes(const stack::array<T, m> & x, 
                       const stack::array<T, n> & y) {
  auto x_filtered = remove_ones(x);
  auto y_filtered = remove_ones(y);
  for (uint32_t i = 0; i < std::max(m, n); i++) {
    auto xval = (i >= m) ? 0 : x_filtered[i];
    auto yval = (i >= n) ? 0 : y_filtered[i];
    if (xval != yval) { return false; }
  }
  return true;
}

template < mfem::Geometry::Type geom >
auto quadrature_point(uint32_t q, const nd::view<const double, 2> xi) {

  if constexpr (mfem::Geometry::SQUARE == geom) {
    uint32_t q1D = xi.shape[0];
    uint32_t qx = q % q1D;
    uint32_t qy = q / q1D;
    return vec2{xi(qx, 0), xi(qy, 0)};
  }

  if constexpr (mfem::Geometry::CUBE == geom) {
    uint32_t q1D = xi.shape[0];
    uint32_t qx = q % q1D;
    uint32_t qy = (q % (q1D * q1D)) / q1D;
    uint32_t qz = q / (q1D * q1D);
    return vec3{xi(qx, 0), xi(qy, 0), xi(qz, 0)};
  }

  // all other geometries
  constexpr int gdim = dimension(geom);
  vec< gdim, double > xi_q;
  for (uint32_t c = 0; c < gdim; c++) {
    xi_q(c) = xi(q, c);
  }
  return xi_q;

}

namespace impl {

template < Family family, uint32_t n >
auto value_transformation(const mat<n,n,double> & A) {
  if constexpr (family == Family::H1) {
    return mat1{1.0};
  }
  if constexpr (family == Family::HCURL) {
    return inv(A); 
  }
}

template < Family family, uint32_t n >
auto derivative_transformation(const mat<n,n,double> & A) {
  if constexpr (family == Family::H1) {
    return contravariant_piola(A);
  }
  if constexpr (family == Family::HCURL) {
    return covariant_piola(A);
  }
}

template < Family family, uint32_t n >
auto source_transformation(const mat<n,n,double> & A) {
  if constexpr (family == Family::H1) {
    return det(A);
  }
  if constexpr (family == Family::HCURL) {
    return inv(A) * det(A); 
  }
}

template < Family family, uint32_t n >
auto flux_transformation(const mat<n,n,double> & A) {
  if constexpr (family == Family::H1) {
    return inv(A) * det(A);
  }
  if constexpr (family == Family::HCURL) {
    if constexpr (n <= 2) {
      return vec1(1.0);
    }
    if constexpr (n == 3) {
      return transpose(A);
    }
  }
}

}

}

namespace nd {

template < typename T, uint32_t n >
struct printer< serac::vec<n,T> >{ 
  static SERAC_HOST_DEVICE void print(const serac::vec<n,T> & v) {
    printf("{");
    printer<T>::print(v(0));
    for (int i = 1; i < n; i++) {
      printf(",");
      printer<T>::print(v(i));
    }
    printf("}");
  }
};

}
