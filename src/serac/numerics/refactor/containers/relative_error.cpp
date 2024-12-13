#include "nd/array.hpp"

#include <cmath>

template < uint32_t rank >
double relative_error_impl(nd::view<const double, rank> x, nd::view<const double, rank> y) {
  if (x.shape != y.shape) {
    std::cout << "shape mismatch" << std::endl;
  }

  double norm = 0.0;
  double error = 0.0;
  uint32_t sz = x.size();
  for (uint32_t i = 0; i < sz; i++) {
    double avg = 0.5 * (x.values[i] + y.values[i]);
    double diff = x.values[i] - y.values[i];
    error += diff * diff;
    norm += avg * avg;
  }
  return sqrt(error / norm);
}

double relative_error(nd::view<const double, 1> a, nd::view<const double,1> b) { return relative_error_impl(a, b); };
double relative_error(nd::view<const double, 2> a, nd::view<const double,2> b) { return relative_error_impl(a, b); };
double relative_error(nd::view<const double, 3> a, nd::view<const double,3> b) { return relative_error_impl(a, b); };
double relative_error(nd::view<const double, 4> a, nd::view<const double,4> b) { return relative_error_impl(a, b); };