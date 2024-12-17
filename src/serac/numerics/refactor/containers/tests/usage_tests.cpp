#include <gtest/gtest.h>

#include "common.hpp"

void accepts_const_ndarray(const nd::array< double, 2 > & arr) {}

void accepts_ndarray(nd::array< double, 2 > & arr) {
  accepts_const_ndarray(arr);
}

void accepts_ndview(nd::view< double, 2 > arr) {}

void accepts_const_ndview(nd::view< const double, 2 > arr) {}

void accepts_1dview(nd::view< double, 1 > arr) {}

void operator*(nd::view<double, 2> arr, double scale) {}

nd::array<double, 2> returns_ndarray() {
  return nd::array<double,2>{{1, 2}};
}

TEST(UnitTest, foo) {

  nd::array<double, 2> arr = returns_ndarray();
  nd::view<double, 2> arr_view = arr;

  accepts_ndarray(arr);

  // nd::array can implicitly convert to nd::view, but not vice versa
  // accepts_ndarray(arr_view);

  accepts_ndview(arr);
  accepts_ndview(arr_view);
  
  // implicit conversion of `nd::array &&` to `nd::view`
  // is disallowed, since it would create a dangling reference
  #if 0
  accepts_ndview(returns_ndarray());
  #endif

  // nd::array<T,n> and nd::view<T,n> can both 
  // implicitly convert to nd::view<const T, n>
  accepts_const_ndview(arr);
  accepts_const_ndview(arr_view);

  // nd::view conversions and transformations can be concatenated
  accepts_1dview(flatten(arr));
  accepts_1dview(flatten(arr_view));

  // implicit conversion on operator overloads
  arr * 3.0;
  
}
