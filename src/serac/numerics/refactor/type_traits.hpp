#pragma once

#include <functional>

#include "serac/numerics/refactor/containers/ndarray.hpp"

namespace refactor {

template < typename T >
struct always_true {
  static constexpr bool value = true;
};

////////////////////////////////////////////////////////////////////////////////

template < typename T >
struct is_a_1D_ndarray {
  static constexpr bool value = false;
};

template < typename T >
struct is_a_1D_ndarray < nd::array< T, 1 > >{
  static constexpr bool value = true;
};

template < typename T >
struct is_a_1D_ndview {
  static constexpr bool value = false;
};

template < typename T >
struct is_a_1D_ndview < nd::view< T, 1 > >{
  static constexpr bool value = true;
};

template < typename T >
struct is_ndarray {
  static constexpr bool value = false;
};

template < typename T, uint32_t n >
struct is_ndarray< nd::array<T, n> > {
  static constexpr bool value = true;
};

////////////////////////////////////////////////////////////////////////////////

template < typename T >
struct is_a_function_pointer {
  static constexpr bool value = false;
};

template < typename return_type, typename ... argument_types >
struct is_a_function_pointer < return_type (*)(argument_types ...) >{
  static constexpr bool value = true;
};

template < typename T >
struct is_a_stdfunction {
  static constexpr bool value = false;
};

template < typename return_type, typename ... argument_types >
struct is_a_stdfunction < std::function< return_type(argument_types ...) > >{
  static constexpr bool value = true;
};

////////////////////////////////////////////////////////////////////////////////

}