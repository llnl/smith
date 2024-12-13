#pragma once

#include <cinttypes>

namespace stack {

  template < typename T, uint32_t dim >
  struct array { 
    SERAC_HOST_DEVICE constexpr T & operator[](uint32_t i) { return values[i]; }
    SERAC_HOST_DEVICE constexpr const T & operator[](uint32_t i) const { return values[i]; }

    SERAC_HOST_DEVICE bool operator==(const array<T,dim> & other) const {
      for (uint32_t i = 0; i < dim; i++) {
        if (values[i] != other[i]) return false;
      } 
      return true;
    }

    SERAC_HOST_DEVICE bool operator!=(const array<T,dim> & other) const { return !(this->operator==(other)); }

    T values[dim]; 
  };

  template < typename T, uint32_t dim >
  constexpr uint32_t size(array<T,dim>) { return dim; }

  template < typename T >
  array(T, T) -> array<T,2>;

  template < typename T >
  array(T, T, T) -> array<T,3>;

  template < typename T >
  array(T, T, T, T) -> array<T,4>;

  template < typename T >
  array(T, T, T, T, T) -> array<T,5>;

  template < typename T >
  array(T, T, T, T, T, T) -> array<T,6>;

}
