#pragma once

#include <cstdio>  // for printf
#include <cstring> // for std::memcpy
#include <utility> // for std::integer_sequence
#include <iostream>
#include <cinttypes>
#include <type_traits>

#include "serac/infrastructure/accelerator.hpp"
#include "serac/numerics/refactor/containers/stack_array.hpp"

namespace memory {
  enum space {CPU, GPU, UNIFIED};

  void * allocate(uint64_t n);
  void deallocate(void * ptr);
  void memcpy(void * dest, void * src, uint64_t n);
  void zero(void * ptr, uint64_t n);

  template < typename T >
  T * allocate(uint64_t n) { 
    return static_cast<T *>(allocate(n * sizeof(T)));
  }

  template < typename T >
  void deallocate(T * ptr) {
    deallocate(static_cast<void*>(ptr));
  }

  template < typename T >
  void memcpy(T * dest, T * src, uint64_t n) { 
    memcpy(static_cast<void*>(dest), static_cast<void*>(src), n * sizeof(T));
  }

  template < typename T >
  void zero(T * ptr, uint64_t n) {
    zero(static_cast<void*>(ptr), n * sizeof(T));
  } 
}

namespace nd {

  template < uint32_t dim >
  SERAC_HOST_DEVICE uint32_t product(stack::array< uint32_t, dim > values) {
    uint32_t p = values[0];
    for (int i = 1; i < dim; i++) { p *= values[i]; }
    return p;
  }

  enum ordering{ row_major, col_major };

  template < uint32_t dim >
  SERAC_HOST_DEVICE stack::array< uint32_t, dim > compute_strides(const stack::array< uint32_t, dim > & shape, ordering o, uint32_t m = 1) {
    stack::array< uint32_t, dim > strides{};
    int32_t k = (o == col_major) ? 0 : dim-1;
    int32_t s = (o == col_major) ? 1 : -1;
    for (uint32_t i = 0; i < dim; i++) {
      strides[k] = (i == 0) ? m : strides[k-s] * shape[k-s];
      k += s;
    }
    return strides;
  }

  // used for slicing arrays and views
  template < typename T >
  struct range { T begin; T end; };

  template < typename T >
  range(T,T) -> range<T>;

  template < typename T >
  SERAC_HOST_DEVICE constexpr T begin(const range<T> & r) { return r.begin; }

  template < typename T >
  SERAC_HOST_DEVICE constexpr T end(const range<T> & r) { return r.end; }

  template < typename T >
  SERAC_HOST_DEVICE constexpr uint32_t is_range(T) { return 0; };

  template < typename T >
  SERAC_HOST_DEVICE constexpr uint32_t is_range(range<T>) { return 1; };

  SERAC_HOST_DEVICE constexpr int32_t begin(int32_t x) { return x; }
  SERAC_HOST_DEVICE constexpr int64_t begin(int64_t x) { return x; }
  SERAC_HOST_DEVICE constexpr uint32_t begin(uint32_t x) { return x; }
  SERAC_HOST_DEVICE constexpr uint64_t begin(uint64_t x) { return x; }

  SERAC_HOST_DEVICE constexpr int32_t end(int32_t x) { return x; }
  SERAC_HOST_DEVICE constexpr int64_t end(int64_t x) { return x; }
  SERAC_HOST_DEVICE constexpr uint32_t end(uint32_t x) { return x; }
  SERAC_HOST_DEVICE constexpr uint64_t end(uint64_t x) { return x; }

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

  template < typename T, uint32_t dim = 1 >
  struct view {
    static_assert(dim > 0);

    static constexpr auto iseq = std::make_integer_sequence< uint32_t, dim >();

    SERAC_HOST_DEVICE constexpr view() {
      values = nullptr;
      sz = 0;
      shape = {};
      stride = {};
    }

    SERAC_HOST_DEVICE constexpr view(T * input, const stack::array< uint32_t, dim > & dimensions) {
      values = input;
      for (uint32_t i = 0; i < dim; i++) {
        uint32_t id = dim - 1 - i;
        shape[id] = dimensions[id];
        stride[id] = (id == dim - 1) ? 1 : stride[id+1] * shape[id+1];
      }
      sz = product(shape);
    }

    SERAC_HOST_DEVICE constexpr view(T * input, const stack::array< uint32_t, dim > & dimensions, const stack::array< uint32_t, dim > & strides) {
      values = input;
      shape = dimensions;
      stride = strides;
      sz = product(shape);
    }

    template < typename ... index_types >
    SERAC_HOST_DEVICE uint32_t index(index_types ... indices) const { 
      static_assert(sizeof ... (indices) == dim);
      return values[index(iseq, indices...)];
    }

    template < uint32_t ... I, typename ... index_types >
    SERAC_HOST_DEVICE uint32_t index(std::integer_sequence<uint32_t, I...>, index_types ... indices) const {
      #ifdef NDARRAY_ENABLE_BOUNDS_CHECKING
        // note: the cast to int32_t is a way to avoid warnings 
        // about pointless comparison between unsigned integer types and 0
        if (((int32_t(indices) < 0 || indices >= shape[I]) || ... )) {
          printf("array index out of bounds\n");
        };
      #endif
      return ((indices * stride[I]) + ...);
    }

    template < typename ... index_types >
    SERAC_HOST_DEVICE decltype(auto) operator()(index_types ... indices) const { 
      constexpr uint32_t num_args = sizeof ... (indices);
      static_assert(num_args <= dim);
      constexpr uint32_t rank = (is_range(index_types{}) + ... ) + (dim - num_args);
      if constexpr (rank == 0) {
        return static_cast<T&>(values[index(iseq, indices...)]);
      } else {
        constexpr uint32_t is_a_range[] = {is_range(index_types{}) ... };
        stack::array<uint32_t, rank> slice_shape{};
        stack::array<uint32_t, rank> slice_stride{};

        uint32_t beginnings[num_args] = {uint32_t(nd::begin(indices)) ... };
        uint32_t endings[num_args] = {uint32_t(nd::end(indices)) ... };
        int k = 0;
        int offset = 0;
        for (int i = 0; i < dim; i++) {
          if (i >= num_args) {
            slice_shape[k] = shape[i];
            slice_stride[k] = stride[i];
            k++;
          } else {
            offset += stride[i] * beginnings[i];
            if (is_a_range[i]) {
              slice_shape[k] = endings[i] - beginnings[i];
              slice_stride[k] = stride[i];
              k++;
            }
          }
        }
        return view<T, rank>{&values[offset], slice_shape, slice_stride};
      }
    }

    template < typename ... index_types >
    SERAC_HOST_DEVICE decltype(auto) operator()(index_types ... indices) { 
      constexpr uint32_t num_args = sizeof ... (indices);
      static_assert(num_args <= dim);
      constexpr uint32_t rank = (is_range(index_types{}) + ... ) + (dim - num_args);
      if constexpr (rank == 0) {
        return static_cast<T&>(values[index(iseq, indices...)]);
      } else {
        constexpr uint32_t is_a_range[] = {is_range(index_types{}) ... };
        stack::array<uint32_t, rank> slice_shape{};
        stack::array<uint32_t, rank> slice_stride{};

        uint32_t beginnings[num_args] = {uint32_t(nd::begin(indices)) ... };
        uint32_t endings[num_args] = {uint32_t(nd::end(indices)) ... };
        int k = 0;
        int offset = 0;
        for (int i = 0; i < dim; i++) {
          if (i >= num_args) {
            slice_shape[k] = shape[i];
            slice_stride[k] = stride[i];
            k++;
          } else {
            offset += stride[i] * beginnings[i];
            if (is_a_range[i]) {
              slice_shape[k] = endings[i] - beginnings[i];
              slice_stride[k] = stride[i];
              k++;
            }
          }
        }
        return view<T, rank>{&values[offset], slice_shape, slice_stride};
      }
    }

    template < typename index_type >
    SERAC_HOST_DEVICE auto & operator[](index_type i) { return values[i]; }

    template < typename index_type >
    SERAC_HOST_DEVICE auto & operator[](index_type i) const { return values[i]; }

    SERAC_HOST_DEVICE operator view<const T, dim>() const {
      return view<const T, dim>{values, shape, stride};
    }

    SERAC_HOST_DEVICE T * data() { return &values[0]; }
    SERAC_HOST_DEVICE const T * data() const { return &values[0]; }

    SERAC_HOST_DEVICE uint32_t size() const { return product(shape); }

    SERAC_HOST_DEVICE T * begin() const { return &values[0]; }
    SERAC_HOST_DEVICE T * end() const { return &values[size()]; }

    T * values;
    uint64_t sz;
    stack::array< uint32_t, dim > shape;
    stack::array< uint32_t, dim > stride;

  };

  template < typename T, uint32_t dim >
  struct view< const T, dim >{
    static_assert(dim > 0);

    static constexpr auto iseq = std::make_integer_sequence< uint32_t, dim >();

    SERAC_HOST_DEVICE constexpr view() {
      values = nullptr;
      sz = 0;
      shape = {};
      stride = {};
    }

    SERAC_HOST_DEVICE constexpr view(const T * input, const stack::array< uint32_t, dim > & dimensions) {
      values = input;
      for (uint32_t i = 0; i < dim; i++) {
        uint32_t id = dim - 1 - i;
        shape[id] = dimensions[id];
        stride[id] = (id == dim - 1) ? 1 : stride[id+1] * shape[id+1];
      }
      sz = product(shape);
    }

    SERAC_HOST_DEVICE constexpr view(const T * input, const stack::array< uint32_t, dim > & dimensions, const stack::array< uint32_t, dim > & strides) {
      values = input;
      shape = dimensions;
      stride = strides;
      sz = product(shape);
    }

    template < typename ... index_types >
    SERAC_HOST_DEVICE uint32_t index(index_types ... indices) const { 
      static_assert(sizeof ... (indices) == dim);
      return values[index(iseq, indices...)];
    }

    template < uint32_t ... I, typename ... index_types >
    SERAC_HOST_DEVICE uint32_t index(std::integer_sequence<uint32_t, I...>, index_types ... indices) const {
      #ifdef NDARRAY_ENABLE_BOUNDS_CHECKING
        // note: the cast to int32_t is a way to avoid warnings 
        // about pointless comparison between unsigned integer types and 0
        if (((int32_t(indices) < 0 || indices >= shape[I]) || ... )) {
          printf("array index out of bounds\n");
        };
      #endif
      return ((indices * stride[I]) + ...);
    }

    template < typename ... index_types >
    SERAC_HOST_DEVICE decltype(auto) operator()(index_types ... indices) const { 
      constexpr uint32_t num_args = sizeof ... (indices);
      static_assert(num_args <= dim);
      constexpr uint32_t rank = (is_range(index_types{}) + ... ) + (dim - num_args);
      if constexpr (rank == 0) {
        return static_cast<const T&>(values[index(iseq, indices...)]);
      } else {
        constexpr uint32_t is_a_range[] = {is_range(index_types{}) ... };
        stack::array<uint32_t, rank> slice_shape{};
        stack::array<uint32_t, rank> slice_stride{};

        uint32_t beginnings[num_args] = {uint32_t(nd::begin(indices)) ... };
        uint32_t endings[num_args] = {uint32_t(nd::end(indices)) ... };
        int k = 0;
        int offset = 0;
        for (int i = 0; i < dim; i++) {
          if (i >= num_args) {
            slice_shape[k] = shape[i];
            slice_stride[k] = stride[i];
            k++;
          } else {
            offset += stride[i] * beginnings[i];
            if (is_a_range[i]) {
              slice_shape[k] = endings[i] - beginnings[i];
              slice_stride[k] = stride[i];
              k++;
            }
          }
        }
        return view<const T, rank>{&values[offset], slice_shape, slice_stride};
      }
    }

    template < typename index_type >
    SERAC_HOST_DEVICE const T & operator[](index_type i) const { return values[i]; }

    SERAC_HOST_DEVICE const T * data() const { return &values[0]; }

    SERAC_HOST_DEVICE uint32_t size() const { return product(shape); }

    SERAC_HOST_DEVICE const T * begin() const { return &values[0]; }
    SERAC_HOST_DEVICE const T * end() const { return &values[size()]; }

    const T * values;
    uint64_t sz;
    stack::array< uint32_t, dim > shape;
    stack::array< uint32_t, dim > stride;

  };

  template < typename T, uint32_t dim >
  using const_view = view<const T, dim>;

  template < typename T, uint32_t dim = 1 >
  struct array : public view< T, dim > {

    using view<T,dim>::iseq;
    using view<T,dim>::values;
    using view<T,dim>::shape;
    using view<T,dim>::stride;
    using view<T,dim>::sz;

    array() : view<T,dim>() {}
    
    array(stack::array< uint32_t, dim > dimensions) : view<T,dim>() { resize(dimensions); }

    array(stack::array< uint32_t, dim > dimensions,
          stack::array< uint32_t, dim > strides) : view<T,dim>() {
      resize(dimensions, strides);
    }

    array(const array & other)  {
      sz = other.sz;
      shape = other.shape;
      stride = other.stride;
      allocate(sz);
      memory::memcpy(values, other.values, other.sz); 
    }

    void operator=(const array & other) {
      shape = other.shape;
      stride = other.stride;
      _resize(other.sz);
      memory::memcpy(values, other.values, sz); 
    }

    array(array && other) {
      sz = other.sz;
      shape = other.shape;
      values = other.values;
      stride = other.stride;
      other.values = nullptr;
    }

    void operator=(array && other) {
      sz = other.sz;
      shape = other.shape;

      deallocate();
      values = other.values;

      stride = other.stride;
      other.values = nullptr;
    }

    ~array() { 
      deallocate(); 
    }

    void resize(uint32_t new_size) {
      static_assert(dim == 1, "resize(uint32_t) only defined for 1D arrays");
      stride[0] = 1;
      shape[0] = new_size;
      _resize(new_size);
    }

    void resize(const stack::array< uint32_t, dim > & new_shape) {
      shape = new_shape;
      _resize(product(shape));
      for (uint64_t i = 0; i < dim; i++) {
        uint64_t id = dim - 1 - i;
        stride[id] = (id == dim - 1) ? 1 : stride[id+1] * shape[id+1];
      }
    }

    void resize(const stack::array< uint32_t, dim > & new_shape, 
                const stack::array< uint32_t, dim > & new_stride) {
      shape = new_shape;
      stride = new_stride;
      _resize(product(shape));
    }

   private:

    void allocate(uint32_t n) { 
      values = memory::allocate<T>(n); 
    }

    void deallocate() { 
      if (values) { 
        memory::deallocate(values); 
        values = nullptr;
      } 
    }

    void _resize(uint32_t new_sz) {
      if (new_sz != sz) {
        deallocate();
        allocate(new_sz);
        sz = new_sz;
        memory::zero(values, sz);
      }
    }

  };

  template < typename T, uint32_t dim >
  view(T*, stack::array<T,dim>) -> view<T,dim>;

  template < typename T, uint32_t dim >
  view(T*, stack::array<T,dim>, stack::array<T,dim>) -> view<T,dim>;

  /////////////////////////////////////////////////////////////////////////////

  template < typename T, uint32_t dim > 
  void zero(array<T, dim> & arr) {
    memory::zero(arr.data(), arr.sz);
  }

//  template < typename T, uint32_t dim > 
//  void fill(array<T, dim> & arr, T value) {
//    for (uint32_t i = 0; i < arr.size(); i++) {
//      arr[i] = value;
//    }
//  }

  /////////////////////////////////////////////////////////////////////////////

  template < typename T, uint32_t dim > 
  SERAC_HOST_DEVICE view<const T,1> flatten(const array<T, dim> & arr) {
    return view<const T,1>{&arr.values[0], {product(arr.shape)}};
  }

  template < typename T, uint32_t dim > 
  SERAC_HOST_DEVICE view<T,1> flatten(array<T, dim> & arr) {
    return view<T,1>{&arr.values[0], {product(arr.shape)}};
  }

  template < typename T, uint32_t dim > 
  SERAC_HOST_DEVICE view<T,1> flatten(view<T, dim> arr) {
    return view<T,1>{arr.values, {product(arr.shape)}};
  }

  /////////////////////////////////////////////////////////////////////////////

  template < uint32_t dim, typename T > 
  SERAC_HOST_DEVICE view<T, dim> reshape(view<T, 1> v, stack::array< uint32_t, dim > new_dimensions, ordering o = ordering::row_major) {
    #ifdef NDARRAY_ENABLE_BOUNDS_CHECKING
      if (product(new_dimensions) != v.shape[0]) {
        printf("reshaping view into incompatible shape");
      };
    #endif

    auto strides = nd::compute_strides(new_dimensions, o, v.stride[0]); 

    return view<T,dim>{v.data(), new_dimensions, strides};
  }

  /////////////////////////////////////////////////////////////////////////////

};

double relative_error(nd::view<const double, 1> a, nd::view<const double,1> b);
double relative_error(nd::view<const double, 2> a, nd::view<const double,2> b);
double relative_error(nd::view<const double, 3> a, nd::view<const double,3> b);
double relative_error(nd::view<const double, 4> a, nd::view<const double,4> b);

namespace nd {
template < typename T > 
struct printer;

template <> struct printer<float>{ static SERAC_HOST_DEVICE void print(float x) { printf("%f", static_cast<double>(x)); } };
template <> struct printer<double>{ static SERAC_HOST_DEVICE void print(double x) { printf("%f", x); } };
template <> struct printer<int8_t>{ static SERAC_HOST_DEVICE void print(int8_t x) { printf("%d", x); } };
template <> struct printer<uint8_t>{ static SERAC_HOST_DEVICE void print(uint8_t x) { printf("%u", x); } };
template <> struct printer<int32_t>{ static SERAC_HOST_DEVICE void print(int32_t x) { printf("%d", x); } };
template <> struct printer<uint32_t>{ static SERAC_HOST_DEVICE void print(uint32_t x) { printf("%u", x); } };
template <> struct printer<int64_t>{ static SERAC_HOST_DEVICE void print(int64_t x) { printf("%ld", x); } };
template <> struct printer<uint64_t>{ static SERAC_HOST_DEVICE void print(uint64_t x) { printf("%lu", x); } };

template < typename T, uint32_t dim >
SERAC_HOST_DEVICE void print_recursive(nd::view< T, dim > arr, int depth) {
  using S = typename std::remove_const<T>::type;

  if constexpr (dim == 1) {
    for (int i = 0; i < depth; i++) printf("  ");
    printf("{");
    for (int i = 0; i < arr.shape[0]; i++) {
      printer<S>::print(arr(i));
      if (i != arr.shape[0] - 1) { printf(","); }
    }
    printf("}");
  } else {
    const T * ptr = arr.data();
    stack::array< uint32_t, dim - 1 > shape;
    stack::array< uint32_t, dim - 1 > stride;
    for (int i = 0; i < dim - 1; i++) {
      shape[i] = arr.shape[i+1];
      stride[i] = arr.stride[i+1];
    }
    nd::view<const T, dim - 1> slice{ptr, shape, stride};

    for (int i = 0; i < depth; i++) printf("  ");
    printf("{");
    if (arr.shape[0] == 1) {
      print_recursive(slice, 0);
    } else {
      printf("\n");
      for (int i = 0; i < arr.shape[0]; i++) {
        print_recursive(slice, depth+1);
        if (i != arr.shape[0] - 1) { printf(","); }
        printf("\n");
        slice.values += arr.stride[0];
      }
      for (int i = 0; i < depth; i++) printf("  ");
    }
    printf("}");
  }
  if (depth == 0) { printf("\n"); }
}

}

template < typename T, uint32_t dim >
SERAC_HOST_DEVICE void print(nd::view< T, dim > arr) {
  nd::print_recursive(arr, 0);
}

template < typename T, uint32_t dim >
std::ostream& operator<<(std::ostream & out, const nd::view< T, dim > & arr) {
  print(out, nd::view<const T, dim>(arr));
  return out;
}

template < typename T, uint32_t dim >
std::ostream& operator<<(std::ostream & out, const nd::array< T, dim > & arr) {
  print(out, nd::view<const T, dim>(arr));
  return out;
}
