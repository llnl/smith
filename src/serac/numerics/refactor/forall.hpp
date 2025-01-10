
#pragma once

#include <functional>

#include "serac/numerics/functional/tensor.hpp"
#include "serac/numerics/refactor/containers/ndarray.hpp"

namespace refactor {

template < typename ... T >
struct type_list{};

template < typename T >
struct FunctionSignature;

template < typename ret_t, typename ... arg_t >
struct FunctionSignature< ret_t (*)(arg_t ...) > {
    using return_type = ret_t;
    using argument_types = type_list< arg_t ... >;
};

template < typename obj_t, typename ret_t, typename ... arg_t >
struct FunctionSignature< ret_t (obj_t::*)(arg_t ...) const > {
    using return_type = ret_t;
    using argument_types = type_list< arg_t ... >;
};

using serac::vec;
using serac::mat;

template < typename T >
auto make_ndarray(uint32_t length, T) {
  return nd::array<T,1>({length}); 
}

auto make_ndarray(uint32_t length, double) {
  return nd::array<double,2>({length, 1}); 
}

template < int n >
auto make_ndarray(uint32_t length, vec<n>) {
  return nd::array<double,2>({length, uint32_t(n)}); 
}

template < int m, int n >
auto make_ndarray(uint32_t length, mat<m, n>) {
  return nd::array<double,3>({length, uint32_t(m), uint32_t(n)}); 
}

template < int m, int n, int p, int q >
auto make_ndarray(uint32_t length, mat<m, n, mat<p, q> >) {
  return nd::array<double,5>({length, m, n, p, q}); 
}

template < typename T >
static constexpr bool is_vec(T) { return false; }

template < typename T, uint32_t n >
static constexpr bool is_vec(vec<n,T>) { return true; }

template < typename T >
static constexpr bool is_mat(T) { return false; }

template < typename T, uint32_t m, int n >
static constexpr bool is_mat(mat<m,n,T>) { return true; }

template < typename T, uint32_t n, uint32_t rank >
auto load_vec(const nd::array< T, rank > & arr, int i, vec<n,T>) {
  vec<n,T> output;
  for (uint32_t j = 0; j < n; j++) {
    if constexpr (rank == 2) {
      output[j] = arr(i,j);
    }
    if constexpr (rank == 3) {
      output[j] = arr(i,0,j);
    }
  }
  return output;
}

template < typename T, uint32_t m, uint32_t n >
auto load_mat(const nd::array< T, 3 > & arr, int i, mat<m,n,T>) {
  mat<m,n,T> output;
  for (uint32_t j = 0; j < m; j++) {
    for (uint32_t k = 0; k < n; k++) {
      output(j,k) = arr(i,j,k);
    }
  }
  return output;
}

template < typename T, typename arr_type, uint32_t rank >
auto load(const nd::array< arr_type, rank > & arr, int i) {
  if constexpr (std::is_same<T, float>::value || 
                std::is_same<T, double>::value) {
    static_assert(rank == 1 || rank == 2 || rank == 3);
    if constexpr (rank == 1) { return arr(i); }
    if constexpr (rank == 2) { return arr(i,0); }
    if constexpr (rank == 3) { return arr(i,0,0); }
  }

  if constexpr (is_vec(T{})) {
    static_assert(rank == 2 || rank == 3);
    return load_vec(arr, i, T{});
  }

  if constexpr (is_mat(T{})) {
    static_assert(rank == 3);
    return load_mat(arr, i, T{});
  }
}

template < typename T, int n, uint32_t rank >
void save_vec(nd::array< T, rank > & arr, int i, const vec<n,T> & value) {
  for (uint32_t j = 0; j < n; j++) {
    if constexpr (rank == 2) {
      arr(i,j) = value[j];
    }
    if constexpr (rank == 3) {
      arr(i,0,j) = value[j];
    }
  }
}

template < typename T, int m, int n >
void save_mat(nd::array< T, 3 > & arr, uint32_t i, const mat<m,n,T> & value) {
  for (uint32_t j = 0; j < m; j++) {
    for (uint32_t k = 0; k < n; k++) {
      arr(i,j,k) = value(j,k);
    }
  }
}

template < typename T, typename arr_type, uint32_t rank >
auto save(nd::array< arr_type, rank > & arr, uint32_t i, const T & value) {
  if constexpr (std::is_same<T, float>::value || 
                std::is_same<T, double>::value) {
    if constexpr (rank == 1) arr(i) = value;
    if constexpr (rank == 2) arr(i, 0) = value;
  }

  if constexpr (is_vec(T{})) {
    static_assert(rank == 2 || rank == 3);
    save_vec(arr, i, value);
  }

  if constexpr (is_mat(T{})) {
    static_assert(rank == 3);
    save_mat(arr, i, value);
  }
}

namespace impl {

template < typename T, typename return_type, typename ... parameter_types >
auto forall(uint32_t n, const T & f, return_type * output, const parameter_types * ... args) {
  for (uint32_t i = 0; i < n; i++) {
    output[i] = f(args[i] ... );
  }
}

template < typename output_type, typename ... input_types, typename callable, typename ... arg_types >
auto forall(output_type, type_list< input_types ... >, callable func, const arg_types & ... args) {

  uint32_t leading_dimensions[] = {args.shape[0] ...};
  uint32_t n = leading_dimensions[0];
  auto output = make_ndarray(n, output_type{});

  {
    impl::forall(
      n, 
      func,
      reinterpret_cast< output_type * >(output.data()), 
      reinterpret_cast< const input_types * >(args.data()) ... 
    );
  }

  return output;
}

}

template < typename return_type, typename ... parameter_types, typename ... arg_types >
auto forall(std::function< return_type(const parameter_types & ...) > f, const arg_types & ... args) {

  uint32_t leading_dimensions[] = {args.shape[0] ...};
  uint32_t n = leading_dimensions[0];
  auto output = make_ndarray(n, return_type{});

  impl::forall(
    n, 
    f,
    reinterpret_cast< return_type * >(output.data()), 
    reinterpret_cast< const parameter_types * >(args.data()) ... 
  );

  return output;
}

template < typename return_type, typename ... parameter_types, typename ... arg_types >
auto forall(return_type (*f)(const parameter_types & ...), const arg_types & ... args) {

  uint32_t leading_dimensions[] = {args.shape[0] ...};
  uint32_t n = leading_dimensions[0];
  auto output = make_ndarray(n, return_type{});

  impl::forall(
    n, 
    f,
    reinterpret_cast< return_type * >(output.data()), 
    reinterpret_cast< const parameter_types * >(args.data()) ... 
  );

  return output;
}

template < typename callable, typename ... arg_types >
auto forall(callable func, const arg_types & ... args) {

  using signature = FunctionSignature< decltype(&callable::operator()) >;

  return impl::forall(
    typename signature::return_type{}, 
    typename signature::argument_types{}, 
    func, 
    args ...
  );

}

#ifdef __CUDACC__

template < typename functor, typename return_type, typename ... input_types >
__global__ void forall_kernel(
  uint32_t n, 
  functor f, 
  return_type * output, 
  const input_types * ... inputs) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n) {
    output[tid] = f(inputs[tid] ...); 
  }
}

namespace impl {

template < typename output_type, typename ... input_types, typename callable, typename ... arg_types >
auto cuda_forall(output_type, type_list< input_types ... >, callable func, const arg_types & ... args) {

  uint32_t leading_dimensions[] = {args.shape[0] ...};
  int n = leading_dimensions[0];
  auto output = make_ndarray(n, output_type{});

  {
    MTR_SCOPE("cuda_forall", "apply functor kernel");
    int blocksize = 128;
    int gridsize = (n + blocksize - 1) / blocksize;
    forall_kernel<<< gridsize, blocksize >>>(
      n, 
      func,
      reinterpret_cast< output_type * >(output.data()), 
      reinterpret_cast< const input_types * >(args.data()) ... 
    );
    cudaDeviceSynchronize();
  }

  return output;
}

}

template < typename callable, typename ... arg_types >
auto cuda_forall(callable func, const arg_types & ... args) {

  using signature = FunctionSignature< decltype(&callable::operator()) >;

  return impl::cuda_forall(
    typename signature::return_type{}, 
    typename signature::argument_types{}, 
    func, 
    args ...
  );

}
#endif

} // namespace refactor
