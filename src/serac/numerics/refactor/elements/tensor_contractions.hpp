#pragma once

#include <cassert>
#include "serac/numerics/refactor/containers/ndarray.hpp"

inline void contract(const nd::view<const double,3> & A, nd::view<double,2> & B, nd::view<double,3> C, bool accumulate = false) {
  //  A1(qx, iz, iy) = B(qx, ix) * in(iz, iy, ix) 
  //  A2(qy, qx, iz) = B(qy, iy) * A1(qx, iz, iy) 
  // out(qz, qy, qx) = B(qz, iz) * A2(qy, qx, iz) 
  uint32_t n0 = C.shape[0];
  uint32_t n1 = C.shape[1];
  uint32_t n2 = C.shape[2];
  uint32_t m = A.shape[2];
  for (uint32_t i0 = 0; i0 < n0; i0++) {
    for (uint32_t i1 = 0; i1 < n1; i1++) {
      for (uint32_t i2 = 0; i2 < n2; i2++) {
        double sum = (accumulate) ? C(i0, i1, i2) : 0.0;
        for (uint32_t j = 0; j < m; j++) {
          sum += B(i0, j) * A(i1, i2, j);
        }
        C(i0, i1, i2) = sum;
      }
    }
  }
}

/// C(qx, iy) {=,+=} sum_{ix} A(qx, ix) * B(iy, ix)  
inline void _contract(nd::view<double,2> C, const nd::view<const double,2> & A, const nd::view<const double,2> & B, bool accumulate = false) {
  uint32_t n0 = C.shape[0];
  uint32_t n1 = C.shape[1];
  uint32_t m = A.shape[1];
  assert(A.shape[0] == C.shape[0]);
  assert(B.shape[0] == C.shape[1]);
  assert(A.shape[1] == B.shape[1]);
  for (uint32_t i0 = 0; i0 < n0; i0++) {
    for (uint32_t i1 = 0; i1 < n1; i1++) {
      double sum = (accumulate) ? C(i0, i1) : 0.0;
      for (uint32_t j = 0; j < m; j++) {
        sum += A(i0, j) * B(i1, j);
      }
      C(i0, i1) = sum;
    }
  }
}

template < int i >
void contract(const nd::view<const double,2> & A, nd::view<double,2> & B, nd::view<double,2> C, bool accumulate = false) {
  uint32_t n0 = C.shape[0];
  uint32_t n1 = C.shape[1];
  uint32_t m = A.shape[i];
  assert(C.shape[1] == B.shape[0]);
  assert(A.shape[i] == B.shape[1]);
  assert(A.shape[1-i] == C.shape[0]);
  for (uint32_t i0 = 0; i0 < n0; i0++) {
    for (uint32_t i1 = 0; i1 < n1; i1++) {
      double sum = (accumulate) ? C(i0, i1) : 0.0;
      for (uint32_t j = 0; j < m; j++) {
        if constexpr (i == 0) { sum += A(j, i0) * B(i1, j); }
        if constexpr (i == 1) { sum += A(i0, j) * B(i1, j); }
      }
      C(i0, i1) = sum;
    }
  }
}

inline void contract(const nd::view<const double> & A, nd::view<double,2> & B, const nd::view<double> C, bool accumulate = false) {
  uint32_t n = C.shape[0];
  uint32_t m = A.shape[0];
  for (uint32_t i = 0; i < n; i++) {
    double sum = (accumulate) ? C(i) : 0.0;
    for (uint32_t j = 0; j < m; j++) {
      sum += A(j) * B(i, j);
    }
    C(i) = sum;
  }
}

#ifdef __CUDACC__ 
struct threadid {
  int x;
  int y;
  int z;
  int stride;
};

__device__ __forceinline__ void cuda_contract(threadid tid, const nd::view<double,3> & A, const nd::view<double,2> & B, nd::view<double,3> & C, bool accumulate = false) {
  uint32_t n0 = C.shape[0];
  uint32_t n1 = C.shape[1];
  uint32_t n2 = C.shape[2];
  uint32_t m = A.shape[2];

  for (uint32_t i0 = tid.z; i0 < n0; i0 += tid.stride) {
    for (uint32_t i1 = tid.y; i1 < n1; i1 += tid.stride) {
      for (uint32_t i2 = tid.x; i2 < n2; i2 += tid.stride) {
        double sum = (accumulate) ? C(i0, i1, i2) : 0.0;
        for (int j = 0; j < m; j++) {
          sum += B(i0, j) * A(i1, i2, j);
        }
        C(i0, i1, i2) = sum;
      }
    }
  }
  __syncthreads();
}

template < int i >
__device__ __forceinline__ void cuda_contract(threadid tid, const nd::view<double,2> & A, const nd::view<double,2> & B, nd::view<double,2> C, bool accumulate = false) {
  uint32_t n0 = C.shape[0];
  uint32_t n1 = C.shape[1];
  uint32_t m = A.shape[i];
  for (uint32_t i0 = tid.y; i0 < n0; i0 += tid.stride) {
    for (uint32_t i1 = tid.x; i1 < n1; i1 += tid.stride) {
      double sum = (accumulate) ? C(i0, i1) : 0.0;
      for (int j = 0; j < m; j++) {
        if constexpr (i == 0) { sum += A(j, i0) * B(i1, j); }
        if constexpr (i == 1) { sum += A(i0, j) * B(i1, j); }
      }
      C(i0, i1) = sum;
    }
  }
  __syncthreads();
}

///// C(qx, iy) {=,+=} sum_{ix} A(qx, ix) * B(iy, ix)  
//inline void _contract(nd::view<double,2> C, const nd::view<const double,2> & A, const nd::view<const double,2> & B, bool accumulate = false) {
//  uint32_t n0 = C.shape[0];
//  uint32_t n1 = C.shape[1];
//  uint32_t m = A.shape[1];
//  assert(A.shape[0] == C.shape[0]);
//  assert(B.shape[0] == C.shape[1]);
//  assert(A.shape[1] == B.shape[1]);
//  for (uint32_t i0 = 0; i0 < n0; i0++) {
//    for (uint32_t i1 = 0; i1 < n1; i1++) {
//      double sum = (accumulate) ? C(i0, i1) : 0.0;
//      for (int j = 0; j < m; j++) {
//        sum += A(i0, j) * B(i1, j);
//      }
//      C(i0, i1) = sum;
//    }
//  }
//}
//
//template < int i >
//void contract(const nd::view<const double,2> & A, nd::view<double,2> & B, nd::view<double,2> C, bool accumulate = false) {
//  uint32_t n0 = C.shape[0];
//  uint32_t n1 = C.shape[1];
//  uint32_t m = A.shape[i];
//  assert(C.shape[1] == B.shape[0]);
//  assert(A.shape[i] == B.shape[1]);
//  assert(A.shape[1-i] == C.shape[0]);
//  for (uint32_t i0 = 0; i0 < n0; i0++) {
//    for (uint32_t i1 = 0; i1 < n1; i1++) {
//      double sum = (accumulate) ? C(i0, i1) : 0.0;
//      for (int j = 0; j < m; j++) {
//        if constexpr (i == 0) { sum += A(j, i0) * B(i1, j); }
//        if constexpr (i == 1) { sum += A(i0, j) * B(i1, j); }
//      }
//      C(i0, i1) = sum;
//    }
//  }
//}
//
//inline void contract(const nd::view<const double> & A, nd::view<double,2> & B, const nd::view<double> C, bool accumulate = false) {
//  uint32_t n = C.shape[0];
//  uint32_t m = A.shape[0];
//  for (uint32_t i = 0; i < n; i++) {
//    double sum = (accumulate) ? C(i) : 0.0;
//    for (int j = 0; j < m; j++) {
//      sum += A(j) * B(i, j);
//    }
//    C(i) = sum;
//  }
//}
#endif
