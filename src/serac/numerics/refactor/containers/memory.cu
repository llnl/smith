#include "nd/array.hpp"

#include <cuda.h>

#if NDARRAY_ENABLE_CUDA
#define error_check(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

namespace memory {

  void * allocate(uint64_t n) { 
    void * ptr;
    error_check(cudaMallocManaged(&ptr, n));
    return ptr;
  }

  void deallocate(void * ptr) {
    error_check(cudaFree(ptr));
  }

  void memcpy(void * dest, void * src, uint64_t n) { 
    error_check(cudaMemcpy(dest, src, n, cudaMemcpyDefault));
  }

  void zero(void * ptr, uint64_t n) {
    error_check(cudaMemset(ptr, 0, n));
  } 

}
#endif