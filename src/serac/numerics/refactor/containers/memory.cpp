#include "serac/numerics/refactor/containers/ndarray.hpp"

#ifndef NDARRAY_ENABLE_CUDA
namespace memory {

  void * allocate(uint64_t n) { 
    return malloc(n);
  }

  void deallocate(void * ptr) {
    free(ptr);
  }

  void memcpy(void * dest, void * src, uint64_t n) { 
    std::memcpy(dest, src, n);
  }

  void zero(void * ptr, uint64_t n) {
    std::memset(ptr, 0, n);
  } 

}
#endif
