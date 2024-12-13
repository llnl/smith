#pragma once

#include "refactor/interpolation.hpp"

namespace refactor {

template <>
struct FiniteElement < mfem::Geometry::POINT, Family::Hcurl >{
  SERAC_HOST_DEVICE uint32_t num_nodes() const { return 0; }

  uint32_t p;
};

} // namespace refactor
