#pragma once

namespace refactor {

template <>
struct FiniteElement<mfem::Geometry::POINT, Family::H1> {
  using source_type = double;
  using flux_type = double;

  SERAC_HOST_DEVICE uint32_t num_nodes() const { return 1; }

  uint32_t p;
};

} // namespace refactor
