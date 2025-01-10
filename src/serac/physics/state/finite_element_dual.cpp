#include "serac/physics/state/finite_element_dual.hpp"

namespace refactor {

Family get_family(const Residual & r) {
  switch(r.linearForm().FESpace()->FEColl()->GetContType()) {
    case mfem::FiniteElementCollection::CONTINUOUS:    return Family::H1;
    case mfem::FiniteElementCollection::TANGENTIAL:    return Family::HCURL;
    case mfem::FiniteElementCollection::NORMAL:        return Family::HDIV;
    case mfem::FiniteElementCollection::DISCONTINUOUS: return Family::L2;
  }
  return Family::H1; // unreachable
}

uint32_t get_degree(const Residual & r) {
  return uint32_t(r.linearForm().FESpace()->FEColl()->GetOrder());
}

uint32_t get_num_component(const Residual & r) {
  return uint32_t(r.linearForm().ParFESpace()->GetVDim());
}

uint32_t get_num_nodes(const Residual & r) {
  return uint32_t(r.linearForm().FESpace()->GetNDofs());
}

}
