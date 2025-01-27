#pragma once

namespace serac {

// sam: this is a kludge-- it looks like the DG spaces need to use some interface defined ONLY on
//      mfem::ParMesh / mfeme::ParFiniteElementSpace, but I'm not ready to pull the trigger on a big
//      interface change like that, so these typedefs mark the parts that would need to eventually change

/// @cond
using mesh_t = mfem::Mesh;
using fes_t = mfem::FiniteElementSpace;
/// @endcond

}  // namespace serac
