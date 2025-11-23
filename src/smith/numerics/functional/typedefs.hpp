// Copyright (c) Lawrence Livermore National Security, LLC and
// other Smith Project Developers. See the top-level LICENSE file for
// details.
//
// SPDX-License-Identifier: (BSD-3-Clause)

#pragma once

namespace smith {

// sam: this is a kludge-- it looks like the DG spaces need to use some interface defined ONLY on
//      mfem::ParMesh / mfeme::ParFiniteElementSpace, but I'm not ready to pull the trigger on a big
//      interface change like that, so these typedefs mark the parts that would need to eventually change

/// @cond
using mesh_t = mfem::ParMesh;
using fes_t = mfem::ParFiniteElementSpace;
/// @endcond

}  // namespace smith
