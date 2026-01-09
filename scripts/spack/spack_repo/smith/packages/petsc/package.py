# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

from spack.package import *
from spack_repo.builtin.packages.petsc.package import Petsc as BuiltinPetsc

# TODO remove this file once this PR merges https://github.com/spack/spack-packages/pull/2779

class Petsc(BuiltinPetsc):
    """Petsc"""

    # segmentedmempool.hpp(178): error: expression must be a modifiable lvalue
    # https://gitlab.com/petsc/petsc/-/merge_requests/8152
    patch("petsc_modifiable_lvalue.patch", when="@3.21.6:3.22.4+rocm")
    patch("petsc_modifiable_lvalue.patch", when="@3.21.6:3.22.4+cuda")
