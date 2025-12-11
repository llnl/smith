# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

from spack.package import *
from spack_repo.builtin.packages.petsc.package import Petsc as BuiltinPetsc

class Petsc(BuiltinPetsc):
    """Petsc"""

    # Fixes the following:
    # segmentedmempool.hpp(178): error: expression must be a modifiable lvalue
    patch("petsc_modifiable_lvalue.patch", when="@3.21.6")
