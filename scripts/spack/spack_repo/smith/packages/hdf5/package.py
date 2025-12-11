# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

from spack.package import *
from spack_repo.builtin.packages.hdf5.package import Hdf5 as BuiltinHDF5

class Hdf5(BuiltinHDF5):
    """HDF5 (Hierarchical Data Format 5) is a data format for HPC"""

    # Fixes incompatible pointer type error
    patch("hdf5_patch_fc41.txt", when="@1.8.23+tools%gcc@13:")
    patch("hdf5_patch_fc41.txt", when="@1.8.23+tools%clang@19:")
