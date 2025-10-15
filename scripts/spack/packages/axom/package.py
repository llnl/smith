# Copyright (c) Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

from spack.package import *
from spack_repo.builtin.packages.axom.package import Axom as BuiltinAxom

class Axom(BuiltinAxom):
    # Note: Make sure this sha coincides with the git submodule
    # Note: We add a number to the end of the real version number to indicate that we have
    #  moved forward past the release. Increment the last number when updating the commit sha.
    version("0.12.0.0", commit="297544010a3dfb98145a1a85f09f9c648c00a18c", submodules=False)
