# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

from spack.package import *
from spack_repo.builtin.packages.axom.package import Axom as BuiltinAxom

class Axom(BuiltinAxom):
    """Axom provides a robust, flexible software infrastructure for the development
    of multi-physics applications and computational tools."""

    # Note: Make sure this sha coincides with the git submodule
    # Note: We add a number to the end of the real version number to indicate that we have
    # moved forward past the release. Increment the last number when updating the commit sha.
    version("0.14.0.2", commit="c9ece910855f61506a7cd3623a7f2fa4fe3505e0", submodules=True, preferred=True)
