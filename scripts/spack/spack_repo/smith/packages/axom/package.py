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
    version("0.14.0.1", commit="f81109cea1507cd9bbbd2f549c3fb33be18a3936", submodules=True, preferred=True)
