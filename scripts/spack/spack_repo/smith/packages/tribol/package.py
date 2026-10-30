# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

from spack.package import *
from spack_repo.tribol.packages.tribol.package import Tribol as BuiltinTribol

class Tribol(BuiltinTribol):
    """Tribol is an interface physics library."""

    # NOTE: We add a number to the end of the real version number to indicate that we have
    # moved forward past the release. Increment the last number when updating the commit sha.
    version("0.1.0.29", commit="39befbed888ebd1dbb7da1988a76ea781013b41b", submodules=True, preferred=True)
