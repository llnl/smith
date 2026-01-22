# Copyright Spack Project Developers. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack.package import *
from spack_repo.builtin.packages.enzyme.package import Enzyme as BuiltinEnzyme

class Enzyme(BuiltinEnzyme):
    # Add newer enzyme versions not added to Spack package repo
    version("0.0.234", commit="8acdab5e870bf05e3b85f6d95d72877cb7dec470")
