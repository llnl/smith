# Copyright (c) Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

from spack.package import *

class SeracDevtools(BundlePackage):
    """This is a set of tools necessary for the developers of Serac"""

    version('fakeversion')

    depends_on('cmake')
    depends_on('cppcheck')
    depends_on('doxygen')
    depends_on('py-ats')
    depends_on('py-sphinx')
    depends_on('python')
    depends_on("llvm@14+clang+python")
