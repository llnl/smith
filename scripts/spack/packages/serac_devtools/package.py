# Copyright (c) Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

from spack.package import *

class SeracDevtools(BundlePackage):
    """This is a set of tools necessary for the developers of Serac"""

    version('fakeversion')

    variant('compiler_only', default=False, description="Build only required compiler with Enzyme")

    depends_on("llvm@19+clang")
    depends_on("llvm+python", when="~compiler_only")

    depends_on('cmake', when="~compiler_only")
    depends_on('cppcheck', when="~compiler_only")
    depends_on('doxygen', when="~compiler_only")
    # Disabled due to integration tests being disabled
    # depends_on('py-ats', when="~compiler_only")
    depends_on('py-sphinx', when="~compiler_only")
    depends_on('python', when="~compiler_only")
    

