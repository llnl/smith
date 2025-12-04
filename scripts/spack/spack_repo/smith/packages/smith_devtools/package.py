# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

from spack_repo.builtin.build_systems.bundle import BundlePackage
from spack.package import *

class SmithDevtools(BundlePackage):
    """This is a set of tools necessary for the developers of Smith"""

    version('fakeversion')

    depends_on('cmake')
    depends_on('cppcheck')
    depends_on('doxygen')
    # Disabled due to integration tests being disabled
    # depends_on('py-ats')
    depends_on('py-sphinx')
    depends_on('python')
    depends_on("llvm@19+clang+python")
