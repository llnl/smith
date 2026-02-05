# Copyright Spack Project Developers. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack.package import *
from spack_repo.builtin.packages.enzyme.package import Enzyme as BuiltinEnzyme

import os

class Enzyme(BuiltinEnzyme):
    # Add newer enzyme versions not added to Spack package repo
    version("0.0.249", commit="c3c973213c604028762dfbb30cf9f8ec9c83fc38")

    @property
    def llvm_prefix(self):
        spec = self.spec
        if spec.satisfies("%libllvm=llvm"):
            return os.path.join(spec["llvm"].prefix, "llvm")
        if spec.satisfies("%libllvm=llvm-amdgpu"):
            return os.path.join(spec["llvm-amdgpu"].prefix, "llvm")
        raise InstallError("Unknown 'libllvm' provider!")
