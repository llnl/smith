# Copyright Spack Project Developers. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack.package import *
from spack_repo.builtin.packages.enzyme.package import Enzyme as BuiltinEnzyme

import os

class Enzyme(BuiltinEnzyme):
    # Add newer enzyme versions not added to Spack package repo
    version("0.0.309", commit="a92e6b8f14c31045202bdabc28d9b6dbad25a4a3")

    @property
    def llvm_prefix(self):
        spec = self.spec
        if spec.satisfies("%libllvm=llvm"):
            return os.path.join(spec["llvm"].prefix)
        if spec.satisfies("%libllvm=llvm-amdgpu"):
            return os.path.join(spec["llvm-amdgpu"].prefix, "llvm")
        raise InstallError("Unknown 'libllvm' provider!")
