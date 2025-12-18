# Copyright 2013-2025 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import os
from spack.package import *
from spack_repo.builtin.packages.llvm_amdgpu.package import LlvmAmdgpu as BuiltinLlvmAmdgpu

# NOTE: We can remove this once/ if we switch to ROCm 7+ https://github.com/spack/spack-packages/pull/1655

class LlvmAmdgpu(BuiltinLlvmAmdgpu):

    # PR that adds this change is pending: https://github.com/spack/spack-packages/pull/1557
    provides("fortran")
