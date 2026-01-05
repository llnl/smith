# Copyright 2013-2025 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import os
from spack.package import *
from spack_repo.builtin.packages.llvm_amdgpu.package import LlvmAmdgpu as BuiltinLlvmAmdgpu

class LlvmAmdgpu(BuiltinLlvmAmdgpu):

    # PR that adds this change is pending: https://github.com/spack/spack-packages/pull/1557
    provides("fortran")

    # Fix from slack: 
    # https://spackpm.slack.com/archives/C08Q62S7XEX/p1751072888930439?thread_ts=1750704656.170759&cid=C08Q62S7XEX
    compiler_wrapper_link_paths = {
        "c": "rocmcc/amdclang",
        "cxx": "rocmcc/amdclang++",
        "fortran": "rocmcc/amdflang"
    }
