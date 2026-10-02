# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

from spack_repo.builtin.packages.strumpack.package import Strumpack as BuiltinStrumpack


class Strumpack(BuiltinStrumpack):

    def cmake_args(self):
        options = BuiltinStrumpack.cmake_args(self)

        # STRUMPACK's GPU setup cost dominates Smith's small direct solves.
        # Force its solver onto the CPU even when the surrounding DAG uses a
        # CUDA- or ROCm-enabled MFEM build.
        gpu_options = ("-DSTRUMPACK_USE_CUDA=", "-DSTRUMPACK_USE_HIP=")
        options = [
            option for option in options
            if not option.startswith(gpu_options)
        ]
        options.extend([
            "-DSTRUMPACK_USE_CUDA=OFF",
            "-DSTRUMPACK_USE_HIP=OFF",
        ])

        return options
