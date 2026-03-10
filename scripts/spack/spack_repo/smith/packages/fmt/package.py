# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

from spack.package import *
from spack_repo.builtin.packages.fmt.package import Fmt as BuiltinFmt

class Fmt(BuiltinFmt):
  # Fix NVCC error
  patch("system_error_cuda.patch", when="@12.1.0: ^cuda")
