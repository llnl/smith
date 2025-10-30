# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

from spack.package import *
from spack_repo.builtin.packages.amgx.package import Amgx as BuiltinAmgx

class Amgx(BuiltinAmgx):
    """AmgX provides a simple path to accelerated core solver technology on
    NVIDIA GPUs. AmgX provides up to 10x acceleration to the computationally
    intense linear solver portion of simulations, and is especially well
    suited for implicit unstructured methods. It is a high performance,
    state-of-the-art library and includes a flexible solver composition
    system that allows a user to easily construct complex nested solvers and
    preconditioners."""

    # white238: CUSPARSE_CSRMV_ALG2 undefined error in 2.3.0
    git = "https://github.com/NVIDIA/AMGX.git"
    version("2.3.0.1", commit="d2344958f43c103893c4400fe8ad42d02ac773f5", submodules=True)
