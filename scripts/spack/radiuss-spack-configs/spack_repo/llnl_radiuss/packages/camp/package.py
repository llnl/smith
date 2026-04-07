# Copyright 2013-2025 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import glob
import re

from os.path import dirname

from spack_repo.builtin.build_systems.cached_cmake import (
    cmake_cache_path,
    cmake_cache_string,
)
from spack_repo.builtin.build_systems.cmake import CMakePackage
from spack_repo.builtin.build_systems.cuda import CudaPackage
from spack_repo.builtin.build_systems.rocm import ROCmPackage

from spack.package import *
from spack.util.executable import which_string


def spec_uses_toolchain(spec):
    gcc_toolchain_regex = re.compile(".*gcc-toolchain.*")
    using_toolchain = list(filter(gcc_toolchain_regex.match, spec.compiler_flags["cxxflags"]))
    return using_toolchain

def spec_uses_gccname(spec):
    gcc_name_regex = re.compile(".*gcc-name.*")
    using_gcc_name = list(filter(gcc_name_regex.match, spec.compiler_flags["cxxflags"]))
    return using_gcc_name

def hip_for_radiuss_projects(options, spec, compiler):
    # adrienbernede-22-11:
    #   Specific to Umpire, attempt port to RAJA and CHAI
    rocm_root = dirname(spec["llvm-amdgpu"].prefix)
    hip_link_flags = ""
    if spec_uses_toolchain(spec):
        gcc_prefix = spec_uses_toolchain(spec)[0]
        options.append(cmake_cache_string("HIP_CLANG_FLAGS", "--gcc-toolchain={0}".format(gcc_prefix)))
        options.append(cmake_cache_string("CMAKE_EXE_LINKER_FLAGS", hip_link_flags + " -Wl,-rpath {}/lib64".format(gcc_prefix)))
    else:
        options.append(cmake_cache_string("CMAKE_EXE_LINKER_FLAGS", "-Wl,-rpath={0}/llvm/lib/".format(rocm_root)))

def cuda_for_radiuss_projects(options, spec):
    # Here is what is typically needed for radiuss projects when building with cuda

    # CUDA_FLAGS
    cuda_flags = []

    if not spec.satisfies("cuda_arch=none"):
        cuda_archs = ";".join(spec.variants["cuda_arch"].value)
        options.append(cmake_cache_string("CMAKE_CUDA_ARCHITECTURES", cuda_archs))

    if spec_uses_toolchain(spec):
        cuda_flags.append("-Xcompiler {}".format(spec_uses_toolchain(spec)[0]))

    if spec.satisfies("target=ppc64le %gcc@8.1:"):
        cuda_flags.append("-Xcompiler -mno-float128")

    options.append(cmake_cache_string("CMAKE_CUDA_FLAGS", " ".join(cuda_flags)))

def mpi_for_radiuss_projects(options, spec, env):

    if spec["mpi"].name == "spectrum-mpi" and spec.satisfies("^blt"):
        options.append(cmake_cache_string("BLT_MPI_COMMAND_APPEND", "mpibind"))

    sys_type = spec.architecture
    if "SYS_TYPE" in env:
        sys_type = env["SYS_TYPE"]
    # Replace /usr/bin/srun path with srun flux wrapper path on TOSS 4
    # TODO: Remove this logic by adding `using_flux` case in
    #  spack/lib/spack/spack/build_systems/cached_cmake.py:196 and remove hard-coded
    #  path to srun in same file.
    if "toss_4" in sys_type:
        srun_wrapper = which_string("srun")
        mpi_exec_index = [
            index for index, entry in enumerate(options) if "MPIEXEC_EXECUTABLE" in entry
        ]
        if len(mpi_exec_index) > 0:
            del options[mpi_exec_index[0]]
        mpi_exec_flag_index = [
            index for index, entry in enumerate(options) if "MPIEXEC_NUMPROC_FLAG" in entry
        ]
        if len(mpi_exec_flag_index) > 0:
            del options[mpi_exec_flag_index[0]]
        options.append(cmake_cache_path("MPIEXEC_EXECUTABLE", srun_wrapper))
        options.append(cmake_cache_string("MPIEXEC_NUMPROC_FLAG", "-n"))


class Camp(CMakePackage, CudaPackage, ROCmPackage):
    """
    Compiler agnostic metaprogramming library providing concepts,
    type operations and tuples for C++ and cuda
    """

    homepage = "https://github.com/LLNL/camp"
    git = "https://github.com/LLNL/camp.git"
    url = "https://github.com/LLNL/camp/archive/v0.1.0.tar.gz"

    maintainers("adrienbernede", "kab163", "trws")

    license("BSD-3-Clause")

    version("main", branch="main", submodules=False)
    version(
        "2025.03.0",
        tag="v2025.03.0",
        commit="ee0a3069a7ae72da8bcea63c06260fad34901d43",
        submodules=False,
    )
    version(
        "2024.07.0",
        tag="v2024.07.0",
        commit="0f07de4240c42e0b38a8d872a20440cb4b33d9f5",
        submodules=False,
    )
    version(
        "2024.02.1",
        tag="v2024.02.1",
        commit="79c320fa09db987923b56884afdc9f82f4b70fc4",
        submodules=False,
    )
    version(
        "2024.02.0",
        tag="v2024.02.0",
        commit="03c80a6c6ab4f97e76a52639563daec71435a277",
        submodules=False,
    )
    version(
        "2023.06.0",
        tag="v2023.06.0",
        commit="ac34c25b722a06b138bc045d38bfa5e8fa3ec9c5",
        submodules=False,
    )
    version("2022.10.1", sha256="2d12f1a46f5a6d01880fc075cfbd332e2cf296816a7c1aa12d4ee5644d386f02")
    version("2022.10.0", sha256="3561c3ef00bbcb61fe3183c53d49b110e54910f47e7fc689ad9ccce57e55d6b8")
    version("2022.03.2", sha256="bc4aaeacfe8f2912e28f7a36fc731ab9e481bee15f2c6daf0cb208eed3f201eb")
    version("2022.03.0", sha256="e9090d5ee191ea3a8e36b47a8fe78f3ac95d51804f1d986d931e85b8f8dad721")
    version("0.3.0", sha256="129431a049ca5825443038ad5a37a86ba6d09b2618d5fe65d35f83136575afdb")
    version("0.2.3", sha256="58a0f3bd5eadb588d7dc83f3d050aff8c8db639fc89e8d6553f9ce34fc2421a7")
    version("0.2.2", sha256="194d38b57e50e3494482a7f94940b27f37a2bee8291f2574d64db342b981d819")
    version("0.1.0", sha256="fd4f0f2a60b82a12a1d9f943f8893dc6fe770db493f8fae5ef6f7d0c439bebcc")

    # TODO: figure out gtest dependency and then set this default True.
    variant("tests", default=False, description="Build tests")
    variant("openmp", default=False, description="Build with OpenMP support")
    variant("omptarget", default=False, description="Build with OpenMP Target support")
    variant("sycl", default=False, description="Build with Sycl support")

    depends_on("c", type="build")
    depends_on("cxx", type="build")

    with when("+cuda"):
        depends_on("cub", when="^cuda@:10")

    depends_on("blt", type="build")
    depends_on("blt@0.6.2:", type="build", when="@2024.02.1:")
    depends_on("blt@0.6.1", type="build", when="@2024.02.0")
    depends_on("blt@0.5.0:0.5.3", type="build", when="@2022.03.0:2023.06.0")

    patch("libstdc++-13-missing-header.patch", when="@:2022.10")

    patch("camp-rocm6.patch", when="@0.2.3 +rocm ^hip@6:")

    conflicts("^blt@:0.3.6", when="+rocm")

    conflicts("+omptarget +rocm")
    conflicts("+sycl +omptarget")
    conflicts("+sycl +rocm")
    conflicts(
        "+sycl",
        when="@:2024.02.99",
        msg="Support for SYCL was introduced in RAJA after 2024.02 release, "
        "please use a newer release.",
    )

    def cmake_args(self):
        spec = self.spec

        options = []

        options.append("-DBLT_SOURCE_DIR={0}".format(spec["blt"].prefix))

        options.append(self.define_from_variant("ENABLE_CUDA", "cuda"))
        if spec.satisfies("+cuda"):
            options.append("-DCUDA_TOOLKIT_ROOT_DIR={0}".format(spec["cuda"].prefix))

            if not spec.satisfies("cuda_arch=none"):
                cuda_arch = spec.variants["cuda_arch"].value
                options.append("-DCMAKE_CUDA_ARCHITECTURES={0}".format(cuda_arch[0]))
                options.append("-DCUDA_ARCH=sm_{0}".format(cuda_arch[0]))
                flag = "-arch sm_{0}".format(cuda_arch[0])
                options.append("-DCMAKE_CUDA_FLAGS:STRING={0}".format(flag))

        options.append(self.define_from_variant("ENABLE_HIP", "rocm"))
        if spec.satisfies("+rocm"):
            rocm_root = dirname(spec["llvm-amdgpu"].prefix)
            options.append("-DROCM_PATH={0}".format(rocm_root))

            archs = ";".join(self.spec.variants["amdgpu_target"].value)
            options.append("-DCMAKE_HIP_ARCHITECTURES={0}".format(archs))
            options.append("-DGPU_TARGETS={0}".format(archs))
            options.append("-DAMDGPU_TARGETS={0}".format(archs))

        if spec.satisfies("+omptarget"):
            options.append(cmake_cache_string("RAJA_DATA_ALIGN", 64))

        options.append(self.define_from_variant("ENABLE_TESTS", "tests"))
        options.append(self.define_from_variant("ENABLE_OPENMP", "openmp"))
        options.append(self.define_from_variant("CAMP_ENABLE_TARGET_OPENMP", "omptarget"))
        options.append(self.define_from_variant("ENABLE_SYCL", "sycl"))

        return options
