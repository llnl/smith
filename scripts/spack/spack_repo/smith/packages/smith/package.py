# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

from spack.package import *
from spack.spec import UnsupportedCompilerError
from spack.util.executable import which_string

from spack_repo.builtin.build_systems.cached_cmake import (
    CachedCMakePackage,
    cmake_cache_option,
    cmake_cache_path,
    cmake_cache_string,
)
from spack_repo.builtin.build_systems.cuda import CudaPackage
from spack_repo.builtin.build_systems.rocm import ROCmPackage

import os
import socket
from os.path import join as pjoin


def get_spec_path(spec, package_name, path_replacements={}, use_bin=False):
    """Extracts the prefix path for the given spack package
       path_replacements is a dictionary with string replacements for the path.
    """

    if not use_bin:
        path = spec[package_name].prefix
    else:
        path = spec[package_name].prefix.bin

    path = os.path.realpath(path)

    for key in path_replacements:
        path = path.replace(key, path_replacements[key])

    return path


class Smith(CachedCMakePackage, CudaPackage, ROCmPackage):
    """Smith is a 3D implicit nonlinear thermal-structural simulation code.
       Its primary purpose is to investigate abstraction strategies and implicit
       finite element-based algorithm development for emerging computing architectures.
       It also heavily leverages the [MFEM finite element library](https://mfem.org/)."""

    maintainers("chapman39", "white238")

    homepage   = "https://www.github.com/LLNL/smith"
    git        = "https://github.com/LLNL/smith.git"
    submodules = True

    license("BSD-3-Clause")

    version("main", branch="main")
    version("develop", branch="develop")
    version("0.1.0", tag="v0.1.0", commit="cbe2e173eb1cbc36525ba454a386e4fe0b7e6440")

    # -----------------------------------------------------------------------
    # Variants
    # -----------------------------------------------------------------------
    # variants for build settings
    variant("shared", default=False,
            description="Enable build of shared libraries")
    variant("asan", default=False,
            description="Enable Address Sanitizer flags")
    variant("openmp", default=True,
            description="Enable OpenMP support")

    varmsg = "Build development tools (such as Sphinx, CppCheck, ClangFormat, etc...)"
    variant("devtools", default=False, description=varmsg)

    # variants for package dependencies
    variant("adiak", default=False, sticky=True,
            description="Build with adiak")
    variant("caliper", default=False, sticky=True,
            description="Build with caliper")
    variant("enzyme", default=False, sticky=True,
            description="Enable Enzyme Automatic Differentiation Framework")
    variant("petsc", default=True, sticky=True,
            description="Enable PETSc support")
    variant("raja", default=True, sticky=True,
            description="Build with portable kernel execution support")
    variant("slepc", default=True, sticky=True,
            description="Enable SLEPc integration")
    variant("strumpack", default=True, sticky=True,
            description="Build MFEM TPL with Strumpack, a direct linear solver library")
    variant("sundials", default=True, sticky=True,
            description="Build MFEM TPL with SUNDIALS nonlinear/ODE solver support")
    variant("tribol", default=True, sticky=True,
            description="Build Tribol, an interface physics library")
    variant("umpire", default=True, sticky=True,
            description="Build with portable memory access support")

    # -----------------------------------------------------------------------
    # Dependencies
    # -----------------------------------------------------------------------
    # Basic dependencies
    depends_on("c", type="build")
    depends_on("cxx", type="build")
    # Smith itself has no fortran but needs to pass a constrained fortran compiler
    # to its dependencies
    depends_on("fortran", type="build")

    depends_on("mpi")
    depends_on("cmake@3.14:")
    depends_on("cmake@3.21:", type="build", when="+rocm")

    depends_on("lua")

    depends_on("openblas", when="platform=darwin")

    depends_on("enzyme@0.0.180:", when="+enzyme")
    depends_on("cuda+allow-unsupported-compilers", when="+enzyme+cuda")
    depends_on("enzyme %libllvm=llvm-amdgpu", when="+enzyme+rocm")
    requires("%cxx=llvm-amdgpu", when="+enzyme+rocm")
    requires("%cxx=llvm", when="+enzyme~rocm")

    # Devtool dependencies these need to match smithdevtools/package.py
    with when("+devtools"):
        depends_on("cppcheck")
        depends_on("doxygen")
        depends_on("llvm@19+clang")
        depends_on("python")
        depends_on("py-sphinx")

    with when("+sundials"):
        # MFEM is deprecating the monitoring support with sundials v6.0 and later
        depends_on("sundials+hypre~trilinos~monitoring~examples~examples-install+static~shared~petsc")
        depends_on("sundials+asan", when="+asan")

    depends_on("mfem+netcdf+metis+superlu-dist+lapack+mpi")
    depends_on("mfem+sundials", when="+sundials")
    depends_on("mfem~sundials", when="~sundials")
    depends_on("mfem+amgx", when="+cuda")
    depends_on("mfem+asan", when="+asan")
    depends_on("mfem+strumpack", when="+strumpack")
    depends_on("mfem+petsc", when="+petsc")
    depends_on("mfem~petsc", when="~petsc")
    depends_on("mfem+slepc", when="+slepc")
    depends_on("mfem~slepc", when="~slepc")
    depends_on("mfem+openmp", when="+openmp")
    depends_on("mfem+enzyme", when="+enzyme")

    depends_on("netcdf-c")

    depends_on("hypre@2.26.0:~superlu-dist+mpi")

    with when("+petsc"):
        depends_on("petsc~mmg+metis fflags=-ffree-line-length-none")
        depends_on("petsc+strumpack", when="+strumpack")
        depends_on("petsc~strumpack", when="~strumpack")
        depends_on("petsc+openmp", when="+openmp")
        depends_on("petsc~openmp", when="~openmp")
        depends_on("slepc+arpack", when="+slepc")
        depends_on("arpack-ng", when="+slepc")

    with when("+tribol"):
        depends_on("tribol")
        depends_on("tribol+raja", when="+raja")
        depends_on("tribol~raja", when="~raja")
        depends_on("tribol+umpire", when="+umpire")
        depends_on("tribol~umpire", when="~umpire")
        depends_on("tribol+enzyme", when="+enzyme")
        depends_on("tribol~enzyme", when="~enzyme")

    depends_on("conduit+hdf5+mpi~python~test~silo")

    depends_on("hdf5+hl")

    depends_on("camp@2024.02.0:")

    with when("+raja"): 
        depends_on("raja@2024.02.0:~examples~exercises")
        depends_on("raja+openmp", when="+openmp")
        depends_on("raja~openmp", when="~openmp")

    with when("+umpire"):
        depends_on("umpire@2024.02.0:~examples~device_alloc")
        depends_on("umpire+openmp", when="+openmp")
        depends_on("umpire~openmp", when="~openmp")

    depends_on("axom@0.10:~fortran~tools~examples+mfem+lua")
    depends_on("axom+raja", when="+raja")
    depends_on("axom~raja", when="~raja")
    depends_on("axom+umpire", when="+umpire")
    depends_on("axom~umpire", when="~umpire")
    depends_on("axom+openmp", when="+openmp")
    depends_on("axom~openmp", when="~openmp")

    depends_on("metis@5.1.0")
    depends_on("parmetis@4.0.3")

    depends_on("adiak+mpi", when="+adiak")

    with when("+caliper"):
        depends_on("caliper+mpi~papi")
        depends_on("caliper+adiak", when="+adiak")
        # NOTE: Fixes undefined reference to symbol 'pthread_join@@GLIBC_2.2.5'
        depends_on("caliper cxxflags='-pthread'", when="+cuda ^caliper~shared")

    depends_on("superlu-dist@8.1.2")

    # The optional slate dependency is not handled in the MFEM spack package
    with when("+strumpack"):
        depends_on("strumpack~slate~butterflypack~zfp")
        depends_on("strumpack+openmp", when="+openmp")
        depends_on("strumpack~openmp", when="~openmp")

    #
    # CUDA
    #
    depends_on("amgx", when="+cuda")

    for val in CudaPackage.cuda_arch_values:
        ext_cuda_dep = f"+cuda cuda_arch={val}"

        # required
        depends_on(f"axom {ext_cuda_dep}", when=f"{ext_cuda_dep}")
        depends_on(f"mfem {ext_cuda_dep}", when=f"{ext_cuda_dep}")
        depends_on(f"hypre {ext_cuda_dep}", when=f"{ext_cuda_dep}")

        # optional
        depends_on(f"caliper {ext_cuda_dep}", when=f"+caliper {ext_cuda_dep}")
        depends_on(f"petsc {ext_cuda_dep}", when=f"+petsc {ext_cuda_dep}")
        depends_on(f"raja {ext_cuda_dep}", when=f"+raja {ext_cuda_dep}")
        depends_on(f"slepc {ext_cuda_dep}", when=f"+slepc {ext_cuda_dep}")
        depends_on(f"sundials {ext_cuda_dep}", when=f"+sundials {ext_cuda_dep}")
        depends_on(f"tribol {ext_cuda_dep}", when=f"+tribol {ext_cuda_dep}")
        depends_on(f"umpire {ext_cuda_dep}", when=f"+umpire {ext_cuda_dep}")

    #
    # ROCm
    #
    for val in ROCmPackage.amdgpu_targets:
        ext_rocm_dep = f"+rocm amdgpu_target={val}"

        # required
        depends_on(f"axom {ext_rocm_dep}", when=f"{ext_rocm_dep}")
        depends_on(f"mfem+raja+umpire {ext_rocm_dep}", when=f"{ext_rocm_dep}")
        depends_on(f"hypre {ext_rocm_dep}", when=f"{ext_rocm_dep}")

        # optional
        depends_on(f"caliper {ext_rocm_dep}", when=f"+caliper {ext_rocm_dep}")
        depends_on(f"petsc {ext_rocm_dep}", when=f"+petsc {ext_rocm_dep}")
        depends_on(f"raja {ext_rocm_dep}", when=f"+raja {ext_rocm_dep}")
        depends_on(f"slepc {ext_rocm_dep}", when=f"+slepc {ext_rocm_dep}")
        depends_on(f"sundials {ext_rocm_dep}", when=f"+sundials {ext_rocm_dep}")
        depends_on(f"tribol {ext_rocm_dep}", when=f"+tribol {ext_rocm_dep}")
        depends_on(f"umpire {ext_rocm_dep}", when=f"+umpire {ext_rocm_dep}")

    depends_on("rocprim", when="+rocm")
    depends_on("hipblas", when="+rocm")


    # -----------------------------------------------------------------------
    # Conflicts
    # -----------------------------------------------------------------------

    # Enzyme required an LLVM-based compiler
    for compiler_ in ["aocc", "cce", "gcc", "nag", "fj", "intel", "nvhpc", "xl"]:
        conflicts("+enzyme", when=f"%[virtuals=c,cxx] {compiler_}")

    requires("%cxx=llvm-amdgpu", when="+enzyme+rocm")
    requires("%cxx=llvm", when="+enzyme~rocm")

    conflicts("+openmp", when="+rocm")
    conflicts("~umpire", when="+raja", msg="Axom requires both raja and umpire in order to properly set CAMP_DIR.")
    conflicts("~petsc", when="+slepc", msg="PETSc must be built when building with SLEPc!")

    conflicts("+cuda", when="+rocm")
    conflicts("cuda_arch=none", when="+cuda", msg="CUDA architecture is required")
    conflicts("amdgpu_target=none", when="+rocm", msg="AMD GPU target is required when building with ROCm")

    conflicts("%intel", msg="Intel has a bug with C++17 support as of May 2020")

    # ASan is only supported by GCC and (some) LLVM-derived
    # compilers.
    asan_compiler_denylist = {"aocc", "arm", "cce", "fj", "intel", "nag",
                              "nvhpc", "oneapi", "pgi", "xl", "xl_r"}
    asan_compiler_allowlist = {"gcc", "clang", "apple-clang"}

    # ASan compiler denylist and allowlist should be disjoint.
    assert len(asan_compiler_denylist & asan_compiler_allowlist) == 0

    for compiler_ in asan_compiler_denylist:
        conflicts(
            "%{0}".format(compiler_),
            when="+asan",
            msg="{0} compilers do not support Address Sanitizer".format(compiler_)
        )


    def _get_sys_type(self, spec):
        sys_type = spec.architecture
        # if on llnl systems, we can use the SYS_TYPE
        if "SYS_TYPE" in env:
            sys_type = env["SYS_TYPE"]
        return sys_type


    def is_fortran_compiler(self, compiler):
        if self.compiler.fc is not None and compiler in self.compiler.fc:
            return True
        return False


    @property
    def cache_name(self):
        hostname = socket.gethostname()
        if "SYS_TYPE" in env:
            # Are we on a LLNL system then strip node number
            hostname = hostname.rstrip("1234567890")
        special_case = ""
        if "+cuda" in self.spec:
            special_case += "_cuda"
        if "+asan" in self.spec:
            special_case += "_asan"
        if "+rocm" in self.spec:
            special_case += "_hip"
        return "{0}-{1}-{2}@{3}{4}.cmake".format(
            hostname,
            self._get_sys_type(self.spec),
            self.spec.compiler.name,
            self.spec.compiler.version,
            special_case,
        )

    def initconfig_hardware_entries(self):
        spec = self.spec
        entries = super(Smith, self).initconfig_hardware_entries()

        entries.append(cmake_cache_option("ENABLE_OPENMP",
                                          spec.satisfies("+openmp")))

        if spec.satisfies("^cuda"):
            entries.append(cmake_cache_option("ENABLE_CUDA", True))
            entries.append(cmake_cache_option("CMAKE_CUDA_SEPARABLE_COMPILATION", True))

            # CXX flags will be propagated to the host compiler
            cxxflags = " ".join(spec.compiler_flags["cxxflags"])
            cuda_flags = cxxflags
            cuda_flags += " ${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr "
            entries.append(cmake_cache_string("CMAKE_CUDA_FLAGS", cuda_flags, force=True))

            entries.append("# nvcc does not like gtest's 'pthreads' flag\n")
            entries.append(cmake_cache_option("gtest_disable_pthreads", True))

        if spec.satisfies("+rocm"):
            entries.append(cmake_cache_option("ENABLE_HIP", True))

            hip_link_flags = ""

            rocm_root = spec["llvm-amdgpu"].prefix
            entries.append(cmake_cache_path("ROCM_ROOT_DIR", rocm_root))

            # Recommended MPI flags
            hip_link_flags += "-lxpmem "
            hip_link_flags += "-L/opt/cray/pe/mpich/{0}/gtl/lib ".format(spec["mpi"].version.up_to(3))
            hip_link_flags += "-Wl,-rpath,/opt/cray/pe/mpich/{0}/gtl/lib ".format(
                spec["mpi"].version.up_to(3)
            )
            hip_link_flags += "-lmpi_gtl_hsa "

            if spec.satisfies("^hip@6.0.0:"):
                hip_link_flags += "-L{0}/lib/llvm/lib -Wl,-rpath,{0}/lib/llvm/lib ".format(rocm_root)
            else:
                hip_link_flags += "-L{0}/llvm/lib -Wl,-rpath,{0}/llvm/lib ".format(rocm_root)
            # Only amdclang requires this path; cray compiler fails if this is included
            if spec.satisfies("%llvm-amdgpu"):
                hip_link_flags += "-L{0}/lib -Wl,-rpath,{0}/lib ".format(rocm_root)

            # Fixes for mpi for rocm until wrapper paths are fixed
            # These flags are already part of the wrapped compilers on TOSS4 systems
            if self.is_fortran_compiler("amdflang"):
                hip_link_flags += "-Wl,--disable-new-dtags "
                hip_link_flags += "-lflang -lflangrti "

            # Remove extra link library for crayftn
            if "+fortran" in spec and self.is_fortran_compiler("crayftn"):
                entries.append(
                    cmake_cache_string("BLT_CMAKE_IMPLICIT_LINK_LIBRARIES_EXCLUDE", "unwind")
                )

            # Additional libraries for TOSS4
            hip_link_flags += "-lamdhip64 -lhsakmt -lhsa-runtime64 -lamd_comgr -lpgmath "
            if spec.satisfies("+openmp"):
                hip_link_flags += "-lompstub "
            if spec.satisfies("^hipblas"):
                hip_link_flags += "-lhipblas "

            entries.append(cmake_cache_string("CMAKE_EXE_LINKER_FLAGS", hip_link_flags))

        if spec.satisfies("target=ppc64le:"):
            # Fix for working around CMake adding implicit link directories
            # returned by the BlueOS compilers to link executables with
            # non-system default stdlib
            _roots = ["/usr/tce/packages/gcc/gcc-4.9.3",
                      "/usr/tce/packages/gcc/gcc-4.9.3/gnu"]
            _subdirs = ["lib64",
                        "lib64/gcc/powerpc64le-unknown-linux-gnu/4.9.3"]
            _existing_paths = []
            for root in _roots:
                for subdir in _subdirs:
                    _curr_path = pjoin(root, subdir)
                    if os.path.exists(_curr_path):
                        _existing_paths.append(_curr_path)
            if _existing_paths:
                entries.append(cmake_cache_string(
                    "BLT_CMAKE_IMPLICIT_LINK_DIRECTORIES_EXCLUDE",
                    ";".join(_existing_paths)))

        return entries


    def initconfig_mpi_entries(self):
        spec = self.spec
        entries = super(Smith, self).initconfig_mpi_entries()

        entries.append(cmake_cache_option("ENABLE_MPI", True))
        if spec["mpi"].name == "spectrum-mpi":
            entries.append(cmake_cache_string("BLT_MPI_COMMAND_APPEND",
                                              "mpibind"))

        return entries

    def find_path_replacement(self, path1, path2, path_replacements, name, entries):
        root = os.path.commonprefix([path1, path2])
        if root.endswith(os.path.sep):
            root = root[:-len(os.path.sep)]
        if root:
            path_replacements[root] = "${" + name + "}"
            entries.append(cmake_cache_path(name, root))

    def initconfig_package_entries(self):
        spec = self.spec
        entries = []

        # TPL locations
        entries.append("#------------------{0}".format("-" * 60))
        entries.append("# TPLs")
        entries.append("#------------------{0}\n".format("-" * 60))

        path_replacements = {}

        # Try to find the common prefix of the TPL directory. 
        # If found, we will use this in the TPL paths
        path1 = os.path.realpath(spec["conduit"].prefix)
        path2 = os.path.realpath(self.prefix)
        self.find_path_replacement(path1, path2, path_replacements, "TPL_ROOT", entries)

        # required tpls
        # Note: lua is included in the case that axom is built via submodule
        for dep in ("axom", "camp", "conduit", "lua", "mfem", "hdf5",
                    "hypre", "metis", "parmetis"):
            dep_dir = get_spec_path(spec, dep, path_replacements)
            entries.append(cmake_cache_path("%s_DIR" % dep.upper(),
                                            dep_dir))

        dep_dir = get_spec_path(spec, "netcdf-c", path_replacements)
        entries.append(cmake_cache_path("NETCDF_DIR", dep_dir))

        dep_dir = get_spec_path(spec, "superlu-dist", path_replacements)
        entries.append(cmake_cache_path("SUPERLUDIST_DIR", dep_dir))

        if spec.satisfies("^arpack-ng"):
            dep_dir = get_spec_path(spec, "arpack-ng", path_replacements)
            entries.append(cmake_cache_path("ARPACK_DIR", dep_dir))

        # optional tpls
        for dep in ("adiak", "amgx", "caliper", "enzyme", "petsc", "raja", "slepc",
                    "strumpack", "sundials", "umpire", "tribol"):
            if spec.satisfies("^{0}".format(dep)):
                dep_dir = get_spec_path(spec, dep, path_replacements)
                entries.append(cmake_cache_path("%s_DIR" % dep.upper(),
                                                dep_dir))
            else:
                entries.append("# %s not built\n" % dep.upper())

        ##################################
        # Devtools
        ##################################

        entries.append("#------------------{0}".format("-" * 60))
        entries.append("# Devtools")
        entries.append("#------------------{0}\n".format("-" * 60))

        # Add common prefix to path replacement list
        if "+devtools" in spec:
            # Grab common devtools root and strip the trailing slash
            path1 = os.path.realpath(spec["cppcheck"].prefix)
            path2 = os.path.realpath(spec["doxygen"].prefix)
            self.find_path_replacement(path1, path2, path_replacements, "DEVTOOLS_ROOT", entries)

            # Only turn on clang tools support if devtools is on
            llvm_path = get_spec_path(spec, "llvm", path_replacements, use_bin=True)
            
            clang_fmt_path = pjoin(llvm_path, "clang-format")
            entries.append(cmake_cache_path("CLANGFORMAT_EXECUTABLE", clang_fmt_path))

            clang_tidy_path = pjoin(llvm_path, "clang-tidy")
            entries.append(cmake_cache_path("CLANGTIDY_EXECUTABLE", clang_tidy_path))
        else:
            entries.append("# Code checks disabled due to disabled devtools\n")
            entries.append(cmake_cache_option("SMITH_ENABLE_CODE_CHECKS", False))
            entries.append(cmake_cache_option("ENABLE_CLANGFORMAT", False))
            entries.append(cmake_cache_option("ENABLE_CLANGTIDY", False))

        enable_docs = spec.satisfies("^doxygen") or spec.satisfies("^py-sphinx")
        entries.append(cmake_cache_option("ENABLE_DOCS", enable_docs))

        if spec.satisfies("^py-sphinx"):
            sphinx_bin_dir = get_spec_path(spec, "py-sphinx",
                                           path_replacements,
                                           use_bin=True)
            entries.append(cmake_cache_path("SPHINX_EXECUTABLE",
                                            pjoin(sphinx_bin_dir,
                                                  "sphinx-build")))

        for dep in ("cppcheck", "doxygen"):
            if spec.satisfies("^{0}".format(dep)):
                dep_bin_dir = get_spec_path(spec, dep, path_replacements,
                                            use_bin=True)
                entries.append(cmake_cache_path("%s_EXECUTABLE" % dep.upper(),
                                                pjoin(dep_bin_dir, dep)))

        return entries


    def cmake_args(self):
        is_asan_compiler = self.compiler.name in self.asan_compiler_allowlist
        if self.spec.satisfies("+asan") and not is_asan_compiler:
            raise UnsupportedCompilerError(
                "Smith cannot be built with Address Sanitizer flags "
                "using {0} compilers".format(self.compiler.name)
            )

        options = []

        if self.run_tests is False:
            options.append("-DENABLE_TESTS=OFF")
        else:
            options.append("-DENABLE_TESTS=ON")

        options.append(self.define_from_variant(
            "BUILD_SHARED_LIBS", "shared"))

        options.append(self.define_from_variant(
            "ENABLE_ASAN", "asan"))

        return options
