# Copyright 2013-2024 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import os
import socket
from os.path import join as pjoin

from spack.package import *
from spack.util.executable import which_string

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

# NOTE: Cannot inherit from builtin until tribol becomes a builtin spack package within spack itself.

class Tribol(CachedCMakePackage, CudaPackage, ROCmPackage):
    """Tribol is an interface physics library."""

    homepage = "https://github.com/LLNL/Tribol"
    git      = "https://github.com/LLNL/Tribol.git"

    # SERAC EDIT START
    #version("develop", branch="develop", submodules=True, preferred=True)
    version("develop", branch="develop", submodules=True)
    # SERAC EDIT END


    # SERAC EDIT START
    # Note: We add a number to the end of the real version number to indicate that we have
    #  moved forward past the release. Increment the last number when updating the commit sha.
    version("0.1.0.17", commit="ed9247387bc54aab78362dd81339b6709bcaff51", submodules=True, preferred=True)
    # SERAC EDIT END

    # -----------------------------------------------------------------------
    # Variants
    # -----------------------------------------------------------------------
    variant("redecomp", default=True,
            description="Build redecomp domain redecomposition library")
    variant("fortran", default=False,
            description="Enable Fortran support")
    variant("tests", default=False,
            description="Build tests")
    variant("examples", default=False,
            description="Build examples")
    variant("devtools", default=False, 
            description="Build development tools (Sphinx, Doxygen, Shroud, clang-format)")
    variant("umpire",   default=False,
            description="Build with portable memory access support")
    variant("raja",     default=False,
            description="Build with portable kernel execution support")

    # -----------------------------------------------------------------------
    # Dependencies
    # -----------------------------------------------------------------------
    # Basic dependencies
    depends_on("cmake@3.14:", type="build")
    depends_on("cmake@3.21:", type="build", when="+rocm")
    depends_on("blt@0.6.2:", type="build")

    depends_on("mpi")

    # Other libraries
    depends_on("mfem+lapack")
    depends_on("axom@0.9:")

    depends_on("raja@2024.02.0:", when="+raja")
    depends_on("umpire@2024.02.0:", when="+umpire")

    with when("+redecomp"):
        depends_on("mfem+metis+mpi")
        depends_on("mfem+raja", when="+raja")
        depends_on("mfem+umpire", when="+umpire")
        depends_on("axom+raja", when="+raja")
        depends_on("axom+umpire", when="+umpire")

    for val in CudaPackage.cuda_arch_values:
        ext_cuda_dep = f"+cuda cuda_arch={val}"
        depends_on(f"mfem{ext_cuda_dep}", when=f"{ext_cuda_dep}")
        depends_on(f"axom{ext_cuda_dep}", when=f"{ext_cuda_dep}")
        depends_on(f"raja{ext_cuda_dep}", when=f"+raja{ext_cuda_dep}")
        depends_on(f"umpire{ext_cuda_dep}", when=f"+umpire{ext_cuda_dep}")

    for val in ROCmPackage.amdgpu_targets:
        ext_rocm_dep = f"+rocm amdgpu_target={val}"
        depends_on(f"mfem{ext_rocm_dep}", when=f"{ext_rocm_dep}")
        depends_on(f"axom{ext_rocm_dep}", when=f"{ext_rocm_dep}")
        depends_on(f"raja{ext_rocm_dep}", when=f"+raja{ext_rocm_dep}")
        depends_on(f"umpire{ext_rocm_dep}", when=f"+umpire{ext_rocm_dep}")

    depends_on("rocprim", when="+rocm")
    
    # Optional (require our variant in "when")
    for dep in ["raja", "umpire"]:
        depends_on("{0} build_type=Debug".format(dep), when="+{0} build_type=Debug".format(dep))
        
    # Required
    for dep in ["axom", "conduit", "metis", "parmetis"]:
        depends_on("{0} build_type=Debug".format(dep), when="build_type=Debug")

    # Required but not CMake
    for dep in ["hypre", "mfem"]:
        depends_on("{0}+debug".format(dep), when="build_type=Debug")

    # Devtool dependencies these need to match tribol_devtools/package.py
    depends_on("doxygen", when="+devtools")
    depends_on("python", when="+devtools")
    depends_on("py-shroud", when="+devtools+fortran")
    depends_on("py-sphinx", when="+devtools")
    depends_on("llvm+clang@10.0.0", when="+devtools", type="build")

    conflicts("+cuda", when="+rocm")

    def _get_sys_type(self, spec):
        sys_type = spec.architecture
        # if on llnl systems, we can use the SYS_TYPE
        if "SYS_TYPE" in env:
            sys_type = env["SYS_TYPE"]
        return sys_type

    @property
    def cache_name(self):
        hostname = socket.gethostname()
        if "SYS_TYPE" in env:
            # Are we on a LLNL system then strip node number
            hostname = hostname.rstrip('1234567890')
        special_case = ""
        if "+cuda" in self.spec:
            special_case += "_cuda"
        if "+fortran" in self.spec:
            special_case += "_fortran"
        if "+rocm" in self.spec:
            special_case += "_hip"
        return "{0}-{1}-{2}@{3}{4}.cmake".format(
            hostname,
            self._get_sys_type(self.spec),
            self.spec.compiler.name,
            self.spec.compiler.version,
            special_case,
        )

    def initconfig_compiler_entries(self):
        spec = self.spec
        entries = super(Tribol, self).initconfig_compiler_entries()

        if "+fortran" in spec:
            entries.append(cmake_cache_option("ENABLE_FORTRAN", True))
            if self.is_fortran_compiler("gfortran") and "clang" in self.compiler.cxx:
                libdir = pjoin(os.path.dirname(os.path.dirname(self.compiler.cxx)), "lib")
                flags = ""
                for _libpath in [libdir, libdir + "64"]:
                    if os.path.exists(_libpath):
                        if spec.satisfies('^cuda'):
                            flags += " -Xlinker -rpath -Xlinker {0}".format(_libpath)
                        else:
                            flags += " -Wl,-rpath,{0}".format(_libpath)
                description = "Adds a missing libstdc++ rpath"
                if flags:
                    entries.append(cmake_cache_string("BLT_EXE_LINKER_FLAGS", flags, description))
        else:
            entries.append(cmake_cache_option("ENABLE_FORTRAN", False))

        entries.append(cmake_cache_string("BLT_CXX_STD", "c++14", ""))

        # Add optimization flag workaround for Debug builds with
        # cray compiler or newer HIP
        if "+rocm" in spec:
            entries.append(cmake_cache_string("CMAKE_CXX_FLAGS_DEBUG","-O1 -g -DNDEBUG"))

        return entries

    def initconfig_hardware_entries(self):
        spec = self.spec
        entries = super(Tribol, self).initconfig_hardware_entries()

        if "+cuda" in spec:
            entries.append(cmake_cache_option("ENABLE_CUDA", True))
            entries.append(cmake_cache_option("CMAKE_CUDA_SEPARABLE_COMPILATION", True))

            # CUDA_FLAGS
            cudaflags = "${CMAKE_CUDA_FLAGS} -restrict --expt-extended-lambda "

            # Pass through any cxxflags to the host compiler via nvcc's Xcompiler flag
            host_cxx_flags = spec.compiler_flags["cxxflags"]
            cudaflags += " ".join(["-Xcompiler=%s " % flag for flag in host_cxx_flags])
            entries.append(cmake_cache_string("CMAKE_CUDA_FLAGS", cudaflags, force=True))

            entries.append("# nvcc does not like gtest's 'pthreads' flag\n")
            entries.append(cmake_cache_option("gtest_disable_pthreads", True))

        if "+rocm" in spec:
            entries.append("#------------------{0}".format("-" * 60))
            entries.append("# Tribol ROCm specifics")
            entries.append("#------------------{0}\n".format("-" * 60))

            entries.append(cmake_cache_option("ENABLE_HIP", True))

            hip_root = spec["hip"].prefix
            rocm_root = hip_root + "/.."

            # Fix blt_hip getting HIP_CLANG_INCLUDE_PATH-NOTFOUND bad include directory
            if (self.spec.satisfies('%cce') or self.spec.satisfies('%clang')) and 'toss_4' in self._get_sys_type(spec):
                # Set the patch version to 0 if not already
                clang_version= str(self.compiler.version)[:-1] + "0"
                hip_clang_include_path = rocm_root + "/llvm/lib/clang/" + clang_version + "/include"
                if os.path.isdir(hip_clang_include_path):
                    entries.append(cmake_cache_path("HIP_CLANG_INCLUDE_PATH", hip_clang_include_path))

            # Fixes for mpi for rocm until wrapper paths are fixed
            # These flags are already part of the wrapped compilers on TOSS4 systems
            hip_link_flags = ""
            if self.spec.satisfies('%clang'):
                # only with fortran for axom, but seems to always be needed for tribol
                hip_link_flags += "-Wl,--disable-new-dtags "
            if "+fortran" in spec and self.is_fortran_compiler("amdflang"):
                hip_link_flags += "-L{0}/../llvm/lib -L{0}/lib ".format(hip_root)
                hip_link_flags += "-Wl,-rpath,{0}/../llvm/lib:{0}/lib ".format(hip_root)
                hip_link_flags += "-lpgmath -lflang -lflangrti -lompstub -lamdhip64 "

            # Remove extra link library for crayftn
            if "+fortran" in spec and self.is_fortran_compiler("crayftn"):
                entries.append(cmake_cache_string("BLT_CMAKE_IMPLICIT_LINK_LIBRARIES_EXCLUDE",
                                                  "unwind"))

            # Additional libraries for TOSS4
            hip_link_flags += " -L{0}/../lib64 -Wl,-rpath,{0}/../lib64 ".format(hip_root)
            hip_link_flags += " -L{0}/../lib -Wl,-rpath,{0}/../lib ".format(hip_root)
            hip_link_flags += "-lamd_comgr -lhsa-runtime64 "

            entries.append(cmake_cache_string("CMAKE_EXE_LINKER_FLAGS", hip_link_flags))

        entries.append("#------------------{0}".format("-" * 60))
        entries.append("# Hardware Specifics")
        entries.append("#------------------{0}\n".format("-" * 60))

        if "+fortran" in spec and self.is_fortran_compiler("xlf"):
            # Grab lib directory for the current fortran compiler
            libdir = pjoin(os.path.dirname(os.path.dirname(self.compiler.fc)), "lib")
            description = (
                "Adds a missing rpath for libraries " "associated with the fortran compiler"
            )

            linker_flags = "${BLT_EXE_LINKER_FLAGS} -Wl,-rpath," + libdir

            entries.append(cmake_cache_string("BLT_EXE_LINKER_FLAGS", linker_flags, description))

            if "+shared" in spec:
                linker_flags = "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath," + libdir
                entries.append(
                    cmake_cache_string("CMAKE_SHARED_LINKER_FLAGS", linker_flags, description)
                )

            description = "Converts C-style comments to Fortran style in preprocessed files"
            entries.append(
                cmake_cache_string(
                    "BLT_FORTRAN_FLAGS", "-WF,-C!  -qxlf2003=polymorphic", description
                )
            )

        if spec.satisfies('target=ppc64le:'):
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
        entries = super(Tribol, self).initconfig_mpi_entries()

        entries.append(cmake_cache_option("ENABLE_MPI", True))
        if spec['mpi'].name == 'spectrum-mpi':
            entries.append(cmake_cache_string("BLT_MPI_COMMAND_APPEND",
                                              "mpibind"))

        # Replace /usr/bin/srun path with srun flux wrapper path on TOSS 4
        if 'toss_4' in self._get_sys_type(spec):
            srun_wrapper = which_string("srun")
            mpi_exec_index = [index for index,entry in enumerate(entries)
                                                  if "MPIEXEC_EXECUTABLE" in entry]
            del entries[mpi_exec_index[0]]
            entries.append(cmake_cache_path("MPIEXEC_EXECUTABLE", srun_wrapper))

        return entries

    def initconfig_package_entries(self):
        spec = self.spec
        entries = []

        entries.append("#------------------{0}".format("-" * 60))
        entries.append("# Options")
        entries.append("#------------------{0}\n".format("-" * 60))

        if "+redecomp" in spec:
            entries.append(cmake_cache_option("BUILD_REDECOMP", True))
        else:
            entries.append(cmake_cache_option("BUILD_REDECOMP", False))

        # TPL locations
        entries.append("#------------------{0}".format("-" * 60))
        entries.append("# TPLs")
        entries.append("#------------------{0}\n".format("-" * 60))

        path_replacements = {}

        # Try to find the common prefix of the TPL directory, including the
        # compiler. If found, we will use this in the TPL paths
        compiler_str = str(spec.compiler).replace('@','-')
        prefix_paths = prefix.split(compiler_str)
        tpl_root = ""
        if len(prefix_paths) == 2:
            tpl_root = os.path.join( prefix_paths[0], compiler_str )
            path_replacements[tpl_root] = "${TPL_ROOT}"
            entries.append(cmake_cache_path("TPL_ROOT", tpl_root))

        # required tpls
        for dep in ('axom', 'mfem'):
            dep_dir = get_spec_path(spec, dep, path_replacements)
            entries.append(cmake_cache_path('%s_DIR' % dep.upper(),
                                            dep_dir))

        # optional tpls
        for dep in ('raja', 'umpire'):
            if spec.satisfies('^{0}'.format(dep)):
                dep_dir = get_spec_path(spec, dep, path_replacements)
                entries.append(cmake_cache_path('%s_DIR' % dep.upper(),
                                                dep_dir))
            else:
                entries.append('# %s not built\n' % dep.upper())

        ##################################
        # Devtools
        ##################################

        entries.append("#------------------{0}".format("-" * 60))
        entries.append("# Devtools")
        entries.append("#------------------{0}\n".format("-" * 60))

        enable_docs = spec.satisfies('^doxygen') or spec.satisfies('^py-sphinx')
        entries.append(cmake_cache_option("TRIBOL_ENABLE_DOCS", enable_docs))

        if spec.satisfies('^py-sphinx'):
            sphinx_path = spec['py-sphinx'].prefix.bin.join('sphinx-build')
            entries.append(cmake_cache_path("SPHINX_EXECUTABLE", sphinx_path))

        if spec.satisfies('^py-shroud'):
            shroud_path = spec['py-shroud'].prefix.bin.join('shroud')
            entries.append(cmake_cache_path("SHROUD_EXECUTABLE", shroud_path))

        if spec.satisfies('^doxygen'):
            doxygen_path = spec['doxygen'].prefix.bin.join('doxygen')
            entries.append(cmake_cache_path("DOXYGEN_EXECUTABLE", doxygen_path))

        if spec.satisfies('^llvm') and 'toss_4' not in self._get_sys_type(spec):
            # Only turn on clangformat support if not on TOSS4
            clang_fmt_path = spec['llvm'].prefix.bin.join('clang-format')
            entries.append(cmake_cache_path("CLANGFORMAT_EXECUTABLE", clang_fmt_path))

        return entries


    def cmake_args(self):
        options = []

        options.append("-DBLT_SOURCE_DIR:PATH={0}".format(self.spec["blt"].prefix))

        options.append(self.define_from_variant(
            'TRIBOL_ENABLE_EXAMPLES', 'examples'))
        options.append(self.define_from_variant(
            'TRIBOL_ENABLE_TESTS', 'tests'))

        return options
