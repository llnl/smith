# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

from spack.package import *
from spack_repo.builtin.packages.mfem.package import Mfem as BuiltinMfem

class Mfem(BuiltinMfem):

    # Note: Make sure this sha coincides with the git submodule
    # Note: We add a number to the end of the real version number to indicate that we have
    # moved forward past the release. Increment the last number when updating the commit sha.
    # Note: MFEM is not at 4.9, but 4.9 is required to enable enzyme in the spack package
    version("4.9.0.2", commit="563cd25971f1f86e72168fdc022a2472d08b4661")

    variant('asan', default=False, description='Add Address Sanitizer flags')

    depends_on("fortran", type="build", when="+strumpack")

    # AddressSanitizer (ASan) is only supported by GCC and (some) LLVM-derived
    # compilers. Denylist compilers not known to support ASan
    asan_compiler_denylist = {
        'aocc', 'arm', 'cce', 'fj', 'intel', 'nag', 'nvhpc', 'oneapi', 'pgi',
        'xl', 'xl_r'
    }

    # Allowlist of compilers known to support Address Sanitizer
    asan_compiler_allowlist = {'gcc', 'clang', 'apple-clang'}

    # ASan compiler denylist and allowlist should be disjoint.
    assert len(asan_compiler_denylist & asan_compiler_allowlist) == 0

    for compiler_ in asan_compiler_denylist:
        conflicts("%{0}".format(compiler_),
                  when="+asan",
                  msg="{0} compilers do not support Address Sanitizer".format(
                      compiler_))

    def setup_build_environment(self, env):
        BuiltinMfem.setup_build_environment(self, env)

        if '+asan' in self.spec:
            for flag in ("CFLAGS", "CXXFLAGS", "LDFLAGS"):
                env.append_flags(flag, "-fsanitize=address")

            for flag in ("CFLAGS", "CXXFLAGS"):
                env.append_flags(flag, "-fno-omit-frame-pointer")
                if '+debug' in self.spec:
                    env.append_flags(flag, "-fno-optimize-sibling-calls")


    # Override hypre make options to include extra rocm libs...
    # TODO remove once this PR merges into Spack https://github.com/spack/spack-packages/pull/2363
    def get_make_config_options(self, spec, prefix):
        options = BuiltinMfem.get_make_config_options(self, spec, prefix)

        # Remove old options
        options[:] = [opt for opt in options if "HYPRE_OPT" not in opt and "HYPRE_LIB" not in opt]

        # We need to add rpaths explicitly to allow proper export of link flags
        # from within MFEM. We use the following two functions to do that.
        ld_flags_from_library_list = self.ld_flags_from_library_list

        if "+mpi" in spec:
            hypre = spec["hypre"]
            all_hypre_libs = hypre.libs

            hypre_gpu_libs = ""
            if "+rocm" in hypre:
                hypre_rocm_libs = LibraryList([])
                if "^rocsparse" in hypre:
                    hypre_rocm_libs += hypre["rocsparse"].libs
                if "^rocrand" in hypre:
                    hypre_rocm_libs += hypre["rocrand"].libs
                # https://github.com/spack/spack-packages/pull/2363
                if hypre.version >= Version("2.29.0"):
                    if "^rocsolver" in hypre:
                        hypre_rocm_libs += hypre["rocsolver"].libs
                    if "^rocblas" in hypre:
                        hypre_rocm_libs += hypre["rocblas"].libs
                hypre_gpu_libs = " " + ld_flags_from_library_list(hypre_rocm_libs)
            options += [
                "HYPRE_OPT=-I%s" % hypre.prefix.include,
                "HYPRE_LIB=%s%s" % (ld_flags_from_library_list(all_hypre_libs), hypre_gpu_libs),
            ]

        return options
