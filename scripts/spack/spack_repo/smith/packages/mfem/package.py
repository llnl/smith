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
    version("4.10.0.2", commit="321c083f7974adde2114c03613cfde933c3841c1")

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

    def get_make_config_options(self, spec, prefix):
        options = BuiltinMfem.get_make_config_options(self, spec, prefix)

        if "+rocm" not in spec:
            return options

        # Smith does not use MFEM's hipBLAS batched linear algebra backend.
        # Keep the rest of MFEM's HIP support enabled without linking hipBLAS.
        options.append("MFEM_USE_HIPBLAS=NO")
        for index, option in enumerate(options):
            if option.startswith("HIP_LIB="):
                options[index] = " ".join(
                    flag for flag in option.split() if flag != "-lhipblas"
                )

        return options
