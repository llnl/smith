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
    version("4.10.0.1", commit="a57763ace9c9ff67a57cdb4457a6d7caf79eae93")

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


def _remove_inherited_gpu_strumpack_dependencies(package, base_package):
    """Allow GPU-enabled MFEM to use a CPU-only STRUMPACK dependency.

    The built-in MFEM package forces STRUMPACK to enable CUDA or ROCm whenever
    MFEM enables that GPU option. Remove those rules from Smith's MFEM package.
    """
    for mfem_condition, base_dependencies in base_package.dependencies.items():
        base_strumpack = base_dependencies.get("strumpack")
        if base_strumpack is None:
            continue

        # True when built-in MFEM has a GPU option enabled and forces STRUMPACK
        # to enable the same option.
        forces_gpu_option_on_strumpack = any(
            mfem_condition.satisfies(f"+{variant}")
            and base_strumpack.spec.satisfies(f"+{variant}")
            for variant in ("cuda", "rocm")
        )
        if not forces_gpu_option_on_strumpack:
            continue

        # Smith's MFEM package has its own dependency table containing both the
        # inherited rules and any Smith-specific rules. Remove the inherited
        # STRUMPACK rule only if Smith has not changed it.
        dependencies = package.dependencies.get(mfem_condition)
        strumpack = dependencies.get("strumpack") if dependencies is not None else None
        if strumpack is None or strumpack.spec != base_strumpack.spec:
            continue

        del dependencies["strumpack"]
        # Remove the condition itself if STRUMPACK was its only dependency.
        if not dependencies:
            del package.dependencies[mfem_condition]


# Spack has finished building Smith's MFEM dependency table at this point, so
# remove the inherited rules that unnecessarily require GPU-enabled STRUMPACK.
_remove_inherited_gpu_strumpack_dependencies(Mfem, BuiltinMfem)
