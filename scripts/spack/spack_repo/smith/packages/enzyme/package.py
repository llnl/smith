# Copyright Spack Project Developers. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack_repo.builtin.build_systems.cmake import CMakePackage

# NOTE: this custom package can be removed after this PR merges into Spack
# https://github.com/spack/spack-packages/pull/2365

from spack.package import *


class Enzyme(CMakePackage):
    """
    The Enzyme project is a tool for performing reverse-mode automatic
    differentiation (AD) of statically-analyzable LLVM IR.
    This allows developers to use Enzyme to automatically create gradients
    of their source code without much additional work.
    """

    homepage = "https://enzyme.mit.edu"
    url = "https://github.com/wsmoses/Enzyme/archive/v0.0.172.tar.gz"
    list_url = "https://github.com/wsmoses/Enzyme/releases"
    git = "https://github.com/wsmoses/Enzyme"

    maintainers("wsmoses", "vchuravy", "tgymnich")

    root_cmakelists_dir = "enzyme"

    version("main", branch="main")
    version("0.0.196", sha256="2b9cfcb7c34e56fc8191423042df06241cf32928eefbb113ac3c5199e3361cb2")
    version("0.0.186", sha256="125e612df0b6b82b07e1e13218c515bc54e04aa1407e57f4f31d3abe995f4714")
    version("0.0.180", sha256="d65a8e889413bb9518da00d65524c07352f1794b55c163f0db6828844c779ed4")
    version("0.0.173", sha256="b8477fb5bead9e9ece76d450ebd0afee99914235c6e1a6ef8c05bf288e3c0478")
    version("0.0.172", sha256="688200164787d543641cb446cff20f6a8e8b5c92bb7032ebe7f867efa67ceafb")
    version("0.0.135", sha256="49c798534faec7ba524a3ed053dd4352d690a44d3cad5a14915c9398dc9b175b")
    version("0.0.100", sha256="fbc53ec02adc0303ff200d7699afface2d9fbc7350664e6c6d4c527ef11c2e82")
    version("0.0.81", sha256="4c17d0c28f0572a3ab97a60f1e56bbc045ed5dd64c2daac53ae34371ca5e8b34")
    version("0.0.69", sha256="144d964187551700fdf0a4807961ceab1480d4e4cd0bb0fc7bbfab48fe053aa2")
    version("0.0.48", sha256="f5af62448dd2a8a316e59342ff445003581bc154f06b9b4d7a5a2c7259cf5769")
    version("0.0.32", sha256="9d42e42f7d0faf9beed61b2b1d27c82d1b369aeb9629539d5b7eafbe95379292")
    version("0.0.15", sha256="1ec27db0d790c4507b2256d851b256bf7e074eec933040e9e375d6e352a3c159")
    version("0.0.14", sha256="740641eeeeadaf47942ac88cc52e62ddc0e8c25767a501bed36ec241cf258b8d")
    version("0.0.13", sha256="d4a53964ec1f763772db2c56e6734269b7656c8b2ecd41fa7a41315bcd896b5a")

    depends_on("c", type="build")  # generated
    depends_on("cxx", type="build")  # generated
    depends_on("fortran", type="build")  # generated

    depends_on("libllvm@7:12", when="@0.0.13:0.0.15")
    depends_on("libllvm@7:14", when="@0.0.32:0.0.47")
    depends_on("libllvm@7:14", when="@0.0.48:0.0.68")
    depends_on("libllvm@9:16", when="@0.0.69:0.0.79")
    depends_on("libllvm@11:16", when="@0.0.80:0.0.99")
    depends_on("libllvm@11:19", when="@0.0.100:0.0.148")
    depends_on("libllvm@15:19", when="@0.0.149:0.0.185")
    depends_on("libllvm@15:20", when="@0.0.186:")
    depends_on("cmake@3.13:", type="build")

    for compiler_ in ["aocc", "cce", "gcc", "nag", "fj", "intel", "nvhpc", "xl"]:
        conflicts(f"%[virtuals=c,cxx] {compiler_}")

    def cmake_args(self):
        args = ["-DLLVM_DIR=" + self.llvm_prefix + "/lib/cmake/llvm"]
        return args

    @property
    def llvm_prefix(self):
        spec = self.spec
        if spec.satisfies("%libllvm=llvm"):
            return spec["llvm"].prefix
        if spec.satisfies("%libllvm=llvm-amdgpu"):
            return spec["llvm-amdgpu"].prefix
        raise InstallError("Unknown 'libllvm' provider!")

    @property
    def llvm_version(self):
        llvm_config = Executable(self.llvm_prefix + "/bin/llvm-config")
        return Version(llvm_config("--version", output=str))

    @property
    def libs(self):
        ver = self.llvm_version.up_to(1)
        libs = ["LLVMEnzyme-{0}".format(ver), "ClangEnzyme-{0}".format(ver)]
        if self.version >= Version("0.0.32"):  # TODO actual lower bound
            libs.append("LLDEnzyme-{0}".format(ver))

        return find_libraries(libs, root=self.prefix, recursive=True)

    def setup_dependent_build_environment(
        self, env: EnvironmentModifications, dependent_spec: Spec
    ) -> None:
        # Get the LLVMEnzyme, ClangEnzyme and LLDEnzyme lib paths and set
        # environment variables
        ver = self.llvm_version.up_to(1)

        llvm = find_libraries("LLVMEnzyme-{0}".format(ver), root=self.prefix, recursive=True)
        env.set("LLVMENZYME", ";".join(llvm))

        clang = find_libraries("ClangEnzyme-{0}".format(ver), root=self.prefix, recursive=True)
        env.set("CLANGENZYME", ";".join(clang))

        if self.version >= Version("0.0.32"):  # TODO actual lower bound
            lld = find_libraries("LLDEnzyme-{0}".format(ver), root=self.prefix, recursive=True)
            env.set("LLDENZYME", ";".join(lld))
