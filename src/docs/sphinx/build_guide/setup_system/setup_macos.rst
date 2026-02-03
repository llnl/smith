.. ## Copyright (c) Lawrence Livermore National Security, LLC and
.. ## other Smith Project Developers. See the top-level COPYRIGHT file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)

.. _setup_macos-label:

=====================
Setup macOS TPL Build
=====================

.. note::
   View an example host-config for MacOS in ``host-configs/other/firion-macos_sonoma_aarch64-<compiler>.cmake``.

Homebrew is recommended to install base dependencies. Relying on pure Spack historically leads to more failed builds.

To start, install the following packages using Homebrew.

.. code-block:: bash

   $ brew install autoconf automake bzip2 clingo cmake diffutils fmt gcc gettext gnu-sed graphviz hwloc lapack libx11 llvm@19 m4 make ninja open-mpi openblas pkg-config readline zlib

If you plan to install the developer tools, you should also run:

.. code-block:: bash

   $ brew install cppcheck doxygen
   $ ln -fs /opt/homebrew/opt/llvm@19/bin/clang-format /opt/homebrew/bin/clang-format

If you have installed Homebrew using the default installation prefix, most packages will be accessible through the prefix ``/opt/homebrew``.
Note for Intel-based Macs, the installation prefix is ``/usr/local``. If you set a custom prefix or aren't sure what the prefix is, run ``brew --prefix``.
For the rest of this section, we will assume the prefix is ``/opt/homebrew``.
Some packages are not linked into this prefix to prevent conflicts with MacOS-provided versions.
These will only be accessible via the prefix ``/opt/homebrew/opt/[package-name]``.
Homebrew will warn about such packages after installing them.

In order for the correct compilers to be used for the installation, you should also add the bin directory for LLVM clang to your path in your ``.bash_profile``, ``.bashrc``, or ``.zshrc``, etc.
This is also useful for a few additional packages:

.. code-block:: bash

   $ export PATH="/opt/homebrew/opt/llvm@19/bin:/opt/homebrew/opt/m4/bin:/opt/homebrew/opt/gnu-sed/libexec/gnubin:$PATH"

.. note::

    We provide a basic MacOS Spack environment file that
    may work for most people. If you want to try using that, skip to :ref:`build_tpls-label`
    below and use this command line option instead ``--spack-env-file=scripts/spack/configs/darwin/spack.yaml``. You will likely
    need to update the versions of packages to match the versions installed by Homebrew. The versions for all installed packages can be listed via
    the command ``brew list --versions``.


Given that Homebrew can only install CMake version 4.0 and it breaks some TPL builds (e.g. metis), its recommended to install an older version of CMake
manually. You can do this by downloading from `CMake's official archive <https://cmake.org/files/v3.23/cmake-3.23.5-macos-universal.dmg>`_. After installing
CMake 3.23, you will need to specify the path in the Spack environment like so:

.. code-block:: yaml

    cmake:
      version: [3.23.5]
      buildable: false
      externals:
      - spec: cmake@3.23.5
        prefix: /Applications/CMake.app/Contents

Optionally, you can install the developer tools via ``pip``. This step is only required if you wish to use Smith's developer tools.
In order to use Python devtools, you will need to create a Python venv. This is much more reliable than having Spack install 20+ Python packages.
In this example, we are using the builtin Python in ``/usr/bin``, but it is possible to use a version installed from Homebrew or elsewhere.
Install wheel and Sphinx:

.. code-block:: bash

   python3 -m venv venv
   source venv/bin/activate
   pip install wheel sphinx
   sphinx-build --version

Keep track of the Sphinx version while installing, since you'll need it for the next step.

To have Spack recognize your pre-installed Developer Tools, you should add the following under ``packages`` in the ``spack.yaml`` files.
Versions and prefixes may vary.

.. code-block:: yaml

    # Devtools (optional)
    cppcheck:
      version: [2.15.0]
      buildable: false
      externals:
      - spec: cppcheck@2.15.0
        prefix: /opt/homebrew
    doxygen:
      version: [1.12.0]
      buildable: false
      externals:
      - spec: doxygen@1.12.0
        prefix: /opt/homebrew
    py-sphinx:
      buildable: false
      externals:
      - spec: py-sphinx@7.4.7
        prefix: /path/to/venv


-------------------------------
Generate Spack Environment File
-------------------------------

Spack uses an environment file, or ``spack.yaml``, to describe where system level packages are to minimize what it builds.
This file describes the compilers and associated flags required for the platform as well as the low-level libraries
on the system to prevent Spack from building the world. Documentation on these environment files is located
in the `Spack docs <https://spack.readthedocs.io/en/latest/environments.html>`_.

The following command will download the specific Spack version we use and run a minimal set of commands to
generate an environment file for you. This should be a good starting point and should be used in following
Spack builds.

.. code-block:: bash

    ./scripts/uberenv/uberenv.py --prefix=<path/outside/repository> --setup-and-env-only

This command will create a Spack environment file, ``spack.yaml``, where you ran the above command.
If you want to use Clang as your compiler, alter the following section in that file by changing
``null`` in the ``f77`` and ``fc`` lines to ``/usr/bin/gfortran``:

.. code-block:: yaml

    - compiler:
        spec: clang@=19.1.1
        paths:
            cc: /usr/bin/clang
            cxx: /usr/bin/clang++
            f77: null # Change null to /usr/bin/gfortran
            fc: null # and this one too
        flags: {}
        operating_system: ubuntu24.04
        target: x86_64
        modules: []
        environment: {}
        extra_rpaths: []


To speed up the build, you can add packages that exist on your system to the same Spack environment file. For example,
you can install lua via homebrew with this command, ``brew install lua``. Then you can add it as a Spack external
in the ``packages:`` section of the Spack Environment yaml file:

.. code-block:: yaml

    lua:
      externals:
      - spec: lua@5.2
        prefix: /opt/homebrew
      buildable: false

The above spack command will output a concretization that looks like the following:

.. code-block:: shell

    ==> Concretized 1 spec:
     -   uu3sgzv  smith@develop cflags=-fPIC cxxflags=-fPIC fflags=-fPIC ~asan~cuda~devtools+enzyme~ipo+openmp+petsc~profiling+raja~rocm~shared+slepc+strumpack+sundials+tribol+umpire build_system=cmake build_type=Release dev_path=/home/white238/projects/smith/repo generator=make arch=linux-ubuntu24.04-skylake %clang@19.1.1
     -   x77izrn      ^axom@0.10.1.1 cflags=-fPIC cxxflags=-fPIC fflags=-fPIC +cpp14~cuda~devtools~examples~fortran+hdf5~ipo+lua+mfem+mpi~opencascade+openmp~python+raja~rocm~scr~shared~tools+umpire build_system=cmake build_type=Release generator=make arch=linux-ubuntu24.04-skylake %clang@19.1.1
     -   prysqkw          ^blt@0.6.2 cflags=-fPIC cxxflags=-fPIC fflags=-fPIC  build_system=generic arch=linux-ubuntu24.04-skylake %clang@19.1.1
     -   6vi46wm      ^camp@2024.07.0 cflags=-fPIC cxxflags=-fPIC fflags=-fPIC ~cuda~ipo~omptarget+openmp~rocm~sycl~tests build_system=cmake build_type=Release generator=make arch=linux-ubuntu24.04-skylake %clang@19.1.1
    [e]  fbcccfh      ^cmake@3.28.3 cflags=-fPIC cxxflags=-fPIC fflags=-fPIC ~doc+ncurses+ownlibs~qtgui build_system=generic build_type=Release patches=dbc3892 arch=linux-ubuntu24.04-skylake %clang@19.1.1
     -   zqg3svf      ^conduit@0.9.3 cflags=-fPIC cxxflags=-fPIC fflags=-fPIC ~adios+blt_find_mpi~caliper~doc~doxygen+examples~fortran+hdf5+hdf5_compat~ipo+mpi+parmetis~python+shared~silo~test+utilities~zfp build_system=cmake build_type=Release generator=make arch=linux-ubuntu24.04-skylake %clang@19.1.1
     -   lbmoj2n      ^enzyme@0.0.180 cflags=-fPIC cxxflags=-fPIC fflags=-fPIC ~ipo build_system=cmake build_type=Release generator=make arch=linux-ubuntu24.04-skylake %clang@19.1.1
    [e]  6bbmbqw          ^llvm@19.1.1 cflags=-fPIC cxxflags=-fPIC fflags=-fPIC +clang~cuda~flang+gold+libomptarget~libomptarget_debug~link_llvm_dylib~lld~lldb+llvm_dylib+lua~mlir+offload+polly~python~split_dwarf~z3~zstd build_system=cmake build_type=Release compiler-rt=runtime generator=ninja libcxx=runtime libunwind=runtime openmp=runtime shlib_symbol_version=none targets=all version_suffix=none arch=linux-ubuntu24.04-skylake %clang@19.1.1
    [e]  wyizjq2      ^glibc@2.39 cflags=-fPIC cxxflags=-fPIC fflags=-fPIC  build_system=autotools arch=linux-ubuntu24.04-skylake %clang@19.1.1
    [e]  74zxzg7      ^gmake@4.3 cflags=-fPIC cxxflags=-fPIC fflags=-fPIC ~guile build_system=generic patches=599f134 arch=linux-ubuntu24.04-skylake %clang@19.1.1
     -   32xbf3o      ^hdf5@1.8.23 cflags=-fPIC cxxflags=-fPIC fflags=-fPIC ~cxx~fortran+hl~ipo~mpi~shared~szip~threadsafe+tools api=default build_system=cmake build_type=Release generator=make patches=f42732a arch=linux-ubuntu24.04-skylake %clang@19.1.1
     -   hlqkvfc          ^pkgconf@2.3.0 cflags=-fPIC cxxflags=-fPIC fflags=-fPIC  build_system=autotools arch=linux-ubuntu24.04-skylake %clang@19.1.1
     -   dlqs5c6          ^zlib-ng@2.2.3 cflags=-fPIC cxxflags=-fPIC fflags=-fPIC +compat~new_strategies+opt+pic+shared build_system=autotools arch=linux-ubuntu24.04-skylake %clang@19.1.1
     -   iwu2tah      ^hypre@2.32.0 cflags=-fPIC cxxflags=-fPIC fflags=-fPIC ~caliper~complex~cublas~cuda~debug+fortran~gptune~gpu-aware-mpi~int64~internal-superlu+lapack~magma~mixedint+mpi~openmp~rocblas~rocm~shared~superlu-dist~sycl~umpire~unified-memory build_system=autotools precision=double arch=linux-ubuntu24.04-skylake %clang@19.1.1
     -   g6pkuqj          ^openblas@0.3.29 cflags=-fPIC cxxflags=-fPIC fflags=-fPIC ~bignuma~consistent_fpcsr+dynamic_dispatch+fortran~ilp64+locking+pic+shared build_system=makefile patches=9968625 symbol_suffix=none threads=openmp arch=linux-ubuntu24.04-skylake %clang@19.1.1
    [e]  naati2q      ^lua@5.2 cflags=-fPIC cxxflags=-fPIC fflags=-fPIC +shared build_system=makefile fetcher=curl arch=linux-ubuntu24.04-skylake %clang@19.1.1
     -   5u3dj5i      ^metis@5.1.0 cflags=-fPIC cxxflags=-fPIC fflags=-fPIC ~gdb~int64~ipo~no_warning~real64~shared build_system=cmake build_type=Release generator=make patches=4991da9,93a7903 arch=linux-ubuntu24.04-skylake %clang@19.1.1
     -   ae2mqqr      ^mfem@4.8.0.1 cflags=-fPIC cxxflags=-fPIC fflags=-fPIC ~amgx~asan~conduit~cuda~debug~examples~exceptions~fms~ginkgo~gnutls~gslib~hiop+lapack~libceed~libunwind+metis~miniapps~mpfr+mpi~mumps+netcdf~occa+openmp+petsc~pumi~raja~rocm~shared+slepc+static+strumpack~suite-sparse+sundials+superlu-dist~threadsafe~umpire+zlib build_system=generic cxxstd=auto precision=double timer=auto arch=linux-ubuntu24.04-skylake %clang@19.1.1
     -   ohbx2dl      ^netcdf-c@4.7.4 cflags=-fPIC cxxflags=-fPIC fflags=-fPIC ~blosc~byterange~dap~fsync~hdf4~jna~logging~mpi~nczarr_zip+optimize~parallel-netcdf+pic~shared~szip~zstd build_system=autotools arch=linux-ubuntu24.04-skylake %clang@19.1.1
    [e]  yr5mrlv      ^openmpi@4.1.6 cflags=-fPIC cxxflags=-fPIC fflags=-fPIC +atomics~cuda+cxx~cxx_exceptions~debug~gpfs~internal-hwloc~internal-libevent~internal-pmix+java~legacylaunchers~lustre~memchecker~openshmem~orterunprefix+pmi+romio+rsh~singularity~static~two_level_namespace+vt~wrapper-rpath build_system=autotools fabrics=ofi,psm,psm2,ucx romio-filesystem=none schedulers=slurm arch=linux-ubuntu24.04-skylake %clang@19.1.1
     -   2zrmzi6      ^parmetis@4.0.3 cflags=-fPIC cxxflags=-fPIC fflags=-fPIC ~gdb~int64~ipo~shared build_system=cmake build_type=Release generator=make patches=4f89253,50ed208,704b84f arch=linux-ubuntu24.04-skylake %clang@19.1.1
     -   sxzqrlk      ^petsc@3.22.4 cflags=-fPIC cxxflags=-fPIC fflags=-fPIC ~X~batch~cgns~complex~cuda~debug+double~exodusii~fftw+fortran~giflib~hdf5~hpddm~hwloc+hypre~int64~jpeg~knl~kokkos~libpng~libyaml~memkind+metis~mkl-pardiso~mmg~moab~mpfr+mpi~mumps+openmp~p4est~parmmg~ptscotch~random123~rocm~saws~scalapack~shared+strumpack~suite-sparse+superlu-dist~sycl~tetgen~trilinos~valgrind~zoltan build_system=generic clanguage=C memalign=none arch=linux-ubuntu24.04-skylake %clang@19.1.1
    [e]  baweecy          ^diffutils@3.10 cflags=-fPIC cxxflags=-fPIC fflags=-fPIC  build_system=autotools arch=linux-ubuntu24.04-skylake %clang@19.1.1
     -   evwepln          ^netlib-scalapack@2.2.2 cflags=-fPIC cxxflags=-fPIC fflags=-fPIC ~ipo~pic+shared build_system=cmake build_type=Release generator=make arch=linux-ubuntu24.04-skylake %clang@19.1.1
    [e]  timms67          ^python@3.12.3 cflags=-fPIC cxxflags=-fPIC fflags=-fPIC +bz2+crypt+ctypes+dbm~debug+libxml2+lzma~optimizations+pic+pyexpat~pythoncmd+readline+shared+sqlite3+ssl~tkinter+uuid+zlib build_system=generic arch=linux-ubuntu24.04-skylake %clang@19.1.1
     -   cc22mbv      ^raja@2024.07.0 cflags=-fPIC cxxflags=-fPIC fflags=-fPIC ~cuda~desul~examples~exercises~ipo~lowopttest~omptarget~omptask+openmp~plugins~rocm~run-all-tests~shared~sycl~tests+vectorization build_system=cmake build_type=Release generator=make arch=linux-ubuntu24.04-skylake %clang@19.1.1
     -   zmiasfe      ^slepc@3.22.2 cflags=-fPIC cxxflags=-fPIC fflags=-fPIC +arpack~blopex~cuda~hpddm~rocm build_system=generic arch=linux-ubuntu24.04-skylake %clang@19.1.1
     -   fvqm3cm          ^arpack-ng@3.9.1 cflags=-fPIC cxxflags=-fPIC fflags=-fPIC ~icb~ipo+mpi+shared build_system=cmake build_type=Release generator=make arch=linux-ubuntu24.04-skylake %clang@19.1.1
     -   3ka3cml      ^strumpack@8.0.0 cflags=-fPIC cxxflags=-fPIC fflags=-fPIC ~butterflypack+c_interface~count_flops~cuda~ipo~magma+mpi+openmp+parmetis~rocm~scotch~shared~slate~task_timers~zfp build_system=cmake build_type=Release generator=make arch=linux-ubuntu24.04-skylake %clang@19.1.1
     -   mkzronq      ^sundials@6.7.0 cflags=-fPIC cxxflags=-fPIC fflags=-fPIC +ARKODE+CVODE+CVODES+IDA+IDAS+KINSOL~asan~cuda~examples~examples-install~f2003~fcmix+generic-math~ginkgo+hypre~int64~ipo~klu~kokkos~kokkos-kernels~lapack~magma~monitoring+mpi~openmp~petsc~profiling~pthread~raja~rocm~shared+static~superlu-dist~superlu-mt~sycl~trilinos build_system=cmake build_type=Release cstd=99 cxxstd=14 generator=make logging-level=2 logging-mpi=OFF precision=double arch=linux-ubuntu24.04-skylake %clang@19.1.1
     -   vqgxdhb      ^superlu-dist@8.1.2 cflags=-fPIC cxxflags=-fPIC fflags=-fPIC ~cuda~int64~ipo~openmp+parmetis~rocm~shared build_system=cmake build_type=Release generator=make arch=linux-ubuntu24.04-skylake %clang@19.1.1
     -   ldxkes6      ^tribol@0.1.0.18 cflags=-fPIC cxxflags=-fPIC fflags=-fPIC ~asan~cuda~devtools~examples~fortran~ipo+openmp+raja+redecomp~rocm~tests+umpire build_system=cmake build_type=Release generator=make arch=linux-ubuntu24.04-skylake %clang@19.1.1
     -   3ftpxxz      ^umpire@2024.07.0 cflags=-fPIC cxxflags=-fPIC fflags=-fPIC ~asan~backtrace+c~cuda~dev_benchmarks~device_alloc~deviceconst~examples+fmt_header_only~fortran~ipc_shmem~ipo~mpi~numa~omptarget+openmp~rocm~sanitizer_tests~shared~sqlite_experimental~tools+werror build_system=cmake build_type=Release generator=make tests=none arch=linux-ubuntu24.04-skylake %clang@19.1.1
     -   xts3eqq          ^fmt@11.0.2 cflags=-fPIC cxxflags=-fPIC fflags=-fPIC ~ipo+pic~shared build_system=cmake build_type=Release cxxstd=11 generator=make arch=linux-ubuntu24.04-skylake %clang@19.1.1


Lines starting with ``[e]`` are external packages that Spack recognizes are on the system and will not rebuild them.
By adding Lua to the Spack environment file, Spack will no longer build Lua and any of its dependencies that are
needed by anything else. In this case, ``lua``, ``readline``, and ``unzip`` will not be built. ``unzip`` may be needed
by another package, so you can also add it with this yaml section:

.. code-block:: yaml

    unzip:
      externals:
      - spec: unzip@6.0
        prefix: /usr
      buildable: false

.. important::

    Uberenv will override existing ``spack.yaml`` files in the current working directory. Now that we have made modifications,
    you should rename/move the file so the changes are not lost and adjust the `uberenv.py` commands to reflect the new file name.

--------------------------------------
Building Smith's Third-party Libraries
--------------------------------------

It is now time to build Smith's Third-party Libraries (TPLs). For detailed instructions see :ref:`build_tpls-label`.
