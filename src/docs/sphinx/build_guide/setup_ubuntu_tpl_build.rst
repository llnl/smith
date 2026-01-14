.. ## Copyright (c) Lawrence Livermore National Security, LLC and
.. ## other Smith Project Developers. See the top-level COPYRIGHT file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)

.. _setup_ubuntu_tpl_build-label:

=========================
Setup Ubuntu 24 TPL Build
=========================


Install clang version 19 and make it the default compiler:

.. code-block:: bash

    sudo apt install -y --no-install-recommends clang-19 libclang-19-dev clang-format-19 llvm-19 llvm-19-dev libzstd-dev libomp-19-dev gfortran-13
    # Set clang-19 as the default clang
    sudo update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-19 101 \
    && sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-19 101 \
    && sudo update-alternatives --install /usr/bin/clang-format clang-format /usr/bin/clang-format-19 101

Install required build packages to minimize what Spack will build:

.. code-block:: bash

    sudo apt install -y --no-install-recommends build-essential bzip2 cmake libopenblas-dev \
    lua5.2 liblua5.2-dev openmpi-bin libopenmpi-dev unzip

Optionally you can install packages to generate documentation:

.. code-block:: bash

    sudo apt install -y --no-install-recommends graphviz python3-sphinx texlive-full doxygen

.. note::

    The documentation packages require a lot of disk space.

.. note::

    We provide a basic Ubuntu 24 Spack environment file in ``scripts/spack/configs/linux_ubuntu_24`` that
    may work for most people. If you want to try using that, skip to :ref:`build_tpls-label`
    below and use this command line option instead ``--spack-env-file=scripts/spack/configs/linux_ubuntu_24/spack.yaml``


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
we installed lua in the above ``apt`` commands. To do so, add the following lines under the ``packages:`` section of the yaml:

.. code-block:: yaml

    lua:
      externals:
      - spec: lua@5.2
        prefix: /usr
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
