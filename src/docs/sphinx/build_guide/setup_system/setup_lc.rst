.. ## Copyright (c) Lawrence Livermore National Security, LLC and
.. ## other Smith Project Developers. See the top-level COPYRIGHT file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)

.. _setup_lc-label:

========================================
Setup Livermore Computing (LC) TPL Build
========================================

We provide Spack Environment files for each of LC's systems:

 * TOSS4: ``scripts/spack/configs/toss_4_x86_64_ib/spack.yaml``
 * TOSS4 Cray: ``scripts/spack/configs/toss_4_x86_64_ib_cray/spack.yaml``

Unless otherwise specified, Spack will default to a compiler.  This is generally not a good idea when
developing large codes. To specify which compiler to use add the compiler specification to the ``--spec`` Uberenv
command line option. We provide recommended Spack specs for LC in ``scripts/spack/specs.json``.

You can use these directly in the ``uberenv.py`` command in the :ref:`build_tpls-label`
section by substituting the values in these two command line options: ``--spack-env-file=ubuntu24.yaml --spec="%clang_19"``.

.. note::
  On LC machines, it is good practice to do the build step in parallel on a compute node.
  You should add the following to the start of your commands: ``salloc -ppdebug -N1 --exclusive python3 scripts/uberenv/uberenv.py``

.. note::
   If you do not have access to the ``smithdev`` linux group. You cannot currently use our prebuilt Dev Tools
   referenced in the Spack Environment files listed above. You will be required to turn off the devtool variant
   on your Spack spec by adding ``~devtools`` to your uberenv or Spack spec.

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


To speed up the build, you can add packages that exist on your system and were not detected by Spack to the same Spack environment file.
For example, we installed lua in the above ``apt`` commands. To do so, add the following lines under the ``packages:`` section of the yaml:

.. code-block:: yaml

    lua:
      externals:
      - spec: lua@5.2
        prefix: /usr
      buildable: false


Lines starting with ``[e]`` are external packages that Spack recognizes are on the system and will not rebuild them.
By adding Lua to the Spack environment file, Spack will no longer build Lua and any of its dependencies that are
needed by anything else. In this case, ``lua``, ``readline``, and ``unzip`` will not be built.

.. important::

    Uberenv will override existing ``spack.yaml`` files in the current working directory. Now that we have made modifications,
    you should rename/move the file so the changes are not lost and adjust the `uberenv.py` commands to reflect the new file name.

--------------------------------------
Building Smith's Third-party Libraries
--------------------------------------

It is now time to build Smith's Third-party Libraries (TPLs). For detailed instructions see :ref:`build_tpls-label`.

