.. ## Copyright (c) Lawrence Livermore National Security, LLC and
.. ## other Serac Project Developers. See the top-level COPYRIGHT file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)

.. _ubuntu_build-label:

=====================
Serac Ubuntu 24 Build
=====================

------------------
Basic System Setup
------------------

We recommend installing some basic system-level development packages to minimize the
amount of packages that Spack will build.

Install clang version 14 and make it the default compiler:

.. code-block:: bash

    sudo apt install -y --no-install-recommends clang-14 clang-format-14 llvm-14 libomp-14-dev gfortran-13
    # Set clang-14 as the default clang
    sudo update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-14 100 \
    && sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-14 100 \
    && sudo update-alternatives --install /usr/bin/clang-format clang-format /usr/bin/clang-format-14 100

Install required build packages to minimize what Spack will build:

.. code-block:: bash

    sudo apt install -y --no-install-recommends build-essential bzip2 cmake libopenblas-dev \
    lua5.2 liblua5.2-dev openmpi-bin libopenmpi-dev unzip

Optionally you can install packages to generate documentation:

.. code-block:: bash

    sudo apt install -y --no-install-recommends graphviz python3-sphinx texlive-full doxygen

.. note::

    The documentation packages require a lot of disk space.

-------------------------------
Generate Spack Environment File
-------------------------------

Spack uses an environment file to describe where system level packages are to minimize what it builds.
The following command will download the specific Spack version we use and run a minimal set of commands to
generate an environment file for you. This should be a good starting point and should be used in following
Spack builds.

.. note::

    We provide a basic Ubuntu 24 Spack environment file in ``scripts/spack/configs/linux_ubuntu_24`` that
    may work for most people. If you want to try using that, skip to :ref:`Build Serac's Third-party Libraries <ubuntu_tpl_build-label>`
    below and use this command line option instead ``--spack-env-file=scripts/spack/configs/linux_ubuntu_24/ubuntu24.yaml``

.. code-block:: bash

    scripts/uberenv/uberenv.py --prefix=<path/outside/repository> --setup-and-env-only

This command will create a Spack environment file, ``spack.yaml``, where you ran the above command.
If you want to use Clang as your compiler. Alter the following section in that file, by changing
``null`` in the ``f77`` and ``fc`` lines to ``/usr/bin/gfortran``:

.. code-block:: yaml

    - compiler:
        spec: clang@=14.0.6
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


If you are using the GNU compiler, you can ignore the above step.

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
    -   56woyw5  serac@develop%gcc@13.3.0~asan~cuda~devtools~ipo+openmp+petsc~profiling+raja~rocm~shared+slepc+strumpack+sundials+tribol+umpire build_system=cmake build_type=Release dev_path=/home/white238/projects/serac/repo generator=make arch=linux-ubuntu24.04-skylake
    -   nrxx23t      ^axom@0.10.1.1%gcc@13.3.0+cpp14~cuda~devtools~examples~fortran+hdf5~ipo+lua+mfem+mpi~opencascade+openmp~python+raja~rocm~scr~shared~tools+umpire build_system=cmake build_type=Release generator=make arch=linux-ubuntu24.04-skylake
    -   nqkgp22          ^blt@0.6.2%gcc@13.3.0 build_system=generic arch=linux-ubuntu24.04-skylake
    -   rj42gk3      ^camp@2024.07.0%gcc@13.3.0~cuda~ipo~omptarget+openmp~rocm~sycl~tests build_system=cmake build_type=Release generator=make arch=linux-ubuntu24.04-skylake
    [e]  zw5kcoy         ^cmake@3.28.3%gcc@13.3.0~doc+ncurses+ownlibs~qtgui build_system=generic build_type=Release patches=dbc3892 arch=linux-ubuntu24.04-skylake
    -   syv5u67      ^conduit@0.9.2%gcc@13.3.0~adios+blt_find_mpi~caliper~doc~doxygen+examples~fortran+hdf5+hdf5_compat~ipo+mpi+parmetis~python+shared~silo~test+utilities~zfp build_system=cmake build_type=Release generator=make arch=linux-ubuntu24.04-skylake
    -   oaqv5fr      ^gcc-runtime@13.3.0%gcc@13.3.0 build_system=generic arch=linux-ubuntu24.04-skylake
    [e]  jozorw3         ^glibc@2.39%gcc@13.3.0 build_system=autotools arch=linux-ubuntu24.04-skylake
    [e]  xulikhw         ^gmake@4.3%gcc@13.3.0~guile build_system=generic patches=599f134 arch=linux-ubuntu24.04-skylake
    -   ywot3zz      ^hdf5@1.8.23%gcc@13.3.0~cxx~fortran+hl~ipo~mpi~shared~szip~threadsafe+tools api=default build_system=cmake build_type=Release generator=make patches=f42732a arch=linux-ubuntu24.04-skylake
    -   orkrbck          ^pkgconf@2.3.0%gcc@13.3.0 build_system=autotools arch=linux-ubuntu24.04-skylake
    -   sjsn4ra          ^zlib-ng@2.2.3%gcc@13.3.0+compat~new_strategies+opt+pic+shared build_system=autotools arch=linux-ubuntu24.04-skylake
    -   ojsmmb5      ^hypre@2.32.0%gcc@13.3.0~caliper~complex~cublas~cuda~debug+fortran~gptune~gpu-aware-mpi~int64~internal-superlu+lapack~magma~mixedint+mpi~openmp~rocblas~rocm~shared~superlu-dist~sycl~umpire~unified-memory build_system=autotools precision=double arch=linux-ubuntu24.04-skylake
    -   qwjnrd5          ^openblas@0.3.29%gcc@13.3.0~bignuma~consistent_fpcsr+dynamic_dispatch+fortran~ilp64+locking+pic+shared build_system=makefile symbol_suffix=none threads=openmp arch=linux-ubuntu24.04-skylake
    -   rndoezc      ^lua@5.4.6%gcc@13.3.0+shared build_system=makefile fetcher=curl arch=linux-ubuntu24.04-skylake
    [e]  x2me7ec          ^curl@8.5.0%gcc@13.3.0+gssapi+ldap~libidn2~librtmp~libssh~libssh2+nghttp2 build_system=autotools libs=shared,static tls=openssl arch=linux-ubuntu24.04-skylake
    [e]  arxo7wy          ^ncurses@6.4.20240113%gcc@13.3.0~symlinks+termlib abi=6 build_system=autotools patches=7a351bc arch=linux-ubuntu24.04-skylake
    -   vg2hsrt           ^readline@8.2%gcc@13.3.0 build_system=autotools patches=1ea4349,24f587b,3d9885e,5911a5b,622ba38,6c8adf8,758e2ec,79572ee,a177edc,bbf97f1,c7b45ff,e0013d9,e065038 arch=linux-ubuntu24.04-skylake
    -   4s5yarf           ^unzip@6.0%gcc@13.3.0 build_system=makefile patches=881d2ed,f6f6236 arch=linux-ubuntu24.04-skylake
    -   o5f4jyb      ^metis@5.1.0%gcc@13.3.0~gdb~int64~ipo~no_warning~real64~shared build_system=cmake build_type=Release generator=make patches=4991da9,93a7903,b1225da arch=linux-ubuntu24.04-skylake
    -   qlkm3gq      ^mfem@4.8.0.1%gcc@13.3.0~amgx~asan~conduit~cuda~debug~examples~exceptions~fms~ginkgo~gnutls~gslib~hiop+lapack~libceed~libunwind+metis~miniapps~mpfr+mpi~mumps+netcdf~occa+openmp+petsc~pumi~raja~rocm~shared+slepc+static+strumpack~suite-sparse+sundials+superlu-dist~threadsafe~umpire+zlib build_system=generic cxxstd=auto precision=double timer=auto arch=linux-ubuntu24.04-skylake
    [e]  saipzs5          ^openmpi@4.1.6%gcc@13.3.0+atomics~cuda+cxx~cxx_exceptions~debug~gpfs~internal-hwloc~internal-libevent~internal-pmix+java~legacylaunchers~lustre~memchecker~openshmem~orterunprefix+pmi+romio+rsh~singularity~static~two_level_namespace+vt~wrapper-rpath build_system=autotools fabrics=ofi,psm,psm2,ucx romio-filesystem=none schedulers=slurm arch=linux-ubuntu24.04-skylake    -   pp3ewgz      ^netcdf-c@4.7.4%gcc@13.3.0~blosc~byterange~dap~fsync~hdf4~jna~logging~mpi~nczarr_zip+optimize~parallel-netcdf+pic~shared~szip~zstd build_system=autotools arch=linux-ubuntu24.04-skylake
    -   d4dxhlq      ^parmetis@4.0.3%gcc@13.3.0~gdb~int64~ipo~shared build_system=cmake build_type=Release generator=make patches=4f89253,50ed208,704b84f arch=linux-ubuntu24.04-skylake
    -   7jbj4ll      ^petsc@3.22.3%gcc@13.3.0~X~batch~cgns~complex~cuda~debug+double~exodusii~fftw+fortran~giflib~hdf5~hpddm~hwloc+hypre~int64~jpeg~knl~kokkos~libpng~libyaml~memkind+metis~mkl-pardiso~mmg~moab~mpfr+mpi~mumps+openmp~p4est~parmmg~ptscotch~random123~rocm~saws~scalapack~shared+strumpack~suite-sparse+superlu-dist~sycl~tetgen~trilinos~valgrind~zoltan build_system=generic clanguage=C memalign=none arch=linux-ubuntu24.04-skylake
    [e]  hx35vpd          ^diffutils@3.10%gcc@13.3.0 build_system=autotools arch=linux-ubuntu24.04-skylake
    -   v3wd4md           ^netlib-scalapack@2.2.2%gcc@13.3.0~ipo~pic+shared build_system=cmake build_type=Release generator=make arch=linux-ubuntu24.04-skylake
    [e]  ebf563j          ^python@3.12.3%gcc@13.3.0+bz2+crypt+ctypes+dbm~debug+libxml2+lzma~optimizations+pic+pyexpat~pythoncmd+readline+shared+sqlite3+ssl~tkinter+uuid+zlib build_system=generic arch=linux-ubuntu24.04-skylake
    -   i3mwd3s      ^raja@2024.07.0%gcc@13.3.0~cuda~desul~examples~exercises~ipo~lowopttest~omptarget~omptask+openmp~plugins~rocm~run-all-tests~shared~sycl~tests+vectorization build_system=cmake build_type=Release generator=make arch=linux-ubuntu24.04-skylake
    -   ku2ueig      ^slepc@3.22.2%gcc@13.3.0+arpack~blopex~cuda~hpddm~rocm build_system=generic arch=linux-ubuntu24.04-skylake
    -   ch6sfkd           ^arpack-ng@3.9.1%gcc@13.3.0~icb~ipo+mpi+shared build_system=cmake build_type=Release generator=make arch=linux-ubuntu24.04-skylake
    -   c6negoy      ^strumpack@8.0.0%gcc@13.3.0~butterflypack+c_interface~count_flops~cuda~ipo~magma+mpi+openmp+parmetis~rocm~scotch~shared~slate~task_timers~zfp build_system=cmake build_type=Release generator=make arch=linux-ubuntu24.04-skylake
    -   bg2x2yp      ^sundials@7.2.1%gcc@13.3.0+ARKODE+CVODE+CVODES+IDA+IDAS+KINSOL~asan~cuda~examples~examples-install~f2003~fcmix~ginkgo+hypre~int64~ipo~klu~kokkos~kokkos-kernels~lapack~magma~monitoring+mpi~openmp~petsc~profiling~pthread~raja~rocm~shared+static~superlu-dist~superlu-mt~sycl~trilinos build_system=cmake build_type=Release cstd=99 cxxstd=14 generator=make logging-level=2 precision=double arch=linux-ubuntu24.04-skylake
    -   5jrfz2j      ^superlu-dist@8.1.2%gcc@13.3.0~cuda~int64~ipo~openmp+parmetis~rocm~shared build_system=cmake build_type=Release generator=make arch=linux-ubuntu24.04-skylake
    -   fz6b2hd      ^tribol@0.1.0.18%gcc@13.3.0~asan~cuda~devtools~examples~fortran~ipo+openmp+raja+redecomp~rocm~tests+umpire build_system=cmake build_type=Release generator=make arch=linux-ubuntu24.04-skylake
    -   mqmexdv      ^umpire@2024.07.0%gcc@13.3.0~asan~backtrace+c~cuda~dev_benchmarks~device_alloc~deviceconst~examples+fmt_header_only~fortran~ipc_shmem~ipo~mpi~numa~omptarget+openmp~rocm~sanitizer_tests~shared~sqlite_experimental~tools+werror build_system=cmake build_type=Release generator=make tests=none arch=linux-ubuntu24.04-skylake
    -   vjmiu7m           ^fmt@11.0.2%gcc@13.3.0~ipo+pic~shared build_system=cmake build_type=Release cxxstd=11 generator=make arch=linux-ubuntu24.04-skylake

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
    you should rename or move the file so they are not lost. For the rest of instruction, we will assume you renamed the file to
    ``ubuntu24.yaml``.

.. _ubuntu_tpl_build-label:

-----------------------------------
Build Serac's Third-party Libraries
-----------------------------------

It is now time to build Serac's Third-party Libraries (TPLs). Run the command with the compiler
that you want to develop with:

.. code-block:: bash

    # clang
    scripts/uberenv/uberenv.py --prefix=<path/outside/repository> --spack-env-file=ubuntu24.yaml --spec="%clang@=14.0.6"
    # gcc
    scripts/uberenv/uberenv.py --prefix=<path/outside/repository> --spack-env-file=ubuntu24.yaml --spec="%gcc@=13.3.0"

If successful, you will see two things. The first is what we call a host-config. It is all the CMake
inputs you need to build Serac. This file will be a new CMake file in the current directory with your machine
name, system type, and compiler, for example ``mycomputerlinux-ubuntu24.04-skylake-clang@14.0.6.cmake``.
The second will be output from Spack that ends in this:

.. code-block:: bash

    ==> serac: Executing phase: 'initconfig'
    ==> Updating view at /my/prefix/spack_env/.spack-env/view

--------------
Building Serac
--------------

Finally, with the TPL's built and the host-config file, you can build Serac with the following
command:

.. code-block:: bash

    ./config-build.py -hc <host-config file>
    cd <created build directory>
    make -j
    make -j8 test

For more detail instructions on how to build Serac, see :ref:`quickstart guide <build-label>`.



