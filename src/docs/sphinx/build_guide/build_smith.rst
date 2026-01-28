.. ## Copyright (c) Lawrence Livermore National Security, LLC and
.. ## other Smith Project Developers. See the top-level COPYRIGHT file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)

.. _build_smith-label:

=========================
Building Smith with CMake
=========================

Smith uses a CMake build system as its build system. Due to our amount of
Third-party Libraries (TPLs) and configuration options, we recommend
utilizing a :ref:`host-config <host_config-label>` to encapsulate most
of the build information necessary to build Smith. 

If you built the dependencies using Spack/uberenv, the host-config file is output at the
project root. If you are on LC and a member of the ``smithdev`` group, you can use
our provided host-configs in the ``host-config`` directory that follow the pattern
``<machine_name>-<SYS_TYPE>-<compiler>.cmake``. Contact `Brandon Talamini <talamini1@llnl.gov>`_ for access.

We provide a python script that encapsulates the CMake configuration step
but you can also use the CMake executable directly. Below are instructions
for both.


Option 1: Configuring the build with ``config-build.py``
--------------------------------------------------------

``config-build.py`` is a python script that is aimed at simplifying and hardening
running CMake.  It creates a build and install directory then runs the necessary
CMake command for you. You just need to point the script
at the generated or a provided host-config. 

``config-build.py`` has some command line options it understand to simplify the
build, they are listed in the table below. Any extra options will be passed directly
to CMake. Here are some examples on how to run ``config-build.py``:

.. code-block:: bash

   # Just a host-config
   $ ./config-build.py -hc /path/to/host-config.cmake

   # host-config + debug build
   $ ./config-build.py -hc /path/to/host-config.cmake -bt Debug

   # host-config + CMake options
   $ ./config-build.py -hc /path/to/host-config.cmake -DENABLE_WARNINGS_AS_ERRORS=OFF


.. important::

   ``config-build.py`` automatically deletes the build and install directories.
   Do **not** store any files in these directories that you wish to keep. These
   directories should be treated as **temporary**, as their contents may
   be removed at any time during the build process.


.. list-table:: ``config-build.py`` command-line options
   :header-rows: 1
   :widths: 18 28 14 60

   * - Short Option
     - Long Option
     - Default
     - Description

   * - ``-bp``
     - ``--buildpath``
     - ``""``
     - Specify path for the build directory. If not specified, it will be created
       in the current directory.

   * - ``-ip``
     - ``--installpath``
     - ``""``
     - Specify path for the install directory. If not specified, it will be created
       in the current directory.

   * - ``-bt``
     - ``--buildtype``
     - ``Release``
     - Specify the CMake build type. Valid options are ``Release``, ``Debug``,
       ``RelWithDebInfo``, and ``MinSizeRel``.

   * - ``-e``
     - ``--eclipse``
     - ``False``
     - Create an Eclipse project file.

   * - ``-ecc``
     - ``--exportcompilercommands``
     - ``False``
     - Generate a compilation database (``compile_commands.json``) in the build
       directory. Can be used by Clang tools such as ``clang-modernize``.

   * - ``-hc``
     - ``--hostconfig``
     - ``""``
     - Select a specific host-config file to initialize CMake’s cache.

   * - *(none)*
     - ``--print-default-host-config``
     - ``False``
     - Print the default host configuration for this system and exit.

   * - *(none)*
     - ``--print-machine-name``
     - ``False``
     - Print the machine name for this system and exit.

   * - ``-n``
     - ``--ninja``
     - ``False``
     - Use the Ninja generator instead of Make to build the project.


Option 2: Configuring the build with CMake
------------------------------------------

Another option is to use CMake directly, this can also be useful if you configure VSCode
to build Smith.


.. list-table:: CMake configuration options
   :header-rows: 1
   :widths: 35 15 60

   * - Option
     - Default
     - Description

   * - ``CMAKE_BUILD_TYPE``
     - ``Release``
     - Specifies the build type, see the `CMake docs <https://cmake.org/cmake/help/latest/variable/CMAKE_BUILD_TYPE.html>`_

   * - ``ENABLE_WARNINGS_AS_ERRORS``
     - ``ON``
     - Turns compiler warnings into errors.

   * - ``ENABLE_ASAN``
     - ``OFF``
     - Enable AddressSanitizer for memory checking. Supported only with
       Clang or GCC; configuration will fail for other compilers.

   * - ``SMITH_ENABLE_CODEVELOP``
     - ``OFF``
     - Enable Smith’s *codevelop* build, including MFEM and Axom as CMake
       subdirectories.

   * - ``SMITH_ENABLE_CODE_CHECKS``
     - ``ON``
     - Enable Smith’s code checks.

   * - ``SMITH_ENABLE_TESTS``
     - ``ON``
     - Enable Smith test builds. This option is only effective when
       ``ENABLE_TESTS`` is ``ON``.

   * - ``SMITH_ENABLE_CUDA``
     - ``ON``
     - Enable Smith with CUDA support. This option is only effective when
       ``ENABLE_CUDA`` is ``ON``.

   * - ``SMITH_ENABLE_HIP``
     - ``ON``
     - Enable Smith with HIP support. This option is only effective when
       ``ENABLE_HIP`` is ``ON``.

   * - ``SMITH_ENABLE_OPENMP``
     - ``ON``
     - Enable Smith with OpenMP support. This option is only effective when
       ``ENABLE_OPENMP`` is ``ON``.

   * - ``SMITH_ENABLE_GRETL``
     - ``ON``
     - Enable Smith with Gretl support.

   * - ``SMITH_ENABLE_CONTINUATION``
     - ``ON``
     - Enable the Continuation Solver. This option is automatically forced
       to ``OFF`` when either ``SMITH_ENABLE_CUDA`` or ``SMITH_ENABLE_HIP``
       is enabled, as GPU builds are currently unsupported.

   * - ``SMITH_ENABLE_PROFILING``
     - ``OFF``
     - Enable profiling functionality. This option is automatically enabled
       when benchmarking is enabled unless explicitly set by the user.

   * - ``ENABLE_BENCHMARKS``
     - ``OFF``
     - Enables Google Benchmark performance tests.

   * - ``SMITH_ENABLE_BENCHMARKS``
     - ``ON``
     - Enable Smith benchmark executables. This option is only effective
       when ``ENABLE_BENCHMARKS`` is ``ON``. Benchmarking requires
       ``SMITH_ENABLE_PROFILING`` to be enabled; otherwise configuration
       will fail.

   * - ``SMITH_USE_VDIM_ORDERING``
     - ``ON``
     - Use ``mfem::Ordering::byVDIM`` for degree-of-freedom vectors, which
       is typically faster for algebraic multigrid. When disabled,
       ``byNODES`` ordering is used instead.


Build
-----

Once the build has been configured, Smith can be built with one of the following commands:

.. code-block:: bash

   # Makefile
   $ cd <build directory>
   $ make -j16

   # Ninja
   $ cd <build directory>
   $ ninja -j16

   # CMake
   $ cmake --build build -- -j16


.. note::
  On LC machines, it is good practice to do the build step in parallel on a compute node.
  Here is an example command: ``srun -ppdebug -N1 --exclusive make -j16``

We provide the following useful build targets that can be run from the build directory:

* ``test``: Runs our unit tests
* ``docs``: Builds our documentation to the following locations:

   * Sphinx: ``build-*/src/docs/html/index.html``
   * Doxygen: ``/build-*/src/docs/html/doxygen/html/index.html``

* ``style``: Runs styling over source code and replaces files in place
* ``check``: Runs a set of code checks over source code (CppCheck and clang-format)
* ``install``: Installs Smith into the previously given ``CMAKE_INSTALL_PREFIX`` directory



