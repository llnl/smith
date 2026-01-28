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

We provide a python script that encapsulates the CMake configuration step
but you can also use the CMake executable directly. Below are instructions
for both.


config-build.py
---------------

``config-build.py`` is a python script that is aimed at simplifying and hardening
running CMake.  It creates a build and install directory then runs the necessary
CMake command for you. You just need to point the script
at the generated or a provided host-config. This can be accomplished with
one of the following commands:

.. code-block:: bash

   # If you built Smith's dependencies yourself either via Spack or by hand
   $ ./config-build.py -hc <config_dependent_name>.cmake

   # If you are on an LC machine and want to use our public pre-built dependencies
   $ ./config-build.py -hc host-configs/<machine name>-<SYS_TYPE>-<compiler>.cmake

   # If you'd like to configure specific build options, e.g., a debug build
   $ ./config-build.py -hc /path/to/host-config.cmake -DCMAKE_BUILD_TYPE=Debug <more CMake build options...>

If you built the dependencies using Spack/uberenv, the host-config file is output at the
project root. To use the pre-built dependencies on LC, you must be in the appropriate
LC group. Contact `Brandon Talamini <talamini1@llnl.gov>`_ for access.

.. important::

   ``config-build.py`` automatically deletes the build and install directories.
   Do **not** store any files in these directories that you wish to keep.

   These directories should be treated as **temporary**, as their contents may
   be removed at any time during the build process.


``config-build.py`` command line options are:

.. list-table:: Command-line options
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


CMake
-----

Some build options frequently used by Smith include:

* ``CMAKE_BUILD_TYPE``: Specifies the build type, see the `CMake docs <https://cmake.org/cmake/help/latest/variable/CMAKE_BUILD_TYPE.html>`_
* ``ENABLE_BENCHMARKS``: Enables Google Benchmark performance tests, defaults to ``OFF``
* ``ENABLE_WARNINGS_AS_ERRORS``: Turns compiler warnings into errors, defaults to ``ON``
* ``ENABLE_ASAN``: Enables the Address Sanitizer for memory safety inspections, defaults to ``OFF``
* ``SMITH_ENABLE_TESTS``: Enables Smith unit tests, defaults to ``ON``
* ``SMITH_ENABLE_CODEVELOP``: Enables local development build of MFEM/Axom, see :ref:`codevelop-label`, defaults to ``OFF``
* ``SMITH_USE_VDIM_ORDERING``: Sets the vector ordering to be ``byVDIM``, which is significantly faster for algebraic multigrid, defaults to ``ON``.

Once the build has been configured, Smith can be built with the following commands:

.. code-block:: bash

   $ cd build-<system-and-toolchain>
   $ make -j16

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



