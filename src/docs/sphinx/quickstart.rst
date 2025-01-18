.. ## Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
.. ## other Serac Project Developers. See the top-level COPYRIGHT file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)

.. _quickstart-label:

======================
Quickstart Guide
======================

Getting Serac
-------------

Serac is hosted on `GitHub <https://github.com/LLNL/serac>`_. Serac uses git submodules, so the project must be cloned
recursively. Use either of the following commands to pull Serac's repository:

.. code-block:: bash

   # Using SSH keys setup with GitHub
   $ git clone --recursive git@github.com:LLNL/serac.git

   # Using HTTPS which works for everyone but is slightly slower and will require username/password
   # for some commands
   $ git clone --recursive https://github.com/LLNL/serac.git

Overview of the Serac build process
------------------------------------

The Serac build process has been broken into three phases with various related options:

1. (Optional) Build the developer tools
2. Build the third party libraries
3. Build the serac source code

The developer tools are only required if you wish to contribute to the Serac source code. The first two steps involve building all of the
third party libraries that are required by Serac. Two options exist for this process: using the `Spack HPC package manager <https://spack.io/>`_
via the `uberenv wrapper script <https://github.com/LLNL/uberenv>`_ or building the required dependencies on your own. We recommend the first
option as building HPC libraries by hand can be a tedious process. Once the third party libraries are built, Serac can be built using the
cmake-based `BLT HPC build system <https://github.com/LLNL/blt>`_.

.. note::
  If you get the following error ``ERROR: pip version 19.0.3 is too old to install clingo``, run the
  following command to upgrade your pip: ``python3 -m pip install --user --upgrade pip``.  This error
  will not necessarily be the last error on the screen.


.. _devtools-label:

Building Serac's Developer Tools
--------------------------------

.. note::
  This can be skipped if you are not doing Serac development or if you are on an LC machine.
  They are installed in a group space defined in ``host-config/<machine name>-<SYS_TYPE>-<compiler>.cmake``

Serac developers utilize some industry standard development tools in their everyday work.  We build
these with Spack and have them installed in a public space on commonly used LC machines. These are
defined in the host-configs in our repository.

If you wish to build them yourself (which takes a long time), use one of the following commands:

For LC machines:

.. code-block:: bash

   $ python3 scripts/llnl/build_devtools.py --directory=<devtool/build/path>

For other machines:

.. code-block:: bash

   $ python3 scripts/uberenv/uberenv.py --project-json=scripts/spack/devtools.json --spack-env-file=<scripts/spack/configs/platform/spack.yaml> --prefix=<devtool/build/path>

For example on **Ubuntu 20.04**:

.. code-block:: bash

   python3 scripts/uberenv/uberenv.py --project-json=scripts/spack/devtools.json --spack-env-file=scripts/spack/configs/linux_ubuntu_20/spack.yaml --prefix=../path/to/install

Unlike Serac's library dependencies, our developer tools can be built with any compiler because
they are not linked into the serac executable.  We recommend GCC 10 because we have tested that they all
build with that compiler.

Building Serac's Dependencies via Spack/uberenv
-----------------------------------------------

.. note::
  This is optional if you are on an LC machine as we have previously built the dependencies. You
  can see these machines and configurations in the ``host-configs`` repository directory.

Serac only directly requires `MFEM <https://mfem.org/>`_, `Axom <https://github.com/LLNL/axom>`_,
and `Conduit <https://github.com/LLNL/conduit>`_.  Through MFEM, Serac also depends on a number of
other third party libraries (Hypre, METIS, NetCDF, ParMETIS, SuperLU, and zlib).

The easiest path to build Serac's dependencies is to use `Spack <https://github.com/spack/spack>`_.
This has been encapsulated using `Uberenv <https://github.com/LLNL/uberenv>`_. Uberenv helps by
doing the following:

* Pulls a blessed version of Spack locally
* If you are on a known operating system (like TOSS4), we have defined Spack configuration files
  to keep Spack from building the world
* Installs our Spack packages into the local Spack
* Simplifies whole dependency build into one command

Uberenv will create a directory containing a Spack instance with the required Serac
dependencies installed.

.. note::
   This directory **must not** be within the Serac repo - the example below
   uses a directory called ``serac_libs`` at the same level as the Serac repository. The
   ``--prefix`` input argument is required.

It also generates a host-config file (``<config_dependent_name>.cmake``)
at the root of Serac repository. This host-config defines all the required information for building
Serac.

.. code-block:: bash

   $ python3 scripts/uberenv/uberenv.py --prefix=../serac_libs

.. note::
  On LC machines, it is good practice to do the build step in parallel on a compute node.
  Here is an example command: ``salloc -ppdebug -N1 python3 scripts/uberenv/uberenv.py``

Unless otherwise specified Spack will default to a compiler.  This is generally not a good idea when
developing large codes. To specify which compiler to use add the compiler specification to the ``--spec`` Uberenv
command line option. On TOSS4, we recommend and have tested ``--spec=%clang@14.0.6``.  More compiler specs
can be found in the Spack compiler files in our repository:
``scripts/spack/configs/<platform>/spack.yaml``.

We currently regularly test the following Spack configuration files:

* Linux Ubuntu 22.04 (via Github Actions)
* TOSS4 (On Ruby at LC)
* BlueOS (On Lassen at LC)

To install Serac on a new platform, it is a good idea to start with a known Spack environments file, or ``spack.yaml`` file,
(located in the Serac repo at ``scripts/spack/configs/<platform>``). The ``spack.yaml`` file
describes the compilers and associated flags required for the platform as well as the low-level libraries
on the system to prevent Spack from building the world. Documentation on these configuration files is located
in the `Spack docs <https://spack.readthedocs.io/en/latest/configuration.html>`_.

.. note::
   If you do not have a ``spack.yaml`` already, you can leave off that command line option from ``uberenv`` and
   Spack will generate a new one for you. Uberenv will copy it where you ran your uberenv command for future use.
.. note::
   A newer version of cmake (>=3.20) and llvm (>=14) may be required.


Some helpful uberenv options include :

* ``--spec=" build_type=Debug"`` (build the MFEM and Hypre libraries with debug symbols)
* ``--spec=+profiling`` (build the Adiak and Caliper libraries)
* ``--spec=+devtools`` (also build the devtools with one command)
* ``--spec=%clang@14.0.6`` (build with a specific compiler as defined in the ``spack.yaml`` file)
* ``--spack-env-file=<Path to Spack environment file>`` (use specific Spack environment configuration file)
* ``--prefix=<Path>`` (required, build and install the dependencies in a particular location) - this *must be outside* of your local Serac repository

The modifiers to the Spack specification ``spec`` can be chained together, e.g. ``--spec='%clang@14.0.6+devtools build_type=Debug'``.

If you already have a Spack instance from another project that you would like to reuse,
you can do so by changing the uberenv command as follows:

.. code-block:: bash

   $ python3 scripts/uberenv/uberenv.py --upstream=</path/to/my/spack>/opt/spack

Building Serac's Dependencies by Hand
-------------------------------------

To build Serac's dependencies by hand, use of a ``host-config`` CMake configuration file is
strongly encouraged. A good place to start is by copying an existing host config in the
``host-config`` directory and modifying it according to your system setup.

.. _build-label:

Using a Docker Image with Preinstalled Dependencies
---------------------------------------------------

As an alternative, you can build Serac using preinstalled dependencies inside a Docker
container. Instructions for this process are located :ref:`here <docker-label>`.

Building Serac
--------------

Serac uses a CMake build system that wraps its configure step with a script
called ``config-build.py``.  This script creates a build directory and
runs the necessary CMake command for you. You just need to point the script
at the generated or a provided host-config. This can be accomplished with
one of the following commands:

.. code-block:: bash

   # If you built Serac's dependencies yourself either via Spack or by hand
   $ python3 ./config-build.py -hc <config_dependent_name>.cmake

   # If you are on an LC machine and want to use our public pre-built dependencies
   $ python3 ./config-build.py -hc host-configs/<machine name>-<SYS_TYPE>-<compiler>.cmake

   # If you'd like to configure specific build options, e.g., a release build
   $ python3 ./config-build.py -hc /path/to/host-config.cmake -DCMAKE_BUILD_TYPE=Release <more CMake build options...>

If you built the dependencies using Spack/uberenv, the host-config file is output at the
project root. To use the pre-built dependencies on LC, you must be in the appropriate
LC group. Contact `Brandon Talamini <talamini1@llnl.gov>`_ for access.

Some build options frequently used by Serac include:

* ``CMAKE_BUILD_TYPE``: Specifies the build type, see the `CMake docs <https://cmake.org/cmake/help/latest/variable/CMAKE_BUILD_TYPE.html>`_
* ``ENABLE_BENCHMARKS``: Enables Google Benchmark performance tests, defaults to ``OFF``
* ``ENABLE_WARNINGS_AS_ERRORS``: Turns compiler warnings into errors, defaults to ``ON``
* ``ENABLE_ASAN``: Enables the Address Sanitizer for memory safety inspections, defaults to ``OFF``
* ``SERAC_ENABLE_TESTS``: Enables Serac unit tests, defaults to ``ON``
* ``SERAC_ENABLE_CODEVELOP``: Enables local development build of MFEM/Axom, see :ref:`codevelop-label`, defaults to ``OFF``
* ``SERAC_USE_VDIM_ORDERING``: Sets the vector ordering to be ``byVDIM``, which is significantly faster for algebraic multigrid, defaults to ``ON``.

Once the build has been configured, Serac can be built with the following commands:

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

Preparing Windows WSL/Ubuntu for Serac installation
---------------------------------------------------------

For faster installation of the Serac dependencies via Spack on Windows WSL/Ubuntu systems,
install cmake, MPICH, openblas, OpenGL, and the various developer tools using the following commands:

**Ubuntu 20.04**

.. code-block:: bash

   $ sudo apt-get update
   $ sudo apt-get upgrade
   $ sudo apt-get install cmake libopenblas-dev libopenblas-base mpich mesa-common-dev libglu1-mesa-dev freeglut3-dev cppcheck doxygen libreadline-dev python3-sphinx python3-pip clang-format-10 m4 elfutils
   $ sudo ln -s /usr/lib/x86_64-linux-gnu/* /usr/lib

**Ubuntu 18.04**

.. code-block:: bash

   $ sudo apt-get update
   $ sudo apt-get upgrade
   $ sudo apt-get install g++-8 gcc-8
   $ sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 800 --slave /usr/bin/g++ g++ /usr/bin/g++-8
   $ sudo apt-get install cmake libopenblas-dev libopenblas-base mpich mesa-common-dev libglu1-mesa-dev freeglut3-dev cppcheck doxygen libreadline-dev python3-distutils python3-pip
   $ sudo ln -s /usr/lib/x86_64-linux-gnu/* /usr/lib

Note that the last line is required since Spack expects the system libraries to exist in a directory
named ``lib``. During the third party library build phase, the appropriate Spack config directory
must be specified using either:

**Ubuntu 20.04**

``python3 scripts/uberenv/uberenv.py --spack-env-file=scripts/spack/configs/linux_ubuntu_20/spack.yaml --prefix=../path/to/install``

**Ubuntu 18.04**

``python3 scripts/uberenv/uberenv.py --spack-env-file=scripts/spack/configs/linux_ubuntu_18/spack.yaml --prefix=../path/to/install``

Building Serac Dependencies on MacOS
------------------------------------

.. warning::
   These instructions are in development, but have been tested for M2 MacBooks.

Installing base dependencies using Homebrew
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Homebrew is recommended to install base dependencies due to it's stability. Relying on pure Spack historically leads to more failed builds.

To start, install the following packages using Homebrew.

.. code-block:: bash

   $ brew install autoconf automake bzip2 clingo cmake diffutils fmt gcc gettext gnu-sed graphviz hwloc lapack libx11 llvm@14 m4 make ninja open-mpi openblas pkg-config readline zlib

If you plan to install the developer tools, you should also run:

.. code-block:: bash

   $ brew install cppcheck doxygen llvm@14
   $ ln -s /opt/homebrew/opt/llvm@14/bin/clang-format /opt/homebrew/bin/clang-format

If you have installed Homebrew using the default installation prefix, most packages will be accessible through the prefix ``/opt/homebrew``.
Note for Intel-based Macs, the installation prefix is ``/usr/local``. If you set a custom prefix or aren't sure what the prefix is, run ``brew --prefix``.
For the rest of this section, we will assume the prefix is ``/opt/homebrew``.
Some packages are not linked into this prefix to prevent conflicts with MacOS-provided versions.
These will only be accessible via the prefix ``/opt/homebrew/opt/[package-name]``.
Homebrew will warn about such packages after installing them.

In order for the correct compilers to be used for the installation, you should also add the bin directory for LLVM clang to your path in your ``.bash_profile``, ``.bashrc``, or ``.zshrc``, etc.
This is also useful for a few additional packages:

.. code-block:: bash

   $ export PATH="/opt/homebrew/opt/llvm@14/bin:/opt/homebrew/opt/m4/bin:/opt/homebrew/opt/gnu-sed/libexec/gnubin:$PATH"

Configuring Spack
^^^^^^^^^^^^^^^^^

In order to build Serac, we must define a ``spack.yaml`` file which tells Spack what packages we have installed.
You will likely need to update the versions of packages in the provided example script ``scripts/spack/configs/macos_sonoma_aarch64/spack.yaml`` to match the versions installed by Homebrew.
The versions for all installed packages can be listed via:

.. code-block:: bash

   $ brew list --versions

Note that the version format output by the above command is not the same as that expected by Spack, so be sure to add an ``@`` symbol between the package name and version string.

If you are not using an M2 or M3 Mac, you will need to change the ``target`` for the compiler to ``x86_64`` or ``aarch64`` for Intel and M1-based Macs, respectively.
Similarly, you need to set the ``operating_system`` to the proper value if you are not using ``sonoma`` (MacOS 14.X).

Installing Project Developer Tools with Pip
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This step is only required if you wish to use Serac's developer tools. In order to use Python devtools, you will need to create a Python venv. This is much more reliable than having Spack install 20+ Python packages.
In this example, we are using the builtin Python in ``/usr/bin``, but it is possible to use a version installed from Homebrew or elsewhere.

Next, you will need to install wheel, Sphinx, and `ATS <https://github.com/LLNL/ATS/tree/7.0.105>`_:

.. code-block:: bash

   python3 -m venv --system-site-packages venv
   source venv/bin/activate
   pip install wheel
   pip install sphinx
   git clone git@github.com:LLNL/ATS.git --branch 7.0.105
   pip install ATS/

Keep track of the Sphinx version while installing, since you'll need it for the next step.

Adding Developer Tools to Spack Environment File
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Again, skip this step if not using devtools. After setting up a Python venv, you should add the following under ``packages`` in the ``spack.yaml`` files. Versions may vary.

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
    llvm:
      version: [14.0.6]
      buildable: false
      externals:
      - spec: llvm+clang@14.0.6
        prefix: /opt/homebrew/opt/llvm@14
    py-sphinx:
      buildable: false
      externals:
      - spec: py-sphinx@7.4.7
        prefix: /path/to/venv
    py-ats:
      buildable: false
      externals:
      - spec: py-ats@7.0.105
        prefix: /path/to/venv

Building dependencies
^^^^^^^^^^^^^^^^^^^^^

Uberenv is a Spack wrapper simplifing the TPL build process. The invocation of ``uberenv.py`` is slightly modified from the standard instructions above in order to force the use of the Homebrew-installed MPI and compilers:

.. code-block:: bash

   $ ./scripts/uberenv/uberenv.py --spack-env-file=/path/to/spack.yaml --prefix=/path/to/install --spec="%clang@14 ^openmpi@5"

.. note::
   To build with devtools and profiling enabled, change the spec to ``"%clang@14+devtools+profiling ^openmpi@5"``
