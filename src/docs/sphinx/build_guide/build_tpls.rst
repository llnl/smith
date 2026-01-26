.. ## Copyright (c) Lawrence Livermore National Security, LLC and
.. ## other Smith Project Developers. See the top-level COPYRIGHT file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)

.. _build_tpls-label:

======================================
Building Smith's Third-party Libraries
======================================

It is now time to build Smith's Third-party Libraries (TPLs) and optionally our Developer Tools.

.. _devtools-label:

Building Developer Tools
------------------------

.. note::
  This section can be skipped if you are not developing Smith and contributing back to our repository. It
  can also be skipped if you are apart of the ``smithdev`` LC Linux group, as it will be automatically
  added by our Spack Environment.

Smith developers utilize some industry standard development tools in their everyday work. These tools
can take a very long time to build and it is recommended to build them separately from the TPLs then
include them in the followup TPL build. Unlike Smith's library dependencies, our developer tools can be
built with any compiler because they are not linked into the smith executable.  We recommend Clang 19
because we have tested that they all build with that compiler. If you wish to build them yourself
(which takes a long time), use one of the following commands:

For LC machines:

.. code-block:: bash

    # These commands are equivalent, pick one
    $ scripts/llnl/build_devtools.py --directory=<devtool/build/path>
    $ scripts/uberenv/uberenv.py --project-json=scripts/spack/devtools.json --spack-env-file=scripts/spack/devtools_configs/$SYS_TYPE/spack.yaml --prefix=../path/to/install


For other machines utilize the Spack Environment created in the previous :ref:`setup_system-label` step:

.. code-block:: bash

   $ scripts/uberenv/uberenv.py --project-json=scripts/spack/devtools.json --spack-env-file=<created/spack.yaml> --prefix=../path/to/install


For example on **Ubuntu 24.04**:

.. code-block:: bash

   $ scripts/uberenv/uberenv.py --project-json=scripts/spack/devtools.json --spack-env-file=scripts/spack/configs/linux_ubuntu_24/spack.yaml --prefix=../path/to/install


After the Developer Tools have been successfully installed, you have two options to use them in the following step to build
your TPLs.

1. Alter your Spack Environment by adding the following lines with the correct paths and versions:

   .. code-block:: yaml

      cppcheck:
        version: [2.9]
        buildable: false
        externals:
        - spec: cppcheck@2.9
          prefix: /path/to/devtools_install/cppcheck-2.9


1. Add a Spack upstream to the ``uberenv`` commands below with this command line option ``--upstream=../path/to/devtools_install``.


Building TPLs
-------------

Run the command with the compiler that you want to develop with:

.. code-block:: bash

    scripts/uberenv/uberenv.py --prefix=<path/outside/repository> --spack-env-file=<path/to/spack.yaml> --spec="%clang_19"

Some helpful uberenv options include :

* ``--spec=" build_type=Debug"`` (build core TPLs, such as MFEM and Hypre, with debug symbols)
* ``--spec=+profiling`` (build the Adiak and Caliper libraries)
* ``--spec=+devtools`` (also build the devtools with one command)
* ``--spec=%clang_19`` (build with a specific compiler as defined in the ``spack.yaml`` file)
* ``--spack-env-file=<Path to Spack environment file>`` (use specific Spack environment configuration file)
* ``--prefix=<Path>`` (required, build and install the dependencies in a particular location) - this *must be outside* of your local Smith repository

The modifiers to the Spack specification ``spec`` can be chained together, e.g. ``--spec='+devtools build_type=Debug %clang_19'``.

If successful, you will see two things. The first is what we call a host-config. It is all the CMake
inputs you need to build Smith. This file will be a new CMake file in the current directory with your machine
name, system type, and compiler, for example ``mycomputerlinux-ubuntu24.04-skylake-clang@19.1.1.cmake``.
The second will be output from Spack that ends in this:

.. code-block:: bash

    ==> smith: Executing phase: 'initconfig'
    ==> Updating view at /my/prefix/spack_env/.spack-env/view

--------------
Building Smith
--------------

Finally, with the TPL's built and the host-config file, you can build Smith
for more detail instructions on how to build Smith, see :ref:`build_smith-label`.

