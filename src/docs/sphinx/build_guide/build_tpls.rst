.. ## Copyright (c) Lawrence Livermore National Security, LLC and
.. ## other Smith Project Developers. See the top-level COPYRIGHT file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)

.. _building_tpls-label:

========================================================
Building Smith's Third-party Libraries
========================================================

It is now time to build Smith's Third-party Libraries (TPLs). Run the command with the compiler
that you want to develop with:

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
for more detail instructions on how to build Smith, see :ref:`build-label`.

