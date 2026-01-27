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

Some helpful uberenv options include:

* ``--spec="+profiling build_type=Debug %clang_19"`` (Spack spec, ``smith@develop`` automatically prepended)
* ``--spack-env-file=<Path to Spack environment file>`` (use specific Spack environment configuration file)
* ``--prefix=<Path>`` (required, build and install the dependencies in a particular location) - this *must be outside* of your local Smith repository

The rest of Uberenv's command line options can be seen `here <https://uberenv.readthedocs.io/en/latest/#build-configuration>`_.

Basic Spack variants:

+-------------+---------+---------------------------------------------------------------+
| Variant     | Default | Description                                                   |
+=============+=========+===============================================================+
| shared      | False   | Enable build of shared libraries                              |
+-------------+---------+---------------------------------------------------------------+
| asan        | False   | Enable Address Sanitizer flags                                |
+-------------+---------+---------------------------------------------------------------+
| openmp      | True    | Enable OpenMP support                                         |
+-------------+---------+---------------------------------------------------------------+
| devtools    | False   | Build development tools (such as Sphinx, CppCheck,            |
|             |         | ClangFormat, etc...)                                          |
+-------------+---------+---------------------------------------------------------------+

.. note::
   If you are building on LC, using our provided Spack Environments, and do not have access to the ``smithdev`` linux group,
   you cannot use our prebuilt Developer Tools referenced in the Spack Environment files. You will need to turn off the
   devtool variant by adding ``~devtools`` to your Spack spec via the Spack or uberenv command line.


TPL Spack variants:

+-------------+---------+---------------------------------------------------------------+
| Variant     | Default | Description                                                   |
+=============+=========+===============================================================+
| adiak       | False   | Build with adiak                                              |
+-------------+---------+---------------------------------------------------------------+
| caliper     | False   | Build with caliper                                            |
+-------------+---------+---------------------------------------------------------------+
| enzyme      | False   | Enable Enzyme Automatic Differentiation Framework             |
+-------------+---------+---------------------------------------------------------------+
| petsc       | True    | Enable PETSc support                                          |
+-------------+---------+---------------------------------------------------------------+
| raja        | True    | Build with portable kernel execution support                  |
+-------------+---------+---------------------------------------------------------------+
| slepc       | True    | Enable SLEPc integration                                      |
+-------------+---------+---------------------------------------------------------------+
| strumpack   | True    | Build MFEM TPL with Strumpack, a direct linear solver library |
+-------------+---------+---------------------------------------------------------------+
| sundials    | True    | Build MFEM TPL with SUNDIALS nonlinear/ODE solver support     |
+-------------+---------+---------------------------------------------------------------+
| tribol      | True    | Build Tribol, an interface physics library                    |
+-------------+---------+---------------------------------------------------------------+
| umpire      | True    | Build with portable memory access support                     |
+-------------+---------+---------------------------------------------------------------+


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

