.. ## Copyright (c) Lawrence Livermore National Security, LLC and
.. ## other Smith Project Developers. See the top-level COPYRIGHT file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)

.. _quickstart-label:

======================
Quickstart Guide
======================

Getting Smith
-------------

Smith is hosted on `GitHub <https://github.com/LLNL/smith>`_. Smith uses git submodules, so the project must be cloned
recursively. Use either of the following commands to pull Smith's repository:

.. code-block:: bash

   # Using SSH keys setup with GitHub
   $ git clone --recursive git@github.com:LLNL/smith.git

   # Using HTTPS which works for everyone but is slightly slower and will require username/password
   # for some commands
   $ git clone --recursive https://github.com/LLNL/smith.git

Overview of the Smith build process
------------------------------------

The Smith build process has been broken into three phases with various related options:

1. (Optional) Build the developer tools
2. Build the third party libraries
3. Build the Smith source code

The developer tools are only required if you wish to contribute to the Smith source code. The first two steps involve building all of the
third party libraries that are required by Smith. Two options exist for this process: using the `Spack HPC package manager <https://spack.io/>`_
via the `uberenv wrapper script <https://github.com/LLNL/uberenv>`_ or building the required dependencies on your own. We recommend the first
option as building HPC libraries by hand can be a tedious process. Once the third party libraries are built, Smith can be built using the
cmake-based `BLT HPC build system <https://github.com/LLNL/blt>`_.

.. _devtools-label:

Building Smith's Developer Tools
--------------------------------

.. note::
  This can be skipped if you are not doing Smith development or if you are on an LC machine.
  They are installed in a group space defined in ``host-config/<machine name>-<SYS_TYPE>-<compiler>.cmake``

Smith developers utilize some industry standard development tools in their everyday work.  We build
these with Spack and have them installed in a public space on commonly used LC machines. These are
defined in the host-configs in our repository.

If you wish to build them yourself (which takes a long time), use one of the following commands:

For LC machines:

.. code-block:: bash

   $ python3 scripts/llnl/build_devtools.py --directory=<devtool/build/path>

For other machines:

.. code-block:: bash

   $ python3 scripts/uberenv/uberenv.py --project-json=scripts/spack/devtools.json --spack-env-file=<scripts/spack/configs/platform/spack.yaml> --prefix=<devtool/build/path>

For example on **Ubuntu 24.04**:

.. code-block:: bash

   python3 scripts/uberenv/uberenv.py --project-json=scripts/spack/devtools.json --spack-env-file=scripts/spack/configs/linux_ubuntu_24/spack.yaml --prefix=../path/to/install

Unlike Smith's library dependencies, our developer tools can be built with any compiler because
they are not linked into the smith executable.  We recommend Clang 19 because we have tested that they all
build with that compiler.

Building Smith's Dependencies via Spack/uberenv
-----------------------------------------------

For detailed instructions see :ref:`tpl_builds-label`.

.. note::
  This is optional if you are on an LC machine and are in the ``smithdev`` group as we have
  previously built the dependencies. You can see these machines and configurations in the 
  ``host-configs`` repository directory.

Using a Docker Image with Preinstalled Dependencies
---------------------------------------------------

As an alternative, you can build Smith using preinstalled dependencies inside a Docker
container. Instructions for this process are located :ref:`here <docker-label>`.
