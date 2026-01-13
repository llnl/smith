.. ## Copyright (c) Lawrence Livermore National Security, LLC and
.. ## other Smith Project Developers. See the top-level COPYRIGHT file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)

===========
Build Guide
===========

-------------------------------
Third-party Library (TPL) Build
-------------------------------

Spack and Uberenv
-----------------

Smith uses `Spack <https://github.com/spack/spack>`_ to build it's TPLs.
This has been encapsulated using `Uberenv <https://github.com/LLNL/uberenv>`_.
Uberenv helps by doing the following:

* Pulls a blessed version of Spack and Spack Packages locally
* If you are on a known operating system (like TOSS4), Uberenv will automatically
  use our blessed Spack Environment files to keep Spack from building the world
* Installs our Spack packages into the local Spack installation location
* Simplifies whole dependency build into one command

Uberenv will create a directory containing a Spack instance with the required Smith
TPLs installed.

.. note::
   This directory **must not** be within the Smith repo - the example below
   controls this with the ``--prefix`` command line argument which is required.

It also generates a host-config file (``<config_dependent_name>.cmake``)
at the root of Smith repository. This host-config defines all the required information for building
Smith.

Basic System Setup
------------------

We recommend installing some basic system-level development packages to minimize the
amount of packages that Spack will build.

The following pages provide basic guidance on the following platforms:

* :ref:`Livermore Computing (LC) <setup_tpl_lc-label>`
* :ref:`macOS <setup_tpl_mac-label>`
* :ref:`Ubuntu 24 <setup_tpl_ubuntu-label>`

.. note::

   Smith uses the LLVM plugin `Enzyme <https://github.com/EnzymeAD/Enzyme>`_ to perform
   automatic differentiation. To enable this functionality, you have to compile with an
   LLVM-based compiler. We recommend ``clang``.

.. toctree::
   :hidden:
   :maxdepth: 2

   setup_tpl_lc
   setup_tpl_mac
   setup_tpl_ubuntu
   building_tpls
   build


