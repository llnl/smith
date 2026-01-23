.. ## Copyright (c) Lawrence Livermore National Security, LLC and
.. ## other Smith Project Developers. See the top-level COPYRIGHT file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)

.. _build_guide-label:

===========
Build Guide
===========

This guide provides instructions on how to build or install all dependencies of Smith, followed by 
how to build Smith from source. The process consists of the following high-level phases:

#. Setup the machine by installing the required system-level packages and generating a Spack environment file
#. Build Third-party Libraries (TPLs) using Spack and Uberenv to provide a consistent dependency stack
#. Build Smith itself using CMake once all dependencies are in place

.. note::

   Smith uses the LLVM plugin `Enzyme <https://github.com/EnzymeAD/Enzyme>`_ to perform
   automatic differentiation. To enable this functionality, you have to compile with an
   LLVM-based compiler. We recommend ``clang``.

.. toctree::
   :hidden:
   :titlesonly:
   :maxdepth: 2

   setup_system/index
   build_tpls
   build_smith

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


Phase 1: Basic System Setup
---------------------------

We recommend installing some basic system-level development packages to minimize the
amount of packages that Spack will build.

The following pages provide basic guidance on the following platforms and is where you should
start:

* :ref:`Livermore Computing (LC) <setup_lc-label>`
* :ref:`macOS <setup_macos-label>`
* :ref:`Ubuntu 24 <setup_ubuntu-label>`

At the end of each Setup guide, it has a link to the page that shows you how to build
the minimal set of TPLs for Smith; followed by a page on how to build
Smith from the generated host-config file via CMake.


Phase 2: Build Third-party Libraries
------------------------------------

For more information see :ref:`build_tpls-label`.


Phase 3: Build Smith with CMake
-------------------------------

For more information see :ref:`build_smith-label`.

