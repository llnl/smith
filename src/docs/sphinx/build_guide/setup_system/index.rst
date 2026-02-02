.. ## Copyright (c) Lawrence Livermore National Security, LLC and
.. ## other Smith Project Developers. See the top-level COPYRIGHT file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)

.. _setup_system-label:

==================
Basic System Setup
==================

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

.. toctree::
   :hidden:
   :maxdepth: 1

   setup_lc
   setup_macos
   setup_ubuntu

