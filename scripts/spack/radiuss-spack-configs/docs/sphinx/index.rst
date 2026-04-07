.. ## Copyright (c) 2019-2025, Lawrence Livermore National Security, LLC and
.. ## other RADIUSS Project Developers. See the top-level COPYRIGHT file for details.
.. ##
.. ## SPDX-License-Identifier: (MIT)
.. ##

#####################
RADIUSS Spack Configs
#####################

The RADIUSS Spack Configs project provides a *coherent set of Spack
configuration files* targeting Livermore Computing systems. The goal is to
enable any Spack user to quickly reproduce the exact same configuration RADIUSS
projects are vetted with.

RADIUSS Spack Configs can be used to provide the `RADIUSS Shared CI`_
infrastructure with a build and test process and a proposed list of Spack
specs. In practice, sharing that list of Specs helps Umpire, RAJA, CHAI,
Caliper and other projects to ensure, in their CI, that they keep running
correctly with toolchains of interest.

.. note:: LLNL's RADIUSS project (Rapid Application Development via an
   Institutional Universal Software Stack) aims to broaden usage across LLNL
   and the open source community of a set of libraries and tools used for HPC
   scientific application development.


========
Overview
========

RADIUSS Spack Configs User Documentation
========================================

The steps necessary to leverage RADIUSS Spack Configs are documented in the
:ref:`RADIUSS Spack Configs User Guide<sphinx/user_guide/index>`.

.. # RADIUSS Spack Configs Developer Documentation
.. # =============================================
.. #
.. # TODO: In the  :doc:`RADIUSS Spack Configs Developer Guide <sphinx/dev_guide/index>`,
.. # we discuss the layout of the RADIUSS Spack Configs repository and how to
.. # contribute to it.


Shared Spack specs for CI
=========================

This repository also hosts the shared specs we want to be sure to build within each of
our projects CI.


CI implementation for RADIUSS Spack Configs
===========================================

RADIUSS Spack Configs has its own CI workflow that leverages
Spack CI feature to build a spack environment containing our RADIUSS projects for all
the shared Specs.


Custom CI pipelines
===================

The CI implementation allows anyone to build his own environment with desired version
of Spack. We use this feature to expand testing capability and track changes in Spack
that could affect our builds.


=========================
Background and Motivation
=========================

Many RADIUSS projects are packaged with `Spack`_. Packaging the software
streamlines its installation on a variety of systems, like the ones present at
the LLNL. The primary goal of RADIUSS Spack Configs is to provide a
centralized location for the Spack configuration files needed to build RADIUSS
software with their dependencies or Livermore Computing (LC) systems.

For the packaging to be effective, we need to regularly test that our software
stack builds on our systems, both for generic builds, but also more targeted
ones. RADIUSS Spack Configs associated to `Uberenv`_ provides a build
infrastructure usable in CI and helps with integrating packaging into the
developers workflow. Spack then becomes a tool that helps developers easily
manage their dependencies, while packaging is maintained earlier in the
development process.


.. toctree::
   :hidden:
   :caption: User Documentation

   user_guide/index
   ci_implementation/index

.. # .. toctree::
.. #    :hidden:
.. #    :caption: Developer Documentation
.. #
.. #    sphinx/dev_guide/index

.. _RADIUSS Spack Configs: https://radiuss-spack-configs.readthedocs.io/en/latest/index.html
.. _RADIUSS Shared CI: https://radiuss-shared-ci.readthedocs.io/en/latest/index.html
.. _Uberenv: <https://github.com/LLNL/uberenv>
.. _Spack: <https://github.com/Spack/Spack>
