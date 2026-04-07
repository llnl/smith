
.. ##
.. ## Copyright (c) 2022-25, Lawrence Livermore National Security, LLC and
.. ## other RADIUSS Project Developers. See the top-level COPYRIGHT file for
.. ## details.
.. ##
.. ## SPDX-License-Identifier: (MIT)
.. ##

.. _CI Implementation:

#################
CI Implementation
#################

In this repository, we configured GitLab CI to generate CI pipelines that can
build any Spack environment we provide.

We rely on `Spack Pipeline feature <https://spack.readthedocs.io/en/latest/pipelines.html>`_
to generate the CI pipeline.

Each CI pipeline run corresponds to the build of *one* Spack environment. The
environment being built is configured by the variable ``MY_ENV_NAME`` in CI
context. This variable must match a directory in ``.gitlab/spack/envs`` and
defaults to the value "shared-ci": an environment meant to build several
RADIUSS projects using the shared Spack specs.

.. note:: The shared Spack specs can be found in each
   ``gitlab/radiuss-jobs/<machine>.yml`` file. The specs there have been copied
   to the ``shared-ci/spack.yml`` file. Maintaining coherency between both is
   important for the quality of our CI.

The intent of the default configuration is to test that changes in
radiuss-spack-configs does not affect our projects. However, the CI can be set
to build other environments, which will allow us to test larger environments on
a less frequent basis, and to track changes in Spack as well.

Incidentally, this CI setup will populate a buildcache with pre-built binaries
of our RADIUSS projects, which could prove useful to developers willing to
save build time for the selected specs.


=================
CI file structure
=================

Main CI pipeline
================

The root file for CI implementation with GitLab is ``.gitlab-ci.yml``. In this
file, we only define general variables and the list of stages. Then we include
the files that will actually describe our pipelines.

The additional CI files are found under the ``.gitlab`` directory. There, the
``variables.yml`` file sets variable for the different LC machines we run CI on,
namely ``corona``, ``dane``, ``lassen``, ``tioga`` and ``tuo``. The
pipelines for each machine is described in the corresponding ``<machine>.yml``
file.

``spack/envs/<MY_ENV_NAME>/pipeline.yml`` must be defined for each environment
and provide the Spack repo and commit to fetch (``MY_SPACK_REPO``,
``MY_SPACK_COMMIT``), and include the pipeline files for each machine to run
on. I.e., for each environment, we can chose a specific Spack version and the
list of machines.

Machine pipelines
=================

In each ``<machine>.yml`` file, we describe a pipeline that will allocate
resource for the CI on the machine, and then generate and trigger a CI pipeline
to build each required spec as a separate job.

Pipeline generation is handle by spack (see `Spack documentation`_ for more
about the CI feature) and is configured in ``.gitlab/spack/ci.yaml`` and
other files in that directory.

Scripts
=======

Scripts used by the main CI configuration are located in ``.gitlab/scripts``
while scripts used for Spack specific operations (typically occurring when
generating the sub-pipelines or while running them) are located in
``.gitlab/spack``.

.. note:: We ensure to separate the scripts files from the CI configuration.
   This helps with testing, keeping the CI files readable, and reproducing the
   CI process outside of CI context.

Spack Pipeline Feature
======================

To learn more about the Spack Pipeline feature used in this CI configuration,
it is a good idea to first read the dedicated `Spack documentation`_.



.. _Spack documentation: https://spack.readthedocs.io/en/latest/pipelines.html
