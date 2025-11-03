.. ## Copyright (c) Lawrence Livermore National Security, LLC and
.. ## other Smith Project Developers. See the top-level COPYRIGHT file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)

=======================
Building a Docker Image
=======================

The following instructions apply to the creation of a new compiler image.


Create New Docker File
----------------------

.. note:: If a Dockerfile for the desired compiler already exists, you can skip this section and go to `update-docker-image-label`_ .

#. Start by cloning the ``smith`` repository and creating a branch off ``develop``.  
#. Ensure that an image exists on Dockerhub for the desired compiler.
   If no corresponding compiler image exists, it should be 
   created before proceeding. Here is the `RADIUSS Docker repository images <https://github.com/LLNL/radiuss-docker/pkgs/container/radiuss>`_.
#. Go to the ``scripts/docker`` and copy one of the various dockerfiles (e.g. dockerfile_clang-19) - renaming the file and modifying the ``FROM`` image and ``ENV spec`` as needed.
#. Edit ``./github/workflows/docker_build_tpls.yml`` to add new job for the new compiler image.  This can be copy-pasted 
   from one of the existing jobs - the only things that must be changed are the job name and ``TAG``, which should match the
   name of the compiler/generated ``Dockerfile``.  For example, a build for ``dockerfile_clang-14`` must set ``TAG``
   to ``clang-14``.  For clarity, the ``name`` field for the job should also be updated.
#. Commit and push the added YAML file and new Dockerfile.


.. _update-docker-image-label:

Update/Add Docker Image
-----------------------

#. Go to the Actions tab on GitHub, select the "Docker TPL Build" action, and run the workflow on the branch to
   which the above changes were pushed.
#. Once the "Docker TPL Build" action completes, it will produce artifacts for each of the generated host-configs.
   Download these artifacts and rename them to just the compiler spec.  For example, ``buildkitsandbox-linux-clang@14.0.0.cmake``
   to ``clang@14.0.0.cmake`` and commit them to your branch under ``host-configs/docker``.  You will also have to update
   ``.github/workflows/build-and-test.yml`` if you added or change the existing compiler specs. These are all in variables called ``host_config``.
#. Copy the new docker image names from each job under the ``Get dockerhub repo name`` step.  For example,
   ``seracllnl/tpls:clang-14_06-02-22_04h-11m``. This will replace the previous image name at the top of ``.github/workflows/build-and-test.yml``
   under the ``matrix`` section or add a new entry if you are adding a new docker image.
#. To include the new image in CI jobs, add/update the ``matrix`` entry to ``.github/workflows/build-and-test.yml``, modifying its 
   attributes with the appropriate new image name (which is timestamped) and new host-config file.
