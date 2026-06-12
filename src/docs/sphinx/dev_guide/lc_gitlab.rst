.. ## Copyright (c) Lawrence Livermore National Security, LLC and
.. ## other Smith Project Developers. See the top-level COPYRIGHT file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)

.. _lc_gitlab-label:

=====================
LC GitLab Setup
=====================

Smith has a `Gitlab mirror repository <https://lc.llnl.gov/gitlab/smith/smith>`_.
This repository is for running CI pipelines including PR checks.


Moving Jacamar Builds Out of ``$HOME``
======================================

LC GitLab jobs run through Jacamar and create working directories under
``${HOME}/.jacamar-ci`` by default. Smith builds can be large, so it is useful
to move the real storage for this directory to a file system with more space and
leave ``${HOME}/.jacamar-ci`` as a symlink.

Choose a directory on a file system, we recommend either ```/usr/workspace``
or ``/p/vast``, appropriate for build artifacts, then run:

.. code-block:: bash

   LARGE_FILE_SYSTEM_PARENT=/usr/workspace/${USER}

   if [[ -e "${HOME}/.jacamar-ci" && ! -L "${HOME}/.jacamar-ci" ]]; then
       mv "${HOME}/.jacamar-ci" "${LARGE_FILE_SYSTEM_PARENT}/.jacamar-ci"
   else
       mkdir -p "${LARGE_FILE_SYSTEM_PARENT}/.jacamar-ci"
   fi

   ln -sfn "${LARGE_FILE_SYSTEM_PARENT}/.jacamar-ci" "${HOME}/.jacamar-ci"

.. note::

   Use a persistent file system for pipelines you need to debug later. Scratch
   file systems are fine for temporary build space, but they may be purged
   independently of the cleanup script below.

Cleaning Old Jacamar Builds
===========================

The repository includes :file:`scripts/cleanup_gitlab_builds.sh` for removing
old Jacamar build directories. The script resolves ``${HOME}/.jacamar-ci``
first, so it works whether ``${HOME}/.jacamar-ci`` is a real directory or a
symlink to a larger file system.

Run it manually first to make sure it is cleaning the expected directory:

.. code-block:: bash

   scripts/cleanup_gitlab_builds.sh -d 4

The script defaults to a dry run. Add ``-f`` after confirming the output to
delete the matched directories:

.. code-block:: bash

   scripts/cleanup_gitlab_builds.sh -d 4 -f

Setting Up a Cron Job
=====================

Install the cleanup script from the repository somewhere stable, such as
``${HOME}/bin/cleanup_gitlab_builds.sh``, and make it executable:

.. code-block:: bash

   cp scripts/cleanup_gitlab_builds.sh "${HOME}/bin/cleanup_gitlab_builds.sh"
   chmod +x "${HOME}/bin/cleanup_gitlab_builds.sh"

Then log in to ``dane`` and edit your crontab there:

.. code-block:: bash

   crontab -e

The following example runs the cleanup every night at 2:00 AM and keeps the
last 4 days of build directories:

.. code-block:: bash

   0 2 * * * /usr/bin/env bash ${HOME}/bin/cleanup_gitlab_builds.sh -d 4 -f

This will email you what got deleted and which machine the cleanup ran on. Keep
that email so you know which login node to SSH into if you need to edit the
cron job later.
