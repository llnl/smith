.. ## Copyright (c) Lawrence Livermore National Security, LLC and
.. ## other Smith Project Developers. See the top-level COPYRIGHT file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)

.. _release-label:

*******************************************
Release Process
*******************************************

The Smith team decides as a group when the code is ready for a release.
Typically, a release is done when we want to make changes available to users;
e.g., when some new functionality is sufficiently complete or we want users to
try something out and give us feedback early in the development process. A
release may also be done when some other development goal is reached. This
section describes how an Smith releases is done. The process is fairly
informal. However, we want to ensure that the software is in a robust and
stable state when a release is done. We follow the process described in this
section to avoid oversights and issues that we do not want to pass on to users.

In the :ref:`gitflow-label` section, we noted that the **main branch
records the official release history of the project**. Specifically,
whenever, the main branch is changed, it is tagged with a new
version number. We use a git 'lightweight tag' for this purpose. Such
a tag is essentially a pointer to a specific commit on the main branch.

When all pull requests intended to be included in a release have been merged
into the develop branch, we create a release candidate branch 
**from the develop branch**. The release candidate branch is used to finalize 
preparations for the release. At this point, the next release cycle begins and
work may continue on the develop branch.

.. important:: No significant code development should performed on a release
               candidate branch. Apart from finalizing release notes and other
               documentation, **no code changes** are to be merged directly
               into a release candidate branch. If a bug is found, it should be
               fixed and merged into develop via a pull request. Then, the
               change should be included in the release candidate branch by
               merging develop into it. Changing code in a release candidate is
               a rare occurrence since the develop branch should always be
               passing CI tests.

When a release candidate branch is ready, it will be merged into the
**main branch** and a release tag will be made. Then, the main branch is
merged into the develop branch so that changes made to finalize the release
are included there.

Here are the steps we follow for an Smith release.

1: Start Release Candidate Branch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create a release candidate branch **off the develop branch** to initiate a
release. The name of a release branch should contain the associated release
version name. Typically, we use a name like v0.5.0-rc
(i.e., version 0.5.0 release candidate). See :ref:`semver-label` for a
description of how version numbers are chosen.

2: Finalize the Release in the Release Candidate Branch 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All changes to Smith related to finalizing the release documentation, as
opposed to source code changes, are done in the release candidate branch.
Typical changes include:

#. Update the version information (major, minor, and patch version numbers)
   near the top of the ``smith/cmake/SmithVersion.cmake`` file.

#. Make any changes that are needed, for correctness and completeness, in the
   section for the new release in the file ``smith/RELEASE-NOTES.md``. This
   should not take much time as release notes should have been updated in
   pull requests that have been merged into develop during regular development.
   See :ref:`release-notes-label` for information about release notes. Add the
   release version number and release date in the section heading and add a
   link to the new version on GitHub at **the bottom of the file.** Please
   follow the established formatting in the file.

#. Update the mail map in ``smith/.mailmap``, if needed, by adding names and 
   emails of new contributors since the last release.

#. Test the code by running it through all continuous integration tests
   and builds. This will be done automatically when the release pull request is
   made. All build configurations must compile properly and all tests must pass
   before the pull request can be merged.

#. Fix any issues discovered during final release testing in a pull request and
   merge that into develop after it is approved and CI checks pass. Then, 
   merge develop into the release candidate branch, and re-run
   appropriate tests to ensure issues are resolved. If a major bug is
   discovered, and it requires significant code modifications to fix,
   do not fix it on the release branch. `Create a new GitHub issue
   <https://github.com/LLNL/smith/issues/new>`_ and note it in the ``known bugs``
   section of the release notes. Alternatively, if time permits, fix the 
   bug in a different branch and create a pull request as you would do during
   regular development. After the bug is resolved and that pull request is
   merged into develop, merge develop into the release candidate branch where
   checks will run on that.

#. Make sure all documentation (source code, user guides, etc.) is
   updated and reviewed. This should not be a substantial undertaking as
   most of this should have been done during the regular development cycle.

.. important:: It is good practice to have everyone on the team review the
               release notes to ensure that they are complete, correct, and
               sufficiently descriptive so that users understand the content
               of the release. **Please make sure the section for the new
               release follows the same organization as in previous release
               sections.**

3: Create a Pull Request for the Release
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create a pull request to merge the release candidate branch into main after
all release preparation changes have been made. When the release candidate
branch is complete, reviewed and approved, it will be merged into main.

Typically, when a release is being prepared it will have been months since the
previous release when the main branch was last changed. Thus, the *diff* between
branches that will appear in the pull request to merge the release candidate
into main will be large. Fortunately, most of those changes will have been
reviewed and merged into the develop branch and do not require additional
review.

To make the review and approval process easier for the release candidate pull
request, it is helpful to create a companion pull request that shows **only
the changes made in the release candidate branch to finalize the release.**
Specifically, the companion pull request is made to merge the release
candidate branch into develop. This pull request will not be merged, but it
will be much easier since it will show only the changes made in the release
candidate that are not in the develop branch.

To facilitate the review process, we cross reference the two pull requests
in their respective pull request descriptions. Suppose the pull request for
the release (to merge the release candidate into main) is #N and the
companion pull request (to merge the release candidate into develop) is #M.
In the description of pull request N, add a link to pull request M and a
comment to **review pull request M and approve pull request N.** In the
description of pull request M, add a link to pull request M and a statement
that it is the companion pull request for N, that it **should be reviewed but
not merged** and that pull request N should be approved as it will be merged
for the release.

4: Merge Release Candidate
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Merge the release candidate branch into the main branch after it has been
approved and all CI checks have passed. Do not "squash merge" as it will make
the histories of main and develop branches disagree, and we want to preserve
the history. After merging, the release candidate branch can be deleted.

5: Draft a GitHub Release
^^^^^^^^^^^^^^^^^^^^^^^^^

`Draft a new Release on GitHub <https://github.com/LLNL/smith/releases/new>`_

#. Enter a Release title. We typically use titles of the following form *Smith-v0.3.1*

#. Select **main** as the target branch to tag a release.

#. Enter the release tag name, such as v0.5.0, and specify to create the tag 
   when the release is published.

#. Copy and paste the information for the release from the
   ``smith/RELEASE-NOTES.md`` into the release description 
   (omit any sections that are empty).

#. Add a statement at the top of the release note, such as:
   Please download the `Smith-*.tar.gz` tarball below, which includes all of the
   submodules needed to build Smith.

#. Click the checkbox at the bottom of the page that says "Set as the latest release".

#. Publish the release. This will create a tag at the tip of the main
   branch and add corresponding entry in the
   `Releases section <https://github.com/LLNL/smith/releases>`_

.. note::

   GitHub will add tarfile and zip archives consisting of the
   source files for each release. However, these files do not include any
   submodules. Consequently, a tarfile that includes all of the submodules is
   generated manually in a separate step described below.

6: Make Release Tarfile
^^^^^^^^^^^^^^^^^^^^^^^^^^

#. After the release is published on GitHub, checkout the main branch locally
   and run the command ``./scripts/make_release_tarball.sh``
   from the top level ``smith`` directory. This will generate a tarfile: for example,
   ``Smith-v0.3.1.tar.gz``, consisting of the smith repo source.

#. Attach the tarfiles for the corresponding release, by going to the
   `GitHub Releases section <https://github.com/LLNL/smith/releases>`_ and
   ``Edit`` the release created in step 5 above. Click ``Paste, drop, or click to add files``
   at the bottom of the release description section and select the
   Smith release and data tarfiles created in the previous step.

#. Click the ``Update release`` button.

7: Add Version to ReadTheDocs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A new version our ReadTheDocs should be created so that users can refer to older snapshots of Smith's documentation.

#. Go to `our ReadTheDocs app page <https://app.readthedocs.org/projects/llnlsmith/>`_.

#. Click "Add version".

#. Find the version tag that was just created in the previous steps. (It might take a couple minutes to show up.)

#. Toggle the "Activate" to be on.

#. Select "Update version".

8: Merge Main to Develop
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create a pull request to merge main into develop so that changes in the 
release candidate branch are integrated into subsequent Smith development.
When approved, merge it.


.. _release-notes-label:

*******************************************
Release Notes
*******************************************

Smith release notes are maintained in a single file ``smith/RELEASE-NOTES.md``.
The release notes for the latest version are at the top of the file.
Notes for previous releases appear after that in descending version number
order.

For each version, the release notes must contain the following information:

 * Smith version number and date of release

 * One or two sentence overview of release, including any major changes.

 * Release note items should be broken out into the following sections:

    * Added: Descriptions of new features
    * Removed: Notable removed functionality
    * Deprecated: Deprecated features that will be removed in a future release
    * Changed: Enhancements or other changes to existing functionality
    * Fixed: Major bug fixes
    * Known bugs: Existing issues that are important for users to know about

.. note:: Release notes for each Smith version should explain what changed in
          that version of the software -- and nothing else!!

Release notes are an important way to communicate software changes to users
(functionality enhancements, new features, bug fixes, etc.). Arguably, they
are the simplest and easiest way to do so. Each change listed in the release
notes should contain a clear, concise statement of the change. Items should
be ordered based on the impact to users (higher impact - first, lower impact
last).

.. note:: When writing release notes, think about what users need to know and
          what is of value to them.

Release notes should summarize new developments and provide enough detail
for users to get a clear sense of what's new. They should be brief -- don't
make them overly verbose or detailed. Provide enough description for users
to understand a change, but no more than necessary. In other words, release
notes summarize major closed issues in a human-readable narrative. Direct
users to other documentation (user guides, software documentation, example
codes) for details and additional information.

.. note:: Release notes should be updated as work is completed and reviewed
          along with other documentation in a pull request. This is much
          easier than attempting to compile release notes before a release
          by looking at commit logs, etc. Preparing release notes as part
          of the release process should take no more than one hour.

Lastly, release notes provide an easy-to-find retrospective record of
progress for users and other stakeholders. They are useful for developers
and for project reporting and reviews.
