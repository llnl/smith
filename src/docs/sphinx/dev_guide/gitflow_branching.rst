.. ## Copyright (c) Lawrence Livermore National Security, LLC and
.. ## other Smith Project Developers. See the top-level COPYRIGHT file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)

.. _gitflow-label:

*************************
Gitflow Branching Model
*************************

The Smith team uses the 'Gitflow' branch development model, which is
summarized in this section. See the `Atlassian Gitflow Description <https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow>`_ 
for more details.

Gitflow is a branching model centered around software releases. It is a simple 
workflow that makes clear which branches correspond to which phases of 
development and those phases are represented explicitly in the structure of 
the repository. As in other branching models, developers develop code locally 
and push their work to a central repository. 

==============================
Main and Develop Branches
==============================

The **main** and **develop** branches are the two main branches used in Gitflow.
They always exist and the distinction between them is central to the Gitflow
model. Other branches are temporary and used to perform specific development 
tasks.

The main branch records the official release history of the project. 
Each time the main branch is changed, it is tagged with a new version number.
For a description of our versioning scheme, see :ref:`semver-label`.

The develop branch is used to integrate and test new features and most 
bug fixes before they are merged into main. 

.. important:: **Development never occurs directly on the main or develop
               branches.**  

==============================
Topic Branches
==============================

Topic branches are created off of other branches (usually develop)
and are used to develop new features and resolve issues before they 
propagate to main. Topic branches are temporary, living only as long as they
are needed to complete a development task.

Each new feature, or other well-defined portion of work, is developed on its 
own topic branch, with changes being pushed to the central repository regularly
for backup. We typically include a label, such as  "feature" or "bugfix", in 
the topic branch name to make it clear what type of work is being done on the 
branch.

When a feature is complete, a pull request is submitted for review by other 
team members. When all issues arising in a review have been addressed and 
reviewers have approved the pull request, the feature branch is merged into 
develop.

.. important:: **Feature branches never interact directly with the main 
               branch.**

==============================
Release Branches
==============================

Release branches are another important temporary branch type in Gitflow:
When the team has decided that enough features, bug fixes, etc. have been 
merged into develop (for example, all items identified for a release have 
been completed), a release branch is created off of develop to finalize the 
release. Creating a release branch starts the next release cycle on develop. 
At that point, new work can start on feature branches for the next release. 
Only changes required to complete the release are added to a release branch. 
When a release branch is ready, it is merged into main and main is tagged 
with a new version number. Finally, main is merged back into develop since 
it may have changed since the release was initiated.

The basic mechanics for generating a new release of the main branch for the 
Smith project are described in :ref:`release-label`. 

.. important:: **No new features are added to a release branch. Only bug fixes, 
               documentation, and other release-oriented changes go into a 
               release branch.**

==============================
Hotfix Branches
==============================

The last important temporary branch type in Gitflow is a hotfix branch.
Sometimes, there is a need to resolve an issue in a released version on the 
main branch. When the fix is complete, it is reviewed using a pull request 
and then merged into both main and develop when approved. At this point, 
main is tagged with a new version number. A dedicated line of development 
for a bug fix, using a hotfix branch, allows the team to quickly address 
issues without disrupting other parts of the workflow. 

.. important:: Hotfix branches are the only branches created off of main.

==============================
Gitflow Illustrated
==============================

The figure below shows how branches interact in Gitflow.

.. figure:: gitflow-workflow.svg

   This figure shows typical interactions between branches in the Gitflow 
   workflow. Here, main was merged into develop after tagging version v0.1. 
   A fix was needed and so a hotfix branch was created. When the fix was 
   completed, it was merged into main and develop. Main was tagged 
   with version v0.2. Also, work was performed on two feature branches. 
   When one feature branch was done, it was merged into develop. Then, a 
   release branch was created and it was merged into main when the release 
   was finalized. Finally, main was tagged with version v1.0.
