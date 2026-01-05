.. ## Copyright (c) Lawrence Livermore National Security, LLC and
.. ## other Smith Project Developers. See the top-level COPYRIGHT file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)

.. _macmini-label:

==============
Smith Mac Mini
==============

------------------------------------------
Adding an Additional User to Smith MacMini
------------------------------------------

This page assumes you are a Smith developer who requires access to the team's shared MacMini. This machine
tests Smith Mac builds on a regular basis via cron. If you have any questions, reach out to either
`Brandon Talamini <talamini1@llnl.gov>`_, `Alex Chapman <chapman39@llnl.gov>`_, or LivIT. The following
are steps to guide you to gaining access on the machine to the point you're able to build Smith.

1. **Add User**

Without a MyPass, you can still log in using your LLNL username and AD password. Do this first to setup an account on the machine.
You won't be able to do much, since you do not have access to FileVault, and you are not an admin... yet.

2. **MyPass**

Then, acquire a new MyPass with a USB-C port dedicated to this machine. This will grant you access to FileVault.
Contact LivIT directly to setup an appointment and request one.

3. **EARS Admin Request**

Next, request admin access to the machine by visiting either ServiceNow or the `EARS website <https://ears.llnl.gov/dashboard>`_.

4. **Two Logins**

Once you have a MyPass and you have admin rights, try to log in again. There should be two passwords required to log in. The first one
is for your MyPass (assuming it's plugged into to the machine) and the other is for the account login.

5. **Download and setup Homebrew**

Visit `Homebrew's website <https://brew.sh/>`_ to install and setup Homebrew. This is required to install some of Smith's third-party libraries
(TPLs).

6. **Add New SSH Key to GitHub**

Next step setting up a new SSH Key to your GitHub account so that you're able to clone the Smith repo. In case you do not know
how to do this, instructions can be found on
`GitHub's website <https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account>`_.

7. **Install Smith**

You're now able to clone Smith and get started with the installation process. Further instructions for doing so are currently on 
the `quickstart page <https://llnlsmith.readthedocs.io/en/latest/sphinx/quickstart.html#quickstart-label>`_ of the Smith documentation.

----------------
Cron Job Example
----------------

The following is an example of a cron job that could be used to test a Mac build, assuming TPLs have been built for Smith. Run
``crontab -e`` to edit the cron file.

.. code-block:: bash

    0 7 * * 4 /Users/chapman39/dev/smith/ci/repo/scripts/shared-macmini/build-and-test.sh
