.. ## Copyright (c) Lawrence Livermore National Security, LLC and
.. ## other Smith Project Developers. See the top-level COPYRIGHT file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)

.. _testing-label:

=======
Testing
=======

Smith has two levels of tests, unit and integration. Unit tests are used to test
individual components of code, such as a class or function.  While integration tests
are for testing the code as a whole. For example, testing the ``smith`` driver with
an input file against blessed answers.

Unit Tests
----------

Unit Tests can be ran via the build target ``test`` after building the code.


Integration Tests
-----------------

.. note::
  Integration testing is in development and not fully featured.

Requirements:

* Installed ATS
* ``ATS_EXECUTABLE`` defined in the host-config (added automatically to
  Spack generated host-configs) or on 
  command line via ``-DATS_EXECUTABLE=/path/to/ats``.
* If using a personal machine, check the ``ats-config`` directory in the Smith
  repo and create a json file ``<your_machine_name>.json`` if you haven't already.
  Your machine's name can be found by running the following lines of code::

      $ python3
      >>> import socket
      >>> socket.gethostname().rstrip('1234567890')
      >>> exit()

  Currently, there are configuration json files for TOSS4 and BlueOS which can be
  used as reference.

#. **Build the code.**
   Build code with the normal steps. More info in the :ref:`quickstart-label`.
   This generates a script in the build directory called ``ats.sh``.

#. **Run integration tests.**
   Run the corresponding command for the system you are on::

     # BlueOS
     $ lalloc 2 ./ats.sh
     
     # TOSS4
     $ salloc -N2 ./ats.sh
     
     # Personal Machine (currently runs subset of tests)
     $ ./ats.sh

   Append ``--help`` to the command to see the current options.

#. **View results.**
   ATS gives a running summary and the final results.  ATS also outputs the following
   helpful files in the platform and timestamp specific created log directory:

   * ``ats.log`` - All output of ATS
   * ``atss.log`` - Short summary of the run
   * ``atsr.xml`` - JUnit test summary

   ATS also outputs both a ``.log`` and ``.log.err`` for each test and checker that is run.

#. **Rebaseline tests (as needed).**
   If tolerances to tests need to be updated, first ensure you've generated new tolerances by running the integration
   tests like mentioned above. Then, use the ``-b`` option::

     # Single baseline
     $ ./ats.sh -b dyn_solve_serial

     # Comma-separated baselines
     $ ./ats.sh -b dyn_solve_serial,dyn_solve_parallel

     # All baselines
     $ ./ats.sh -b all

   This will update the json files located in the `smith_tests <https://github.com/LLNL/smith_tests>`_ submodule. To
   avoid Caliper files from additionally being generated, configure with ``-DENABLE_BENCHMARKS=OFF``.


Installing ATS
--------------

ATS can be installed via the normal devtools install process.
More info on :ref:`devtools-label`. This method is useful because it
builds all development tools in one process.

If you want to install ATS by itself, ATS provides multiple methods to install in
their `Getting Started section <https://github.com/LLNL/ATS#getting-started>`_.


ATS Test Helper Functions
-------------------------

We provide the following test helper functions to make defining integration tests
easier in ``tests/test.ats``.

* ``tolerance_test``

   .. literalinclude:: ../../../../tests/integration/test.ats
      :start-after: _smith_tolerance_test_start
      :end-before: _smith_tolerance_test_end
      :language: text
      :dedent: 4

