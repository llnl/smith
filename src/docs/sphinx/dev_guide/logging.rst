.. ## Copyright (c) Lawrence Livermore National Security, LLC and
.. ## other Smith Project Developers. See the top-level COPYRIGHT file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)

.. _logging-label:

=======
Logging
=======

Logging is done through Axom's `SLIC <https://axom.readthedocs.io/en/develop/axom/slic/docs/sphinx/index.html>`_
component. SLIC provides a lot of configurable logging functionality which we have consolidated the header
``src/infrastructure/logger.hpp`` and implemented in ``src/infrastructure/logger.cpp``.

.. note::
  On parallel runs, messages can be out of order if flush is not called often enough.

Logging Streams
---------------

SLIC has a concept of logging streams.  Logging streams controls the following:

 * How each message is formatted. More info `here <https://axom.readthedocs.io/en/develop/axom/slic/docs/sphinx/sections/architecture.html#log-message-format>`__ .
 * Where each messages are output, such as ``std::cout``, ``std::cerr``, or to a file stream.
 * Logic for handling and filtering of messages, based on message level or content.

Smith creates the following logging streams under different conditions:


   * GenericOutputStream

     * Serial
     * Debug and info messages to ``std::cout``
     * Warning and error messages to ``std::cerr``
     * Logs all messages directly to given streams.

   * LumberjackStream

     * Parallel
     * Debug and info messages to ``std::cout``
     * Warning and error messages to ``std::cerr``
     * Flushing causes messages to be scalably passed and filtered down to rank 0 then outputted.
     * On error, SLIC does a local flush on the aborting rank, then does an ``MPI_Abort``. Messages
       on other ranks since the last flush will be lost.

Message Levels
--------------

SLIC has 4 message levels to help indicate the important of messages. Descriptions are as follows:

 * Debug - messages that help debugging runs, only on when ``SMITH_DEBUG`` is defined
 * Info - basic informational messages
 * Warning - message indicating that something has gone wrong but not enough to end the simulation
 * Error - message indicating a non-recoverable error has occurred

Logging Macros
--------------

SLIC provides many helper macros that assist in logging messages. Here is a list of them but more information
can be found `in the Axom documentation <https://axom.readthedocs.io/en/develop/axom/slic/docs/sphinx/sections/appendix.html#slic-macros-used-in-axom>`_ :

 * ``SLIC_INFO(msg)`` - Logs info message
 * ``SLIC_INFO_IF(expression, msg)`` - Logs info message if expression is true
 * ``SLIC_INFO_ROOT(msg)`` - Logs info message if on rank 0
 * ``SLIC_INFO_ROOT_IF(expression, msg)`` - Logs info message if on rank 0 and expression is true
 * ``SLIC_WARNING(msg)`` - Logs warning message
 * ``SLIC_WARNING_IF(expression, msg)`` - Logs warning message if expression is true
 * ``SLIC_WARNING_ROOT(msg)`` - Logs warning message if on rank 0
 * ``SLIC_WARNING_ROOT_IF(expression, msg)`` - Logs warning message if on rank 0 and expression is true
 * ``SLIC_ERROR(msg)`` - Logs error message
 * ``SLIC_ERROR_IF(expression, msg)`` - Logs error message if expression is true
 * ``SLIC_ERROR_ROOT(msg)`` - Logs error message if on rank 0
 * ``SLIC_ERROR_ROOT_IF(expression, msg)`` - Logs error message if on rank 0 and expression is true

The following macros are compiled out if not in a debug build:

 * ``SLIC_ASSERT(expression)`` - Logs an error if expression is not true
 * ``SLIC_ASSERT_MSG(expression, msg)``  - Logs an error with a custom message if expression is not true
 * ``SLIC_CHECK(expression)`` - Logs an warning if expression is not true
 * ``SLIC_CHECK_MSG(expression, msg)`` - Logs an warning with a custom message if expression is not true
 * ``SLIC_DEBUG(msg)`` - Logs debug message on rank 0
 * ``SLIC_DEBUG_IF(expression, msg)`` - Logs debug message if expression is true
 * ``SLIC_DEBUG_ROOT(msg)`` - Logs debug message if on rank 0
 * ``SLIC_DEBUG_ROOT_IF(expression, msg)`` - Logs debug message if on rank 0 and expression is true
