.. ## Copyright (c) Lawrence Livermore National Security, LLC and
.. ## other Smith Project Developers. See the top-level COPYRIGHT file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)

====================
Command Line Options
====================

Below is the documentation for Smith's command line options:

.. list-table:: Options
   :widths: 25 25 25 25
   :header-rows: 1

   * - Long form
     - Short form
     - Variable Type
     - Description
   * - --help
     - -h
     - N/A
     - Print this help message and exit
   * - --input-file
     - -i
     - Path
     - Input file to use
   * - --restart-cycle
     - -c
     - Integer
     - Cycle to restart from
   * - --create-input-file-docs
     - -d
     - N/A
     - Writes Sphinx documentation for input file, then exits
   * - --output-directory
     - -o
     - Path
     - Directory to put outputted files
   * - --paraview
     - -p
     - N/A
     - Enable ParaView output
   * - --print-unused
     - -u
     - N/A
     - Prints unused entries in input file, then exits
   * - --version
     - -v
     - N/A
     - Print version and provenance information, then exits
