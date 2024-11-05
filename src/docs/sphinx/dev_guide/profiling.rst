.. ## Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
.. ## other Serac Project Developers. See the top-level COPYRIGHT file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)

=======================================
Profiling Serac using Adiak and Caliper
=======================================

Introduction to Adiak
---------------------

`Adiak <https://github.com/LLNL/Adiak>`_ is a library developed at LLNL for collecting
metadata that can be used to compare multiple runs across programs.  For more information,
read `Adiak's documentation <https://github.com/LLNL/Adiak/blob/master/docs/Adiak%20API.docx>`_. Note that Serac provides some wrapper functions to initialize and finalize Adiak
metadata collection.

Introduction to Caliper
-----------------------

`Caliper <https://github.com/LLNL/Caliper>`_ is a framework developed at LLNL for
measuring the performance of programs.  To find out more, read `Caliper's documentation 
<https://software.llnl.gov/Caliper>`_. Serac also provides convenient macros
that make it easy to instrument and assess the performance of simulation code.

Introduction to SPOT
--------------------

`SPOT <https://software.llnl.gov/news/2021/01/07/spot-new>`_ is a framework developed at
LLNL for visualizing performance data.  SPOT is an external tool and does not need to be
linked into Serac.

TPL Build Instructions
----------------------

To use Adiak and Caliper with Serac, install the ``profiling`` variant of ``serac``
with Spack, i.e., ``serac+profiling``. Note that these libraries are pre-built as
part of the installed set of libraries on LC.

Instrumenting Code
------------------

To use the functions and macros described in the remainder of this section, the ``serac/infrastructure/profiling.hpp`` header must be included.

To enable Adiak and Caliper for a program, call ``serac::profiling::initialize()``.
This will begin the collection of metadata and performance data. Optionally, an MPI
communicator can be passed to configure Adiak and a Caliper `ConfigManager configuration string <https://software.llnl.gov/Caliper/ConfigManagerAPI.html#configmanager-configuration-string-syntax>`_
can be passed to configure Caliper. Note that you must still annotate regions to be
profiled and provide any custom metadata.

Call ``serac::profiling::finalize()`` to conclude metadata and performance monitoring
and to write the data to a ``.cali`` file.

To provide custom metadata for comparing program runs, call ``SERAC_SET_METADATA(name, data)``
after ``serac::profiling::initialize()`` and before ``serac::profiling::finalize``.
This will add extra metadata into the ``.cali`` file. Supported metadata types are
integrals, floating points, and strings. Note that this macro is a no-op if the
``profiling`` variant is not used.

.. code-block:: c++
		
   SERAC_SET_METADATA("dimensions", 2);
   SERAC_SET_METADATA("mesh", "../data/star.mesh");

To add profile regions and ensure that Caliper is only used when it has been enabled
through Spack, only use the macros described below to instrument your code:

Use ``SERAC_MARK_FUNCTION`` at the very top of a function to mark it for profiling.

Use ``SERAC_MARK_BEGIN(name)`` at the beginning of a region and ``SERAC_MARK_END(name)`` at the end of the region.

Use ``SERAC_MARK_LOOP_BEGIN(id, name)`` before a loop to mark it for profiling, ``SERAC_MARK_LOOP_ITERATION(id, i)`` at the beginning
of the  ``i`` th iteration of a loop, and ``SERAC_MARK_LOOP_END(id)`` immediately after the loop ends:

.. code-block:: c++

  SERAC_MARK_BEGIN("region_name");
   
  SERAC_MARK_LOOP_BEGIN(doubling_loop, "doubling_loop");
  for (int i = 0; i < input.size(); i++)
  {
    SERAC_MARK_LOOP_ITERATION(doubling_loop, i);
    output[i] = input[i] * 2;
  }
  SERAC_MARK_LOOP_END(doubling_loop);

  SERAC_MARK_END("region_name");


Note that the ``id`` argument to the ``SERAC_MARK_LOOP_*`` macros can be any identifier as long as it is consistent
between all uses of ``SERAC_MARK_LOOP_*`` for a given loop.  

To reduce the amount of annotation for regions bounded by a particular scope, use ``SERAC_MARK_SCOPE(name)``. This will follow RAII and works with graceful exception handling. When ``SERAC_MARK_SCOPE`` is instantiated, profiling of this region starts, and when the scope exits, profiling of this region will end.

.. code-block:: c++

   // Refine once more and utilize SERAC_MARK_SCOPE
  {
    SERAC_MARK_SCOPE("RefineOnceMore");
    pmesh->UniformRefinement();
  }


Performance Data
----------------

The metadata and performance data are output to a ``.cali`` file. To analyze the contents
of this file, use `cali-query <https://software.llnl.gov/Caliper/tools.html#cali-query>`_.

To view this data with SPOT, open a browser, navigate to the SPOT server (e.g. `LC <https://lc.llnl.gov/spot2>`_), and open the directory containing one or more ``.cali`` files.  For more information, watch this recorded `tutorial <https://www.youtube.com/watch?v=p8gjA6rbpvo>`_.

Benchmarking Serac
------------------

To run all of Serac's benchmarks in one command, first make sure Serac is configured
with benchmarking enabled (off by default). Then, run the build target ``run_benchmarks``.

.. code-block:: bash

  ./config-build.py -hc <host config file> -DENABLE_BENCHMARKS=ON
  cd <serac build location>
  make -j
  make run_benchmarks
  pwd

This will run all of Serac's benchmarks multiple times with varying MPI task counts, and generate a Caliper file for
each benchmark run at ``PROJECT_BINARY_DIR``. Now, you can visualize the results with SPOT, entering the path printed
from ``pwd``.

Visualizing Benchmarks using SPOT
---------------------------------

If you have access to LC, you can go to the following website and enter a directory in CZ/ RZ that contains Caliper
files:

- `SPOT CZ <https://lc.llnl.gov/spot2>`_
- `SPOT RZ <https://rzlc.llnl.gov/spot2>`_

The shared Caliper files for Serac are located here: https://lc.llnl.gov/spot2/?sf=/usr/WS2/smithdev/califiles

.. note::
  There is a bug in SPOT where if you remove Caliper files from a directory, they still show up on SPOT - if you've
  visualized them previously. The current workaround is by removing the ``llnl.gov`` site cache manually.
