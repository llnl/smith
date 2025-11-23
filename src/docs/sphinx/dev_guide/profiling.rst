.. ## Copyright (c) Lawrence Livermore National Security, LLC and
.. ## other Smith Project Developers. See the top-level COPYRIGHT file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)

=======================================
Profiling Smith using Adiak and Caliper
=======================================

Introduction to Adiak
---------------------

`Adiak <https://github.com/LLNL/Adiak>`_ is a library developed at LLNL for collecting
metadata that can be used to compare multiple runs across programs.  For more information,
read `Adiak's documentation <https://github.com/LLNL/Adiak/blob/master/docs/Adiak%20API.docx>`_. Note that Smith provides some wrapper functions to initialize and finalize Adiak
metadata collection.

Introduction to Caliper
-----------------------

`Caliper <https://github.com/LLNL/Caliper>`_ is a framework developed at LLNL for
measuring the performance of programs.  To find out more, read `Caliper's documentation 
<https://software.llnl.gov/Caliper>`_. Smith also provides convenient macros
that make it easy to instrument and assess the performance of simulation code.

Introduction to SPOT
--------------------

`SPOT <https://software.llnl.gov/news/2021/01/07/spot-new>`_ is a framework developed at
LLNL for visualizing performance data.  SPOT is an external tool and does not need to be
linked into Smith.

TPL Build Instructions
----------------------

To use Adiak and Caliper with Smith, install the ``profiling`` variant of ``smith``
with Spack, i.e., ``smith+profiling``. Note that these libraries are pre-built as
part of the installed set of libraries on LC.

Instrumenting Code
------------------

To use the functions and macros described in the remainder of this section, the ``smith/infrastructure/profiling.hpp`` header must be included.

To enable Adiak and Caliper for a program, call ``smith::profiling::initialize()``.
This will begin the collection of metadata and performance data. Optionally, an MPI
communicator can be passed to configure Adiak and a Caliper `ConfigManager configuration string <https://software.llnl.gov/Caliper/ConfigManagerAPI.html#configmanager-configuration-string-syntax>`_
can be passed to configure Caliper. Note that you must still annotate regions to be
profiled and provide any custom metadata.

Call ``smith::profiling::finalize()`` to conclude metadata and performance monitoring
and to write the data to a ``.cali`` file.

To provide custom metadata for comparing program runs, call ``SMITH_SET_METADATA(name, data)``
after ``smith::profiling::initialize()`` and before ``smith::profiling::finalize``.
This will add extra metadata into the ``.cali`` file. Supported metadata types are
integrals, floating points, and strings. Note that this macro is a no-op if the
``profiling`` variant is not used.

.. code-block:: c++
		
   SMITH_SET_METADATA("dimensions", 2);
   SMITH_SET_METADATA("mesh", "../data/star.mesh");

To add profile regions and ensure that Caliper is only used when it has been enabled
through Spack, only use the macros described below to instrument your code:

Use ``SMITH_MARK_FUNCTION`` at the very top of a function to mark it for profiling.

Use ``SMITH_MARK_BEGIN(name)`` at the beginning of a region and ``SMITH_MARK_END(name)`` at the end of the region.

Use ``SMITH_MARK_LOOP_BEGIN(id, name)`` before a loop to mark it for profiling, ``SMITH_MARK_LOOP_ITERATION(id, i)`` at the beginning
of the  ``i`` th iteration of a loop, and ``SMITH_MARK_LOOP_END(id)`` immediately after the loop ends:

.. code-block:: c++

  SMITH_MARK_BEGIN("region_name");
   
  SMITH_MARK_LOOP_BEGIN(doubling_loop, "doubling_loop");
  for (int i = 0; i < input.size(); i++)
  {
    SMITH_MARK_LOOP_ITERATION(doubling_loop, i);
    output[i] = input[i] * 2;
  }
  SMITH_MARK_LOOP_END(doubling_loop);

  SMITH_MARK_END("region_name");


Note that the ``id`` argument to the ``SMITH_MARK_LOOP_*`` macros can be any identifier as long as it is consistent
between all uses of ``SMITH_MARK_LOOP_*`` for a given loop.  

To reduce the amount of annotation for regions bounded by a particular scope, use ``SMITH_MARK_SCOPE(name)``. This will follow RAII and works with graceful exception handling. When ``SMITH_MARK_SCOPE`` is instantiated, profiling of this region starts, and when the scope exits, profiling of this region will end.

.. code-block:: c++

   // Refine once more and utilize SMITH_MARK_SCOPE
  {
    SMITH_MARK_SCOPE("RefineOnceMore");
    pmesh->UniformRefinement();
  }


Performance Data
----------------

The metadata and performance data are output to a ``.cali`` file. To analyze the contents
of this file, use `cali-query <https://software.llnl.gov/Caliper/tools.html#cali-query>`_.

To view this data with SPOT, open a browser, navigate to the SPOT server (e.g. `LC <https://lc.llnl.gov/spot2>`_), and open the directory containing one or more ``.cali`` files.  For more information, watch this recorded `tutorial <https://www.youtube.com/watch?v=p8gjA6rbpvo>`_.

.. _benchmarking-label:

Benchmarking Smith
------------------

To run all of Smith's benchmarks in one command, first make sure Smith is configured
with benchmarking enabled (off by default). Then, run the build target ``run_benchmarks``.
Make sure benchmarks are enabled and the build type is release.

.. code-block:: bash

  ./config-build.py -hc <host config file> -bt Release -DENABLE_BENCHMARKS=ON
  cd <smith build location>
  make -j
  make run_benchmarks
  pwd

This will run all of Smith's benchmarks multiple times with varying MPI task counts, and generate a Caliper file for
each benchmark run at ``PROJECT_BINARY_DIR``. Now, you can visualize the results with SPOT, entering the path printed
from ``pwd``.

Visualizing Benchmarks using SPOT
---------------------------------

If you have access to LC, you can go to the following website and enter a directory in CZ/ RZ that contains Caliper
files:

- `SPOT CZ <https://lc.llnl.gov/spot2>`_
- `SPOT RZ <https://rzlc.llnl.gov/spot2>`_

Smith benchmarks are run weekly to track changes over time. The following are steps to visualize this data in a meaningful
way:

- Go to https://lc.llnl.gov/spot2/?sf=/usr/workspace/smithdev/califiles/smith
- Click the check mark button on the top right to view additional data categories
- Ensure ``mpi.world.size``, ``executable``, ``cluster``, and ``compilers`` are enabled
- Find the pie and bar charts associated with those categories
- Select one option from each category to filter the graph
- Scroll down to the table and and select the "compare" tab to view the graph

Filtering benchmarks in this way will allow you to see changes of one benchmark over time, rather than a mix of many
different ones. When changing the filter options in the pie and bar charts, ensure you deselect the previous options, so
you don't view two of one single category.

.. note::
  There is a bug in SPOT where if you remove Caliper files from a directory, they still show up on SPOT - if you've
  visualized them previously. The current workaround is by removing the ``llnl.gov`` site cache manually.

Compare a PR's benchmarks vs Develop
------------------------------------

Utilizing Hatchet, it is possible to view the performance changes of a prospective PR before it merges into
develop. This process has been conveniently wrapped in a CI pipeline. This Hatchet comparison can only be performed
on LC, since the baseline benchmarks are generated on LC systems.

1. Go to the following CZ GitLab page to create a new pipeline https://lc.llnl.gov/gitlab/smith/smith/-/pipelines/new
2. Choose your branch
3. Under variables, add ``SMITH_CI_WORKFLOW_TYPE`` and ``comparison`` for the key and value, respectively

It's possible to perform this comparison locally. Since baseline benchmarks are generated across different machines and
compilers, a single build won't compare against all baselines. The benchmarks can be compared using dane-gcc and dane-clang builds.

1. Run benchmarks (see :ref:`Benchmarking Smith <benchmarking-label>` above)
2. ``../scripts/llnl/compare_benchmarks.py --current-cali-dir /path/to/caliper/files``

The script generates Hatchet graph frames by calculating the difference between each associated baseline and local
benchmark (``gf_diff = gf_current - gf_baseline``). If there is a positive difference, that means your benchmarks ran
that many seconds slower.

By default, ``compare_benchmarks.py`` will print a table containing the status, id, difference, baseline, and current
times. Running with the verbose option will additionally print the "difference" Hatchet graph frame for each benchmark.
