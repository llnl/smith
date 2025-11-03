.. ## Copyright (c) Lawrence Livermore National Security, LLC and
.. ## other Smith Project Developers. See the top-level COPYRIGHT file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)

==========
User Guide
==========

.. toctree::
  :hidden:
  :maxdepth: 2

  simple_conduction_tutorial
  command_line_options
  input_schema

Smith can be used either by providing input files to the main executable or through a C++ API. Example lua input files are located in the `data 
directory <https://github.com/LLNL/smith/tree/develop/data/input_files>`_ and examples of how to use the C++ API are located in the `tests directory 
<https://github.com/LLNL/smith/tree/develop/tests>`_.

Physics Module C++ Interface
----------------------------

A fundamental data structure in Smith is `BasePhysics <../../doxygen/html/classsmith_1_1BasePhysics.html>`_. Classes derived from ``BasePhysics`` are expected to encapsulate a specific partial differential equation and all of the state data and parameters associated with it. Currently, Smith contains the following physics modules:

* `Solid mechanics <../../doxygen/html/classsmith_1_1SolidMechanics.html>`_
* `Heat transfer <../../doxygen/html/classsmith_1_1HeatTransfer.html>`_
* `Thermal solid mechanics <../../doxygen/html/classsmith_1_1Thermomechanics.html>`_

If you would like to include Smith's simulation capabilities in your software project, these are the classes to include. To set up and use a physics module:

1. Construct the appropriate physics module class using a ``mfem::ParMesh`` and a polynomial order of approximation.
#. Set the material properties via ``mfem::Coefficients``.
#. Set the boundary conditions via a ``std::set`` of boundary attributes and a ``mfem::Coefficient``.
#. Set the right hand side source terms (e.g. body forces).
#. Set the `time integration scheme <../../doxygen/html/solver__config_8hpp.html>`_ (e.g. quasi-static or backward Euler). Note that not all time integrators are available for all physics modules.
#. Complete the setup of the physics module by calling ``completeSetup()``. This allocates and builds all of the underlying linear algebra data structures.
#. Advance the timestep by calling ``advanceTimestep(double dt)``. 
#. Output the state variables in VisIt, and optionally ParaView, format by calling ``outputState()``. You can also access the underlying `state data <../../doxygen/html/classsmith_1_1FiniteElementState.html>`_ via the generic ``getState()`` or physics-specific calls (e.g. ``temperature()``).
