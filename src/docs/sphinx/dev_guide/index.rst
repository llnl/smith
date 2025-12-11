.. ## Copyright (c) Lawrence Livermore National Security, LLC and
.. ## other Smith Project Developers. See the top-level COPYRIGHT file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)

===============
Developer Guide
===============

.. toctree::
   :hidden:
   :maxdepth: 2

   tpl_build.rst
   style_guide
   docker_env
   modern_cpp
   testing
   logging
   profiling
   memory_checking
   new_docker_image
   tensor_dual
   functional
   codevelop
   state_manager
   equation_solver
   macmini

Developing a New Physics Module
-------------------------------

Developers have two workflows for creating new physics modules:

1. Creating a new multi-physics module from existing physics modules.
#. Creating a new single physics PDE simulation module.

In the first case, construct the new physics module by including existing physics modules by composition. See the `Thermal solid mechanics <../../doxygen/html/classsmith_1_1ThermalSolid.html>`_ module for an example.

For the second case, starting with an existing physics module and re-writing it as necessary is a good practice. The following steps describe creation of a new physics module:

1. Create a new class derived from `BasePhysics <../../doxygen/html/classsmith_1_1BasePhysics.html>`_.
#. In the constructor, create new ``std::shared_ptrs`` to `FiniteElementStates <../../doxygen/html/classsmith_1_1FiniteElementState.html>`_ corresponding to each state variable in your PDE.
#. Link these states to the state pointer array in the ``BasePhysics`` class.
#. Create methods for defining problem parameters (e.g. material properties and sources).
#. Create methods for defining boundary conditions. These should be stored as `BoundaryConditions <../../doxygen/html/classsmith_1_1BoundaryCondition.html>`_ and managed in the ``BasePhysics``'s `BoundaryConditionManager <../../doxygen/html/classsmith_1_1BoundaryConditionManager.html>`_.
#. Override the virtual ``completeSetup()`` method. This should include construction of all of the data structures needed for advancing the timestep of the PDE.
#. Override the virtual ``advanceTimestep()`` method. This should solve the discretized PDE based on the chosen time integration method. This often requires defining ``mfem::Operators`` to use MFEM-based nonlinear and time integration methods. 

Important Data Structures
-------------------------

* `BasePhysics <../../doxygen/html/classsmith_1_1BasePhysics.html>`_: Interface class for a generic PDE simulation module.
* `BoundaryCondition <../../doxygen/html/classsmith_1_1BoundaryCondition.html>`_: Class for storage of boundary condition-related data.
* `BoundaryConditionManager <../../doxygen/html/classsmith_1_1BoundaryConditionManager.html>`_: Storage class for related boundary conditions.
* `EquationSolver <../../doxygen/html/classsmith_1_1mfem__ext_1_1EquationSolver.html>`_: Class for solving nonlinear and linear systems of equations.
* `FiniteElementState <../../doxygen/html/classsmith_1_1FiniteElementState.html>`_: Data structure describing a solution field and its underlying finite element discretization.

Source Code Documentation
-------------------------

Doxygen documentation for the Smith source code is located in the `Doxygen directory <../../doxygen/html/index.html>`_.
