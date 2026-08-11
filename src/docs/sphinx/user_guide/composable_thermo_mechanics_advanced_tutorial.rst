.. ## Copyright (c) Lawrence Livermore National Security, LLC and
.. ## other Smith Project Developers. See the top-level COPYRIGHT file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)

#############################################
Composable Thermo-Mechanics Advanced Example
#############################################

This demo extends the minimal thermo-mechanics setup. It adds a parameter field,
a staged solver, a differentiable quantity of interest, a finite-difference
check, and ParaView output.

The full source code lives in ``examples/thermo_mechanics/composable_thermo_mechanics_advanced.cpp``.

Includes and Initialization
---------------------------

.. literalinclude:: ../../../../examples/thermo_mechanics/composable_thermo_mechanics_advanced.cpp
   :start-after: _includes_start
   :end-before: _includes_end
   :language: C++

.. literalinclude:: ../../../../examples/thermo_mechanics/composable_thermo_mechanics_advanced.cpp
   :start-after: _init_start
   :end-before: _init_end
   :language: C++

Mesh and Field Setup
--------------------

.. literalinclude:: ../../../../examples/thermo_mechanics/composable_thermo_mechanics_advanced.cpp
   :start-after: _mesh_start
   :end-before: _mesh_end
   :language: C++

Solver Config and Field Registration
------------------------------------

Registration declares solid, thermal, and parameter fields on one shared
``FieldStore``. The thermal-expansion parameter is registered with
``registerParameterFields(field_store, ...)`` before either system is built.

.. literalinclude:: ../../../../examples/thermo_mechanics/composable_thermo_mechanics_advanced.cpp
   :start-after: _solver_start
   :end-before: _solver_end
   :language: C++

System Build and Coupling
-------------------------

The build step creates solid and thermal systems from the registered field
packs. ``combineSystems(...)`` attaches them to one staged solver.

.. literalinclude:: ../../../../examples/thermo_mechanics/composable_thermo_mechanics_advanced.cpp
   :start-after: _build_start
   :end-before: _build_end
   :language: C++

Boundary Conditions and Loads
-----------------------------

Boundary conditions are applied on the left and right boundaries. Loads are
added through the solid and thermal systems before timestepping.

.. literalinclude:: ../../../../examples/thermo_mechanics/composable_thermo_mechanics_advanced.cpp
   :start-after: _bc_start
   :end-before: _bc_end
   :language: C++

QoI Definition and Timestep Advance
-----------------------------------

.. literalinclude:: ../../../../examples/thermo_mechanics/composable_thermo_mechanics_advanced.cpp
   :start-after: _qoi_start
   :end-before: _qoi_end
   :language: C++

.. literalinclude:: ../../../../examples/thermo_mechanics/composable_thermo_mechanics_advanced.cpp
   :start-after: _run_start
   :end-before: _run_end
   :language: C++

ParaView Output
---------------

.. literalinclude:: ../../../../examples/thermo_mechanics/composable_thermo_mechanics_advanced.cpp
   :start-after: _output_start
   :end-before: _output_end
   :language: C++

Sensitivity and Finite-Difference Check
---------------------------------------

.. literalinclude:: ../../../../examples/thermo_mechanics/composable_thermo_mechanics_advanced.cpp
   :start-after: _sensitivity_start
   :end-before: _sensitivity_end
   :language: C++
