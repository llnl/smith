.. ## Copyright (c) Lawrence Livermore National Security, LLC and
.. ## other Smith Project Developers. See the top-level COPYRIGHT file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)

###############################
Composable Solid Mechanics Demo
###############################

This example shows a solid-only setup using the composable differentiable
numerics interface. It registers fields, builds a solid system, applies a
Neo-Hookean material with a differentiable Young's modulus field, seeds dynamic
initial conditions, runs a cycle-zero startup solve plus several implicit
Newmark steps, checks shape, parameter, and initial-condition sensitivities,
and writes displacement, velocity, acceleration, and stress to ParaView.

The full source code lives in ``examples/solid_mechanics/composable_solid_mechanics.cpp``.

Includes and Initialization
---------------------------

.. literalinclude:: ../../../../examples/solid_mechanics/composable_solid_mechanics.cpp
   :start-after: _includes_start
   :end-before: _includes_end
   :language: C++

.. literalinclude:: ../../../../examples/solid_mechanics/composable_solid_mechanics.cpp
   :start-after: _init_start
   :end-before: _init_end
   :language: C++

Mesh Construction
-----------------

.. literalinclude:: ../../../../examples/solid_mechanics/composable_solid_mechanics.cpp
   :start-after: _mesh_start
   :end-before: _mesh_end
   :language: C++

Solver and Field Registration
-----------------------------

The field registration phase declares the dynamic displacement state pack,
Young's modulus parameter field, and optional stress output field on a shared
``FieldStore``.

.. literalinclude:: ../../../../examples/solid_mechanics/composable_solid_mechanics.cpp
   :start-after: _solver_start
   :end-before: _solver_end
   :language: C++

System Build and Material Setup
-------------------------------

The solid system is built from the registered field pack. The material wrapper
adapts the standard Neo-Hookean material to the ``TimeInfo``-based interface,
pulling bulk and shear response from the Young's modulus parameter field. The
example also seeds non-zero initial displacement and velocity fields.

.. literalinclude:: ../../../../examples/solid_mechanics/composable_solid_mechanics.cpp
   :start-after: _build_start
   :end-before: _build_end
   :language: C++

Boundary Conditions and Loads
-----------------------------

The left boundary uses a component-wise Dirichlet condition, fixing only the
``x`` and ``z`` displacement components.

.. literalinclude:: ../../../../examples/solid_mechanics/composable_solid_mechanics.cpp
   :start-after: _bc_start
   :end-before: _bc_end
   :language: C++

Advance, Sensitivities, and Reactions
-------------------------------------

.. literalinclude:: ../../../../examples/solid_mechanics/composable_solid_mechanics.cpp
   :start-after: _run_start
   :end-before: _run_end
   :language: C++

Write ParaView Output
---------------------

.. literalinclude:: ../../../../examples/solid_mechanics/composable_solid_mechanics.cpp
   :start-after: _output_start
   :end-before: _output_end
   :language: C++
