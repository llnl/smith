.. ## Copyright (c) Lawrence Livermore National Security, LLC and
.. ## other Smith Project Developers. See the top-level COPYRIGHT file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)

###############################
Composable Solid Mechanics Demo
###############################

This demo builds one composable solid mechanics system. It registers fields,
adds a Young's modulus parameter, applies a Neo-Hookean material, advances a
dynamic solve, checks sensitivities, and writes output.

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

Registration declares the displacement fields and the Young's modulus parameter
on the shared ``FieldStore``. Register parameter fields with
``registerParameterFields(field_store, ...)`` before building the system.

.. literalinclude:: ../../../../examples/solid_mechanics/composable_solid_mechanics.cpp
   :start-after: _solver_start
   :end-before: _solver_end
   :language: C++

System Build and Material Setup
-------------------------------

The build step consumes the registered field pack and parameter bundle. The
material reads the parameter field and uses the ``TimeInfo`` material interface.

.. literalinclude:: ../../../../examples/solid_mechanics/composable_solid_mechanics.cpp
   :start-after: _build_start
   :end-before: _build_end
   :language: C++

Boundary Conditions and Loads
-----------------------------

The left boundary fixes selected displacement components. The body force and
traction provide the applied loads.

.. literalinclude:: ../../../../examples/solid_mechanics/composable_solid_mechanics.cpp
   :start-after: _bc_start
   :end-before: _bc_end
   :language: C++

Physics Creation and Initial Conditions
---------------------------------------

The differentiable physics object owns the timestep loop. The example sets the
parameter field and seeds non-zero initial displacement and velocity fields.

.. literalinclude:: ../../../../examples/solid_mechanics/composable_solid_mechanics.cpp
   :start-after: _ic_start
   :end-before: _ic_end
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
