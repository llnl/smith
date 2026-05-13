.. ## Copyright (c) Lawrence Livermore National Security, LLC and
.. ## other Smith Project Developers. See the top-level COPYRIGHT file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)

####################################
Composable Thermo-Mechanics Tutorial
####################################

This tutorial shows a minimal thermo-mechanical setup built from the composable
differentiable numerics systems. It uses separate solid and thermal systems,
couples them with a thermoelastic material, and advances the combined system.

The full source code lives in ``examples/thermo_mechanics/composable_thermo_mechanics.cpp``.

Includes and Initialization
---------------------------

.. literalinclude:: ../../../../examples/thermo_mechanics/composable_thermo_mechanics.cpp
   :start-after: _includes_start
   :end-before: _includes_end
   :language: C++

.. literalinclude:: ../../../../examples/thermo_mechanics/composable_thermo_mechanics.cpp
   :start-after: _init_start
   :end-before: _init_end
   :language: C++

Mesh Construction
-----------------

.. literalinclude:: ../../../../examples/thermo_mechanics/composable_thermo_mechanics.cpp
   :start-after: _mesh_start
   :end-before: _mesh_end
   :language: C++

Field Registration
------------------

Registration is phase 1. It declares the solid and thermal fields up front in a
shared ``FieldStore``. When user parameters are needed, register them in this
same phase with ``registerParameterFields(field_store, ...)`` and carry the
returned ``ParamFields`` into the build step.

.. literalinclude:: ../../../../examples/thermo_mechanics/composable_thermo_mechanics.cpp
   :start-after: _solver_start
   :end-before: _solver_end
   :language: C++

System Build and Coupling
-------------------------

Build is phase 2. Each system consumes its own registered field pack first, then
the other system's field pack for coupling. ``combineSystems(...)`` returns the
final combined system directly, and ``makeDifferentiablePhysics(...)`` later uses
the system-owned cycle-zero and post-solve systems automatically.

.. literalinclude:: ../../../../examples/thermo_mechanics/composable_thermo_mechanics.cpp
   :start-after: _build_start
   :end-before: _build_end
   :language: C++

Boundary Conditions and Loads
-----------------------------

.. literalinclude:: ../../../../examples/thermo_mechanics/composable_thermo_mechanics.cpp
   :start-after: _bc_start
   :end-before: _bc_end
   :language: C++

Advance the Coupled System
--------------------------

.. literalinclude:: ../../../../examples/thermo_mechanics/composable_thermo_mechanics.cpp
   :start-after: _run_start
   :end-before: _run_end
   :language: C++

This example intentionally stays small. Use it as a template for:

- adding parameter fields,
- enabling stress output on the solid system,
- or extending the model to thermo-mechanics plus internal variables.
