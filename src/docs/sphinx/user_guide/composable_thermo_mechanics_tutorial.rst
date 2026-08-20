.. ## Copyright (c) Lawrence Livermore National Security, LLC and
.. ## other Smith Project Developers. See the top-level COPYRIGHT file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)

####################################
Composable Thermo-Mechanics Tutorial
####################################

This demo builds a minimal composable thermo-mechanics problem. It registers
solid and thermal fields, builds two systems, couples them with a thermoelastic
material, and advances the combined system.

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

Registration declares all fields on one shared ``FieldStore`` before any system
is built. Register parameters in the same phase with
``registerParameterFields(field_store, ...)`` and pass the returned bundle into
the build step.

.. literalinclude:: ../../../../examples/thermo_mechanics/composable_thermo_mechanics.cpp
   :start-after: _solver_start
   :end-before: _solver_end
   :language: C++

System Build and Coupling
-------------------------

The build step consumes each system's own field pack, then optional coupling and
parameter bundles. ``combineSystems(...)`` returns the combined system used by
``makeDifferentiablePhysics(...)``.

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

This demo is intentionally small. Use it as a template for:

- adding parameter fields
- enabling stress output
- adding internal variables
