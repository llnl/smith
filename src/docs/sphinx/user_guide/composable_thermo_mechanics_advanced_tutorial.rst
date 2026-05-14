.. ## Copyright (c) Lawrence Livermore National Security, LLC and
.. ## other Smith Project Developers. See the top-level COPYRIGHT file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)

#############################################
Composable Thermo-Mechanics Advanced Example
#############################################

This example extends the basic thermo-mechanics tutorial with a staged solver,
a parameter field, a differentiable quantity of interest, finite-difference
verification, and ParaView output.

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

This example uses a single block solver combining Newton line search for the non-linear
solve and SuperLU for the linear solve. It also registers the thermal-expansion
scaling parameter directly on the shared ``FieldStore`` with
``registerParameterFields(field_store, ...)`` before either system is built.

.. literalinclude:: ../../../../examples/thermo_mechanics/composable_thermo_mechanics_advanced.cpp
   :start-after: _solver_start
   :end-before: _solver_end
   :language: C++

System Build and Coupling
-------------------------

.. literalinclude:: ../../../../examples/thermo_mechanics/composable_thermo_mechanics_advanced.cpp
   :start-after: _build_start
   :end-before: _build_end
   :language: C++

Boundary Conditions and Loads
-----------------------------

The traction call uses ``DependsOn<>{}``, so the user callback receives only the
state arguments it actually needs and none of the trailing coupling or parameter
fields. The system builders now take the registered ``ParamFields`` bundle
explicitly, so the build order stays ``self_fields``, optional
``couplingFields(...)``, then optional params.

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
