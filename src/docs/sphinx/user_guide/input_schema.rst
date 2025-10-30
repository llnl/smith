.. ## Copyright (c) Lawrence Livermore National Security, LLC and
.. ## other Smith Project Developers. See the top-level COPYRIGHT file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)

====================
Input File Structure
====================

Below is the documentation for Smith input files, generated automatically by `Axom's inlet component <https://axom.readthedocs.io/en/develop/axom/inlet/docs/sphinx/index.html>`_.

.. |uncheck|    unicode:: U+2610 .. UNCHECKED BOX
.. |check|      unicode:: U+2611 .. CHECKED BOX

----------
input_file
----------

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - dt
     - Time step.
     - 0.250000
     - 
     - |uncheck|
   * - t_final
     - Final time for simulation.
     - 1.000000
     - 
     - |uncheck|

-------------
thermal_solid
-------------

Description: Thermal solid module


----------------------
coef_thermal_expansion
----------------------

Description: Coefficient of thermal expansion for isotropic thermal expansion

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - constant
     - The constant scalar value to use as the coefficient
     - 
     - 
     - |uncheck|
   * - component
     - The vector component to which the scalar coefficient should be applied
     - 
     - 
     - |uncheck|
.. list-table:: Functions
   :widths: 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Function Name
     - Description
     - Signature
     - Required
   * - scalar_function
     - The function to use for an mfem::FunctionCoefficient
     - Double(Vector, Double)
     - |uncheck|
   * - vector_function
     - The function to use for an mfem::VectorFunctionCoefficient
     - Vector(Vector, Double)
     - |uncheck|

---------------
vector_constant
---------------

Description: The constant vector to use as the coefficient

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - z
     - z-component of vector
     - 
     - 
     - |uncheck|
   * - x
     - x-component of vector
     - 
     - 
     - |uncheck|
   * - y
     - y-component of vector
     - 
     - 
     - |uncheck|

---------------------
reference_temperature
---------------------

Description: Coefficient for the reference temperature for isotropic thermal expansion

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - constant
     - The constant scalar value to use as the coefficient
     - 
     - 
     - |uncheck|
   * - component
     - The vector component to which the scalar coefficient should be applied
     - 
     - 
     - |uncheck|
.. list-table:: Functions
   :widths: 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Function Name
     - Description
     - Signature
     - Required
   * - scalar_function
     - The function to use for an mfem::FunctionCoefficient
     - Double(Vector, Double)
     - |uncheck|
   * - vector_function
     - The function to use for an mfem::VectorFunctionCoefficient
     - Vector(Vector, Double)
     - |uncheck|

---------------
vector_constant
---------------

Description: The constant vector to use as the coefficient

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - z
     - z-component of vector
     - 
     - 
     - |uncheck|
   * - x
     - x-component of vector
     - 
     - 
     - |uncheck|
   * - y
     - y-component of vector
     - 
     - 
     - |uncheck|

-----
solid
-----

Description: Finite deformation solid mechanics module

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - order
     - polynomial order of the basis functions.
     - 1
     - 1 to 3
     - |uncheck|

----------------
initial_velocity
----------------

Description: Coefficient for initial condition

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - constant
     - The constant scalar value to use as the coefficient
     - 
     - 
     - |uncheck|
   * - component
     - The vector component to which the scalar coefficient should be applied
     - 
     - 
     - |uncheck|
.. list-table:: Functions
   :widths: 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Function Name
     - Description
     - Signature
     - Required
   * - scalar_function
     - The function to use for an mfem::FunctionCoefficient
     - Double(Vector, Double)
     - |uncheck|
   * - vector_function
     - The function to use for an mfem::VectorFunctionCoefficient
     - Vector(Vector, Double)
     - |uncheck|

---------------
vector_constant
---------------

Description: The constant vector to use as the coefficient

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - x
     - x-component of vector
     - 
     - 
     - |uncheck|
   * - z
     - z-component of vector
     - 
     - 
     - |uncheck|
   * - y
     - y-component of vector
     - 
     - 
     - |uncheck|

--------------
boundary_conds
--------------


--------------------
Collection contents:
--------------------


--------
dynamics
--------

Description: Parameters for mass matrix inversion

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - enforcement_method
     - Time-varying constraint enforcement method to use
     - 
     - 
     - |uncheck|
   * - timestepper
     - Timestepper (ODE) method to use
     - 
     - 
     - |uncheck|

---------
materials
---------


--------------------
Collection contents:
--------------------


--------------------
initial_displacement
--------------------

Description: Coefficient for initial condition

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - constant
     - The constant scalar value to use as the coefficient
     - 
     - 
     - |uncheck|
   * - component
     - The vector component to which the scalar coefficient should be applied
     - 
     - 
     - |uncheck|
.. list-table:: Functions
   :widths: 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Function Name
     - Description
     - Signature
     - Required
   * - scalar_function
     - The function to use for an mfem::FunctionCoefficient
     - Double(Vector, Double)
     - |uncheck|
   * - vector_function
     - The function to use for an mfem::VectorFunctionCoefficient
     - Vector(Vector, Double)
     - |uncheck|

---------------
vector_constant
---------------

Description: The constant vector to use as the coefficient

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - z
     - z-component of vector
     - 
     - 
     - |uncheck|
   * - x
     - x-component of vector
     - 
     - 
     - |uncheck|
   * - y
     - y-component of vector
     - 
     - 
     - |uncheck|

---------------
equation_solver
---------------

Description: Linear and Nonlinear stiffness Solver Parameters.


---------
nonlinear
---------

Description: Newton Equation Solver Parameters

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - solver_type
     - Solver type (Newton|KINFullStep|KINLineSearch)
     - Newton
     - 
     - |uncheck|
   * - print_level
     - Nonlinear print level.
     - 0
     - 
     - |uncheck|
   * - max_iter
     - Maximum iterations for the Newton solve.
     - 500
     - 
     - |uncheck|
   * - rel_tol
     - Relative tolerance for the Newton solve.
     - 0.010000
     - 
     - |uncheck|
   * - abs_tol
     - Absolute tolerance for the Newton solve.
     - 0.000100
     - 
     - |uncheck|

------
linear
------

Description: Linear Equation Solver Parameters

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - type
     - The type of solver parameters to use (iterative|direct)
     - 
     - iterative, direct
     - |check|

--------------
direct_options
--------------

Description: Direct solver parameters

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - print_level
     - Linear print level.
     - 0
     - 
     - |uncheck|

-----------------
iterative_options
-----------------

Description: Iterative solver parameters

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - prec_type
     - Preconditioner type (JacobiSmoother|L1JacobiSmoother|AMG|ILU).
     - JacobiSmoother
     - 
     - |uncheck|
   * - solver_type
     - Solver type (gmres|minres|cg).
     - gmres
     - 
     - |uncheck|
   * - print_level
     - Linear print level.
     - 0
     - 
     - |uncheck|
   * - rel_tol
     - Relative tolerance for the linear solve.
     - 0.000001
     - 
     - |uncheck|
   * - max_iter
     - Maximum iterations for the linear solve.
     - 5000
     - 
     - |uncheck|
   * - abs_tol
     - Absolute tolerance for the linear solve.
     - 0.000000
     - 
     - |uncheck|

------------------
heat_transfer
------------------

Description: Heat transfer module

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - order
     - Order degree of the finite elements.
     - 1
     - 1 to 8
     - |uncheck|

--------------
boundary_conds
--------------


--------------------
Collection contents:
--------------------


--------
dynamics
--------

Description: Parameters for mass matrix inversion

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - enforcement_method
     - Time-varying constraint enforcement method to use
     - 
     - 
     - |uncheck|
   * - timestepper
     - Timestepper (ODE) method to use
     - 
     - 
     - |uncheck|

---------------
equation_solver
---------------

Description: Linear and Nonlinear stiffness Solver Parameters.


---------
nonlinear
---------

Description: Newton Equation Solver Parameters

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - solver_type
     - Solver type (Newton|KINFullStep|KINLineSearch)
     - Newton
     - 
     - |uncheck|
   * - max_iter
     - Maximum iterations for the Newton solve.
     - 500
     - 
     - |uncheck|
   * - rel_tol
     - Relative tolerance for the Newton solve.
     - 0.010000
     - 
     - |uncheck|
   * - print_level
     - Nonlinear print level.
     - 0
     - 
     - |uncheck|
   * - abs_tol
     - Absolute tolerance for the Newton solve.
     - 0.000100
     - 
     - |uncheck|

------
linear
------

Description: Linear Equation Solver Parameters

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - type
     - The type of solver parameters to use (iterative|direct)
     - 
     - iterative, direct
     - |check|

--------------
direct_options
--------------

Description: Direct solver parameters

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - print_level
     - Linear print level.
     - 0
     - 
     - |uncheck|

-----------------
iterative_options
-----------------

Description: Iterative solver parameters

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - solver_type
     - Solver type (gmres|minres|cg).
     - gmres
     - 
     - |uncheck|
   * - max_iter
     - Maximum iterations for the linear solve.
     - 5000
     - 
     - |uncheck|
   * - prec_type
     - Preconditioner type (JacobiSmoother|L1JacobiSmoother|AMG|ILU).
     - JacobiSmoother
     - 
     - |uncheck|
   * - rel_tol
     - Relative tolerance for the linear solve.
     - 0.000001
     - 
     - |uncheck|
   * - print_level
     - Linear print level.
     - 0
     - 
     - |uncheck|
   * - abs_tol
     - Absolute tolerance for the linear solve.
     - 0.000000
     - 
     - |uncheck|

-------------------
initial_temperature
-------------------

Description: Coefficient for initial condition

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - constant
     - The constant scalar value to use as the coefficient
     - 
     - 
     - |uncheck|
   * - component
     - The vector component to which the scalar coefficient should be applied
     - 
     - 
     - |uncheck|
.. list-table:: Functions
   :widths: 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Function Name
     - Description
     - Signature
     - Required
   * - scalar_function
     - The function to use for an mfem::FunctionCoefficient
     - Double(Vector, Double)
     - |uncheck|
   * - vector_function
     - The function to use for an mfem::VectorFunctionCoefficient
     - Vector(Vector, Double)
     - |uncheck|

---------------
vector_constant
---------------

Description: The constant vector to use as the coefficient

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - z
     - z-component of vector
     - 
     - 
     - |uncheck|
   * - x
     - x-component of vector
     - 
     - 
     - |uncheck|
   * - y
     - y-component of vector
     - 
     - 
     - |uncheck|

------
source
------

Description: Scalar source term (RHS of the heat transfer PDE)

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - constant
     - The constant scalar value to use as the coefficient
     - 
     - 
     - |uncheck|
   * - component
     - The vector component to which the scalar coefficient should be applied
     - 
     - 
     - |uncheck|
.. list-table:: Functions
   :widths: 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Function Name
     - Description
     - Signature
     - Required
   * - scalar_function
     - The function to use for an mfem::FunctionCoefficient
     - Double(Vector, Double)
     - |uncheck|
   * - vector_function
     - The function to use for an mfem::VectorFunctionCoefficient
     - Vector(Vector, Double)
     - |uncheck|

---------------
vector_constant
---------------

Description: The constant vector to use as the coefficient

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - z
     - z-component of vector
     - 
     - 
     - |uncheck|
   * - x
     - x-component of vector
     - 
     - 
     - |uncheck|
   * - y
     - y-component of vector
     - 
     - 
     - |uncheck|

------------------
heat_transfer
------------------

Description: Heat transfer module

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - order
     - Order degree of the finite elements.
     - 1
     - 1 to 8
     - |uncheck|

-------------------
initial_temperature
-------------------

Description: Coefficient for initial condition

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - constant
     - The constant scalar value to use as the coefficient
     - 
     - 
     - |uncheck|
   * - component
     - The vector component to which the scalar coefficient should be applied
     - 
     - 
     - |uncheck|
.. list-table:: Functions
   :widths: 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Function Name
     - Description
     - Signature
     - Required
   * - scalar_function
     - The function to use for an mfem::FunctionCoefficient
     - Double(Vector, Double)
     - |uncheck|
   * - vector_function
     - The function to use for an mfem::VectorFunctionCoefficient
     - Vector(Vector, Double)
     - |uncheck|

---------------
vector_constant
---------------

Description: The constant vector to use as the coefficient

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - z
     - z-component of vector
     - 
     - 
     - |uncheck|
   * - x
     - x-component of vector
     - 
     - 
     - |uncheck|
   * - y
     - y-component of vector
     - 
     - 
     - |uncheck|

--------------
boundary_conds
--------------


--------------------
Collection contents:
--------------------

The input schema defines a collection of this container.
For brevity, only one instance is displayed here.

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - constant
     - The constant scalar value to use as the coefficient
     - 
     - 
     - |uncheck|
   * - component
     - The vector component to which the scalar coefficient should be applied
     - 
     - 
     - |uncheck|
.. list-table:: Functions
   :widths: 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Function Name
     - Description
     - Signature
     - Required
   * - scalar_function
     - The function to use for an mfem::FunctionCoefficient
     - Double(Vector, Double)
     - |uncheck|
   * - vector_function
     - The function to use for an mfem::VectorFunctionCoefficient
     - Vector(Vector, Double)
     - |uncheck|

-----
attrs
-----


--------------------
Collection contents:
--------------------

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - 1
     - 
     - 
     - 
     - |uncheck|

---------------
vector_constant
---------------

Description: The constant vector to use as the coefficient

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - x
     - x-component of vector
     - 
     - 
     - |uncheck|
   * - z
     - z-component of vector
     - 
     - 
     - |uncheck|
   * - y
     - y-component of vector
     - 
     - 
     - |uncheck|

---------------
equation_solver
---------------

Description: Linear and Nonlinear stiffness Solver Parameters.


---------
nonlinear
---------

Description: Newton Equation Solver Parameters

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - solver_type
     - Solver type (Newton|KINFullStep|KINLineSearch)
     - Newton
     - 
     - |uncheck|
   * - print_level
     - Nonlinear print level.
     - 0
     - 
     - |uncheck|
   * - max_iter
     - Maximum iterations for the Newton solve.
     - 500
     - 
     - |uncheck|
   * - rel_tol
     - Relative tolerance for the Newton solve.
     - 0.010000
     - 
     - |uncheck|
   * - abs_tol
     - Absolute tolerance for the Newton solve.
     - 0.000100
     - 
     - |uncheck|

------
linear
------

Description: Linear Equation Solver Parameters

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - type
     - The type of solver parameters to use (iterative|direct)
     - 
     - iterative, direct
     - |check|

--------------
direct_options
--------------

Description: Direct solver parameters

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - print_level
     - Linear print level.
     - 0
     - 
     - |uncheck|

-----------------
iterative_options
-----------------

Description: Iterative solver parameters

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - prec_type
     - Preconditioner type (JacobiSmoother|L1JacobiSmoother|AMG|ILU).
     - JacobiSmoother
     - 
     - |uncheck|
   * - solver_type
     - Solver type (gmres|minres|cg).
     - gmres
     - 
     - |uncheck|
   * - print_level
     - Linear print level.
     - 0
     - 
     - |uncheck|
   * - max_iter
     - Maximum iterations for the linear solve.
     - 5000
     - 
     - |uncheck|
   * - rel_tol
     - Relative tolerance for the linear solve.
     - 0.000001
     - 
     - |uncheck|
   * - abs_tol
     - Absolute tolerance for the linear solve.
     - 0.000000
     - 
     - |uncheck|

--------
dynamics
--------

Description: Parameters for mass matrix inversion

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - enforcement_method
     - Time-varying constraint enforcement method to use
     - 
     - 
     - |uncheck|
   * - timestepper
     - Timestepper (ODE) method to use
     - 
     - 
     - |uncheck|

------
source
------

Description: Scalar source term (RHS of the heat transfer PDE)

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - constant
     - The constant scalar value to use as the coefficient
     - 
     - 
     - |uncheck|
   * - component
     - The vector component to which the scalar coefficient should be applied
     - 
     - 
     - |uncheck|
.. list-table:: Functions
   :widths: 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Function Name
     - Description
     - Signature
     - Required
   * - scalar_function
     - The function to use for an mfem::FunctionCoefficient
     - Double(Vector, Double)
     - |uncheck|
   * - vector_function
     - The function to use for an mfem::VectorFunctionCoefficient
     - Vector(Vector, Double)
     - |uncheck|

---------------
vector_constant
---------------

Description: The constant vector to use as the coefficient

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - z
     - z-component of vector
     - 
     - 
     - |uncheck|
   * - x
     - x-component of vector
     - 
     - 
     - |uncheck|
   * - y
     - y-component of vector
     - 
     - 
     - |uncheck|

---------
main_mesh
---------

Description: The main mesh for the problem

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - mesh
     - Path to Mesh file
     - 
     - 
     - |uncheck|
   * - type
     - Type of mesh
     - 
     - ball, box, disk, file
     - |check|
   * - approx_elements
     - Approximate number of elements in an n-ball mesh
     - 
     - 
     - |uncheck|
   * - ser_ref_levels
     - Number of times to refine the mesh uniformly in serial.
     - 0
     - 
     - |uncheck|
   * - par_ref_levels
     - Number of times to refine the mesh uniformly in parallel.
     - 0
     - 
     - |uncheck|

----
size
----

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - x
     - Size in the x-dimension
     - 
     - 
     - |uncheck|
   * - z
     - Size in the z-dimension
     - 
     - 
     - |uncheck|
   * - y
     - Size in the y-dimension
     - 
     - 
     - |uncheck|

--------
elements
--------

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - z
     - z-dimension
     - 
     - 
     - |uncheck|
   * - x
     - x-dimension
     - 
     - 
     - |uncheck|
   * - y
     - y-dimension
     - 
     - 
     - |uncheck|

-----
solid
-----

Description: Finite deformation solid mechanics module

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - order
     - polynomial order of the basis functions.
     - 1
     - 1 to 3
     - |uncheck|

--------------
boundary_conds
--------------


--------------------
Collection contents:
--------------------

The input schema defines a collection of this container.
For brevity, only one instance is displayed here.

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - constant
     - The constant scalar value to use as the coefficient
     - 
     - 
     - |uncheck|
   * - component
     - The vector component to which the scalar coefficient should be applied
     - 
     - 
     - |uncheck|
.. list-table:: Functions
   :widths: 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Function Name
     - Description
     - Signature
     - Required
   * - scalar_function
     - The function to use for an mfem::FunctionCoefficient
     - Double(Vector, Double)
     - |uncheck|
   * - vector_function
     - The function to use for an mfem::VectorFunctionCoefficient
     - Vector(Vector, Double)
     - |uncheck|

-----
attrs
-----


--------------------
Collection contents:
--------------------

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - 1
     - 
     - 
     - 
     - |uncheck|

---------------
vector_constant
---------------

Description: The constant vector to use as the coefficient

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - z
     - z-component of vector
     - 
     - 
     - |uncheck|
   * - x
     - x-component of vector
     - 
     - 
     - |uncheck|
   * - y
     - y-component of vector
     - 
     - 
     - |uncheck|

--------------------
initial_displacement
--------------------

Description: Coefficient for initial condition

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - constant
     - The constant scalar value to use as the coefficient
     - 
     - 
     - |uncheck|
   * - component
     - The vector component to which the scalar coefficient should be applied
     - 
     - 
     - |uncheck|
.. list-table:: Functions
   :widths: 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Function Name
     - Description
     - Signature
     - Required
   * - scalar_function
     - The function to use for an mfem::FunctionCoefficient
     - Double(Vector, Double)
     - |uncheck|
   * - vector_function
     - The function to use for an mfem::VectorFunctionCoefficient
     - Vector(Vector, Double)
     - |uncheck|

---------------
vector_constant
---------------

Description: The constant vector to use as the coefficient

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - z
     - z-component of vector
     - 
     - 
     - |uncheck|
   * - x
     - x-component of vector
     - 
     - 
     - |uncheck|
   * - y
     - y-component of vector
     - 
     - 
     - |uncheck|

--------
dynamics
--------

Description: Parameters for mass matrix inversion

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - enforcement_method
     - Time-varying constraint enforcement method to use
     - 
     - 
     - |uncheck|
   * - timestepper
     - Timestepper (ODE) method to use
     - 
     - 
     - |uncheck|

---------
materials
---------


--------------------
Collection contents:
--------------------

The input schema defines a collection of this container.
For brevity, only one instance is displayed here.

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - sigma_y
     - Yield stress
     - 
     - 
     - |uncheck|
   * - Hi
     - Isotropic hardening constant
     - 
     - 
     - |uncheck|
   * - E
     - Young's modulus
     - 
     - 
     - |uncheck|
   * - nu
     - Poisson's ratio
     - 
     - 
     - |uncheck|
   * - Hk
     - Kinematic hardening constant
     - 
     - 
     - |uncheck|
   * - mu
     - The shear modulus
     - 
     - 
     - |uncheck|
   * - density
     - Initial mass density
     - 
     - 
     - |uncheck|
   * - model
     - The model of material (e.g. NeoHookean)
     - 
     - 
     - |check|
   * - K
     - The bulk modulus
     - 
     - 
     - |uncheck|

---------
hardening
---------

Description: Hardening law

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - strain_constant
     - Constant dictating how fast the exponential decays
     - 
     - 
     - |uncheck|
   * - sigma_sat
     - Saturation value of flow strength
     - 
     - 
     - |uncheck|
   * - eps0
     - Reference value of accumulated plastic strain
     - 
     - 
     - |uncheck|
   * - n
     - Hardening index in reciprocal form
     - 
     - 
     - |uncheck|
   * - law
     - Name of the hardening law (e.g. PowerLawHardening, VoceHardening)
     - 
     - 
     - |check|
   * - sigma_y
     - Yield strength
     - 
     - 
     - |uncheck|

----------------
initial_velocity
----------------

Description: Coefficient for initial condition

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - constant
     - The constant scalar value to use as the coefficient
     - 
     - 
     - |uncheck|
   * - component
     - The vector component to which the scalar coefficient should be applied
     - 
     - 
     - |uncheck|
.. list-table:: Functions
   :widths: 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Function Name
     - Description
     - Signature
     - Required
   * - scalar_function
     - The function to use for an mfem::FunctionCoefficient
     - Double(Vector, Double)
     - |uncheck|
   * - vector_function
     - The function to use for an mfem::VectorFunctionCoefficient
     - Vector(Vector, Double)
     - |uncheck|

---------------
vector_constant
---------------

Description: The constant vector to use as the coefficient

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - z
     - z-component of vector
     - 
     - 
     - |uncheck|
   * - x
     - x-component of vector
     - 
     - 
     - |uncheck|
   * - y
     - y-component of vector
     - 
     - 
     - |uncheck|

---------------
equation_solver
---------------

Description: Linear and Nonlinear stiffness Solver Parameters.


---------
nonlinear
---------

Description: Newton Equation Solver Parameters

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - solver_type
     - Solver type (Newton|KINFullStep|KINLineSearch)
     - Newton
     - 
     - |uncheck|
   * - max_iter
     - Maximum iterations for the Newton solve.
     - 500
     - 
     - |uncheck|
   * - rel_tol
     - Relative tolerance for the Newton solve.
     - 0.010000
     - 
     - |uncheck|
   * - print_level
     - Nonlinear print level.
     - 0
     - 
     - |uncheck|
   * - abs_tol
     - Absolute tolerance for the Newton solve.
     - 0.000100
     - 
     - |uncheck|

------
linear
------

Description: Linear Equation Solver Parameters

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - type
     - The type of solver parameters to use (iterative|direct)
     - 
     - iterative, direct
     - |check|

--------------
direct_options
--------------

Description: Direct solver parameters

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - print_level
     - Linear print level.
     - 0
     - 
     - |uncheck|

-----------------
iterative_options
-----------------

Description: Iterative solver parameters

.. list-table:: Fields
   :widths: 25 25 25 25 25
   :header-rows: 1
   :stub-columns: 1

   * - Field Name
     - Description
     - Default Value
     - Range/Valid Values
     - Required
   * - solver_type
     - Solver type (gmres|minres|cg).
     - gmres
     - 
     - |uncheck|
   * - print_level
     - Linear print level.
     - 0
     - 
     - |uncheck|
   * - max_iter
     - Maximum iterations for the linear solve.
     - 5000
     - 
     - |uncheck|
   * - rel_tol
     - Relative tolerance for the linear solve.
     - 0.000001
     - 
     - |uncheck|
   * - prec_type
     - Preconditioner type (JacobiSmoother|L1JacobiSmoother|AMG|ILU).
     - JacobiSmoother
     - 
     - |uncheck|
   * - abs_tol
     - Absolute tolerance for the linear solve.
     - 0.000000
     - 
     - |uncheck|
