.. ## Copyright (c) Lawrence Livermore National Security, LLC and
.. ## other Smith Project Developers. See the top-level COPYRIGHT file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)

.. _equationsolver-label:

==============
EquationSolver
==============

Mathematical description
========================

`EquationSolver <../../doxygen/html/classsmith_1_1EquationSolver.html>`__ is an object used for solving nonlinear systems of equations of the form

.. math::

  F(\mathbf{X}) = \mathbf{0}

where :math:`\mathbf{X}` is a parallel distributed vector, e.g. ``mfem::HypreParVector``, and
:math:`F` is a square nonlinear operator, i.e. the dimension of the input vector equals the dimension of the output vector.  
These systems commonly arise from finite element discretizations such as the equations of :ref:`heat transfer <conduction-theory-label>`
  
.. math:: \mathbf{Mu}_{n+1} + \Delta t (\mathbf{Ku}_{n+1} + f(\mathbf{u}_{n+1})) - \Delta t \mathbf{G} - \mathbf{Mu}_n = \mathbf{0}

where :math:`\mathbf{X} = \mathbf{u}_{n+1}`. These systems are commonly solved by Newton-type methods which have embedded linear solvers for a linearized approximation
of the full nonlinear operator. A simple Newton algorithm for solving this system is given below:

.. code-block:: python

  Pick an initial guess X = X_0
  Compute the residual r = F(X)
  while (r is too large) {
    Compute the linearization (Jacobian) of F, J = dF/dX
    Solve J X_update = -r using a linear solver
    X = X + X_update
    Compute the updated residual r = F(X)
  }

To perform this solve, we typically need to configure both the nonlinear solver algorithm and the linear solver algorithm. If we want to use an iterative solver
for the linear part, we also may need to configure a preconditioner for the system. 

Class design
============

`EquationSolver <../../doxygen/html/classsmith_1_1EquationSolver.html>`__ provides an interface to the associated nonlinear and linear solver
algorithms needed to solve these systems of equations. Note that while some nonlinear solvers do not depend on an embedded linear solver (e.g. L-BFGS), we require a linear 
solver to be specified as it is used to compute reasonable initial guesses and perform adjoint solves within Smith. 

The key methods provided by this class are:

1.  ``void SetOperator(const mfem::Operator& op)``: This defines the nonlinear ``mfem::Operator`` representing the vector-valued nonlinear system of equations :math:`F`.
2.  ``void Solve(mfem::Vector& x)``: This solves the nonlinear system :math:`F(\mathbf{X}) = \mathbf{0}` and stores the solution in-place in ``x``.
3.  ``mfem::Solver& LinearSolver()``: This returns the associated linear solver for adjoint solves and initial guess calculations.

Two workflows exist for defining the linear and nonlinear algorithms in a ``EquationSolver``: predefined option structs for common use cases and fully custom ``mfem::Solvers``
for advanced users.

Common configurations via option structs
----------------------------------------

The easist way to build an equation solver is by providing structs containing parameters for common configurations of the linear and nonlinear solvers. They can also
be passed directly to physics module constructors.

.. literalinclude:: ../../../../src/smith/numerics/equation_solver.hpp
   :start-after: _build_equationsolver_start
   :end-before: _build_equationsolver_end
   :language: C++

The nonlinear solver configuration options are provided by the ``NonlinearSolverOptions`` struct:

.. literalinclude:: ../../../../src/smith/numerics/solver_config.hpp
   :start-after: _nonlinear_options_start
   :end-before: _nonlinear_options_end
   :language: C++

The current possible nonlinear solution algorithms are:

.. literalinclude:: ../../../../src/smith/numerics/solver_config.hpp
   :start-after: _nonlinear_solvers_start
   :end-before: _nonlinear_solvers_end
   :language: C++

The linear solver configuration options are provided by the ``LinearSolverOptions`` struct:

.. literalinclude:: ../../../../src/smith/numerics/solver_config.hpp
   :start-after: _linear_options_start
   :end-before: _linear_options_end
   :language: C++

The current possible linear solution algorithms are:

.. literalinclude:: ../../../../src/smith/numerics/solver_config.hpp
   :start-after: _linear_solvers_start
   :end-before: _linear_solvers_end
   :language: C++

The current possible preconditioners for iterative linear solvers are:

.. literalinclude:: ../../../../src/smith/numerics/solver_config.hpp
   :start-after: _preconditioners_start
   :end-before: _preconditioners_end
   :language: C++

Custom configuration via pointers
---------------------------------

If the nonlinear and linear solvers provided by the above options are not sufficient for your application, custom solvers can be written for both the nonlinear
and linear solver objects. For this approach, the direct constructor for ``EquationSolver`` is used:

.. literalinclude:: ../../../../src/smith/numerics/equation_solver.hpp
   :start-after: _equationsolver_constructor_start
   :end-before: _equationsolver_constructor_end
   :language: C++

The nonlinear and linear solvers are required while the preconditioner is optional.

Although the nonlinear solver is expected to be of type ``mfem::NewtonSolver``, it does not have to be a Newton-type method. This 
is simply the preferred MFEM container for nonlinear solvers. For example, the included L-BFGS solver is derived from this class. This class is preferred over type ``mfem::Solver``
as it has the appropriate methods for checking convergence as needed.

Use within physics modules
==========================

An example of configuring a ``SolidMechanics`` simulation module via options stucts is below:

.. literalinclude:: ../../../../src/smith/physics/tests/solid.cpp
   :start-after: _solver_params_start
   :end-before: _solver_params_end
   :language: C++

Alternatively, you can build an ``EquationSolver`` using custom nonlinear and linear solvers if it is required
by your application. An example of a parameterized ``SolidMechanics`` module that uses a custom ``EquationSolver`` 
is below:

.. literalinclude:: ../../../../src/smith/physics/tests/solid.cpp
   :start-after: _custom_solver_start
   :end-before: _custom_solver_end
   :language: C++

.. warning:: Each physics module must have its own ``EquationSolver``. They cannot be reused between modules. 
