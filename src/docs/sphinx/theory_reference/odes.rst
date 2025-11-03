.. ## Copyright (c) Lawrence Livermore National Security, LLC and
.. ## other Smith Project Developers. See the top-level COPYRIGHT file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)

=========================================================
Dirichlet Enforcement for Ordinary Differential Equations
=========================================================

Unconstrained Ordinary Differential Equations
=============================================

Consider a system described by then following linear, second order
differential equation:

.. math::

   \begin{gather}
   \textbf{M} \; \ddot{\textbf{x}}(t) + \textbf{C} \; \dot{\textbf{x}}(t) + \textbf{K} \; \textbf{x}(t)  = \textbf{0} \\
   \textbf{x}(0) = \textbf{x}_0 \\
   \dot{\textbf{x}}(0) = \dot{\textbf{x}}_0
   \end{gather}

In order to numerically integrate this differential equation, we can
start by expressing the the future state as an extrapolation of the
current state (e.g. using backward Euler):

.. math::

   \begin{gathered}
   \begin{cases}
   \textbf{x}(t + \Delta t) = \textbf{x}(t) + \dot{\textbf{x}}(t) \Delta t + \ddot{\textbf{x}}(t + \Delta t) \Delta t^2 \\
   \dot{\textbf{x}}(t + \Delta t) = \dot{\textbf{x}}(t) + \ddot{\textbf{x}}(t + \Delta t) \Delta t
   \end{cases}
   \end{gathered}

Substituting this assumption back in to the differential equation
evaluated at time :math:`t + \Delta t` 
produces an expression that relates a future state:
:math:`\{\textbf{x}(t + \Delta t), \dot{\textbf{x}}(t + \Delta t)\}` to the
current state :math:`\{\textbf{x}(t), \dot{\textbf{x}}(t)\}`:

.. math:: \textbf{M} \; \ddot{\textbf{x}}(t+\Delta t) + \textbf{C} \, \big(\dot{\textbf{x}}(t) + \ddot{\textbf{x}}(t + \Delta t) \Delta t\big) + \textbf{K} \; \big(\textbf{x}(t) + \dot{\textbf{x}}(t) \Delta t + \ddot{\textbf{x}}(t + \Delta t) \Delta t^2\big)  = \textbf{0}

After rearranging terms to put the known quantities on the righthand
side, we are left with

.. math:: \big(\textbf{M} + \textbf{C} \Delta t + \textbf{K} \Delta t^2\big) \; \ddot{\textbf{x}}(t+\Delta t) = - \textbf{C} \; \dot{\textbf{x}}(t) - \textbf{K} \big(\textbf{x}(t) + \dot{\textbf{x}}(t) \Delta t\big)

From here, provided that the coefficient matrix is invertible, a system
of equations can be solved to determine
:math:`\ddot{\textbf{x}}(t+\Delta t)` and advance the solution state to
time :math:`t + \Delta t`.

.. _header-n42:

Imposing Single Degree-of-Freedom Constraints
=============================================

Now, let us consider how the solution process changes if some components
of :math:`\textbf{x}(t)` are controlled externally. In this case, our differential
equation looks like

.. math::

   \begin{gather}
   \textbf{M} \; \ddot{\textbf{x}}(t) + \textbf{C} \; \dot{\textbf{x}}(t) + \textbf{K} \; \textbf{x}(t)  = \textbf{0} \\
   \color{red}{\textbf{G} \; \textbf{x}(t) = \textbf{g}(t)} \\
   \textbf{x}(0) = \textbf{x}_0 \\
   \dot{\textbf{x}}(0) = \dot{\textbf{x}}_0
   \end{gather}

where :math:`\textbf{G}` is a matrix with 1 on the diagonals corresponding
to constrained components, and 0 elsewhere, and :math:`\textbf{g}(t)` is
the vector of prescribed values for the constrained degrees of freedom.
There are several equivalent ways to interpret the
additional constraint equations (we assume the constraints are
consistent with the initial conditions):

.. math::

   \textbf{G} \; \textbf{x}(t) = \textbf{g}(t) \; \iff \;
   \textbf{G} \; \dot{\textbf{x}}(t) = \dot{\textbf{g}}(t) \; \iff \;
   \textbf{G} \; \ddot{\textbf{x}}(t) = \ddot{\textbf{g}}(t)

At the end of the day, we are solving for a single acceleration vector,
:math:`\ddot{\textbf{x}}(t+\Delta t)`, and as a result, we cannot hope to
simultaneously satisfy more than one interpretation of these
constraints. The different Dirichlet Enforcement Methods in Smith relate
to the following different interpretations of the constraint equations.

.. _header-n97:

"Direct Control"
----------------

Direct control proceeds to solve for accelerations by effectively
replacing the constrained rows of the differential equation by the
constraint equations,

.. math:: (\textbf{1} - \textbf{G})(\textbf{M} \; \ddot{\textbf{x}}(t+\Delta t) + \textbf{C} \; \dot{\textbf{x}}(t+\Delta t) + \textbf{K} \; \textbf{x}(t+\Delta t)) + \textbf{G} \; \textbf{x}(t+\Delta t)\;=\;\textbf{g}(t+\Delta t)

Now, like before, we substitute in expressions for extrapolated state
and solve for :math:`\ddot{\textbf{x}}(t+\Delta t)`. In this
interpretation of the constraint, the resulting accelerations ensure
that :math:`\textbf{G} \; \textbf{x}(t+\Delta t) = \textbf{g}(t+\Delta t)` is
satisfied.

.. _header-n79:

"Rate Control"
--------------

Rate control uses a similar idea, but with the rate-form interpretation
of the constraint:

.. math:: (\textbf{1} - \textbf{G})(\textbf{M} \; \ddot{\textbf{x}}(t+\Delta t) + \textbf{C} \; \dot{\textbf{x}}(t+\Delta t) + \textbf{K} \; \textbf{x}(t+\Delta t)) + \textbf{G} \; \dot{\textbf{x}}(t+\Delta t)\;=\;\dot{\textbf{g}}(t+\Delta t)

Here, the end-step accelerations are determined such that the derivative
of the constraint equations is exactly satisfied,
:math:`\textbf{G} \; \dot{\textbf{x}}(t+\Delta t) = \dot{\textbf{g}}(t+\Delta t)`.
This tends to have better stability than the "Direct Control" option,
but is prone to solution "drift" over long periods of time.

.. _header-n94:

"Full Control"
--------------

This last option starts by additively decomposing the solution vector
into two parts: constrained and unconstrained:

.. math::

   \begin{cases}
   \textbf{x}(t) = \textbf{x}_c(t) + \textbf{x}_u(t) = \textbf{G} \; \textbf{x}(t) + (\textbf{1} - \textbf{G}) \; \textbf{x}(t) \\
   \dot{\textbf{x}}(t) = \dot{\textbf{x}}_c(t) + \dot{\textbf{x}}_u(t) = \textbf{G} \; \dot{\textbf{x}}(t) + (\textbf{1} - \textbf{G}) \; \dot{\textbf{x}}(t)\\
   \ddot{\textbf{x}}(t) = \ddot{\textbf{x}}_c(t) + \ddot{\textbf{x}}_u(t) = \textbf{G} \; \ddot{\textbf{x}}(t) + (\textbf{1} - \textbf{G}) \; \ddot{\textbf{x}}(t)
   \end{cases}

From here, the constrained terms
:math:`\{\textbf{x}_c(t), \dot{\textbf{x}}_c(t), \ddot{\textbf{x}}_c(t)\}` are
replaced by their prescribed values,
:math:`\{\textbf{g}(t), \dot{\textbf{g}}(t), \ddot{\textbf{g}}(t)\}` and only
the unconstrained acceleration components,
:math:`\ddot{\textbf{x}}_u(t+\Delta t)`, are solved for. For situations
where :math:`\textbf{g}(t)` has a discontinuous derivative, this approach
may be preferable to Rate Control.

Note: time derivatives of :math:`\textbf{g}(t)` are currently computed by
central finite difference stencils

.. math::

   \begin{gather}
   \dot{\textbf{g}}(t) \approx \frac{\textbf{g}(t + \epsilon) - \textbf{g}(t - \epsilon)}{2 \; \epsilon} \\
   \ddot{\textbf{g}}(t) \approx \frac{\textbf{g}(t + \epsilon) - 2 \, \textbf{g}(t) +\textbf{g}(t - \epsilon)}{\epsilon^2}
   \end{gather}
