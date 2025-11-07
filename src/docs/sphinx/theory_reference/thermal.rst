.. ## Copyright (c) Lawrence Livermore National Security, LLC and
.. ## other Smith Project Developers. See the top-level COPYRIGHT file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)

.. _conduction-theory-label:

==================
Heat Transfer
==================

Strong Form
===========

The heat transfer module solves the heat equation

.. math:: c_p \rho\frac{\partial T}{\partial t} - \nabla \cdot (\kappa \nabla T ) = g(x, t)

subject to the boundary conditions

.. math::

   \begin{align*}
   T(x,0) &= T_0(x)  \\
   T(x,t) &= T_D(x,t) & \text{on } \Gamma_D \\
   \kappa \frac{\partial T}{\partial n} &= q(x,t) & \text{on } \Gamma_N
   \end{align*}

where

.. math::

   \begin{align*}
   T(x,t) & =\text{ temperature} \\
   c_p & =\text{ specific heat capacity} \\
   \rho & =\text{ density} \\
   \kappa & =\text { conductivity} \\
   g(x,t) & =\text{ heat source} \\
   T_0(x) & =\text{ initial temperature} \\
   T_D(x,t) & =\text { fixed boundary temperature} \\
   q(x,t) & = \text { fixed boundary heat flux.}
   \end{align*}

Weak Form
=========

We multiply this strong form of the PDE by an arbitrary function and integrate by
parts to obtain the weak form

.. math::

   \begin{align*}
   &\text{Find } T \in V \text{ such that}\\
   &\int_\Omega \left( \left(c_p \rho\frac{\partial T}{\partial t} - g(x, t) \right) v + \kappa \nabla T \cdot \nabla v \right) dV - \int_{\Gamma_N} q v\, dA = 0, & & \forall v \in \hat V
   \end{align*}

where

.. math::

   \begin{align*}
   V &= \left\{ v \in H_1(\Omega):v=T_D \text{ on } \Gamma_D \right\} \\
   \hat{V} &= \left\{v \in H_1(\Omega):v=0 \text{ on } \Gamma_D \right\}.
   \end{align*}

Discretization
==============

After discretizing by the standard continuous Galerkin finite element
method, i.e.

.. math::

   \begin{align*}
   T(x,t) = \sum_{i=1}^N \phi_i(x) u_i(t), & & v &= \phi_j(x)
   \end{align*}

where :math:`\phi_i` are nodal shape functions and :math:`u_i` are
degrees of freedom, we obtain the discrete equations in residual form

.. math:: \mathbf{M} \dot{\mathbf{u}} +\mathbf{Ku} - \mathbf{G} = \mathbf{0}

where

.. math::

   \begin{align*}
   \mathbf{u} &= \text{degree of freedom vector (unknowns)} \\
   \mathbf{M} &= \text{thermal mass (heat capacity or capacitance) matrix} \\
   \mathbf{K} &= \text{thamal stiffness (thermal conductivity or conductance) matrix} \\
   \mathbf{G} &= \text{source vector}. \\
   \end{align*}

This system can then be solved using the selected nonlinear and ordinary
differential equation solution methods. For example, if we use the
backward Euler method we obtain

.. math:: \mathbf{Mu}_{n+1} + \Delta t \mathbf{Ku}_{n+1} = \Delta t \mathbf{G} + \mathbf{Mu}_n.

Given a known :math:`\mathbf{u}_n`, this is solved at each timestep
:math:`\Delta t` for :math:`\mathbf{u}_{n+1}` using a nonlinear solver,
typically Newton's method. To accomplish this, the above
equation is linearized which yields

.. math:: \left(\mathbf{M} + \Delta t \mathbf{K} \right)\Delta \mathbf{u}^{i+1}_{n+1} = -(\mathbf{M} + \Delta t \mathbf{K}) \mathbf{u}_{n+1}^i + \Delta t \mathbf{G} + \mathbf{Mu}_n.

Note the above equation assumes the thermal capacitance and conductance are independent of temperature.
The computed Newton iterations
:math:`\mathbf{u}_{n+1}^{i+1} = \mathbf{u}_{n+1}^{i} + \Delta \mathbf{u}_{n+1}^{i+1}`
continue until the solution is converged.
