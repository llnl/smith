.. ## Copyright (c) Lawrence Livermore National Security, LLC and
.. ## other Smith Project Developers. See the top-level COPYRIGHT file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)

.. _functional-label:

==========
Functional
==========

``Functional`` is a class that is used specify and evaluate
finite-element-type calculations. For example, the weighted residual for
a solid mechanics simulation may look something like:

.. math::

   r(u) := 
   \underbrace{\int_\Omega \sigma(\nabla u) \cdot \nabla\psi \; \text{d}v}_{\text{stress response}}
   \;+\;
   \underbrace{\int_\Omega b(\mathbf{x}) \; \psi \; \text{d}v}_{\text{body forces}} 
   \;+\;
   \underbrace{\int_{\partial\Omega} \mathbf{t}(\mathbf{x}) \; \psi \; \text{d}a}_{\text{surface loads}},

where :math:`\psi` are the test basis functions. To describe this
residual using ``Functional``, we first create the object itself, providing a
template parameter that expresses the test and trial spaces (i.e. the respective
"outputs" and "inputs" of the residual function, :math:`r`). In this
case, solid mechanics uses nodal displacements and residuals (i.e. H1 test and trial spaces), so we write:

.. code-block:: cpp

   constexpr int order = 2; // the polynomial order of the basis functions
   constexpr int dim = 3; // the number of components per node
   using test = H1<order, dim>;
   using trial = H1<order, dim>;
   Functional< test(trial) > residual(&test_fes, {&trial_fes});

where ``test_fes``, ``trial_fes`` are the ``mfem::FiniteElementSpaces`` for the problem. 
The template argument follows the same convention of ``std::function``:
the output-type appears outside the parentheses, and the input-type(s)
appear, inside the parentheses (in order).  So, the last line of code in
the snippet above is saying that ``residual`` is going to represent a
calculation that takes in an H1 field (displacements), and returns 
weighted residual vectors for each node, using H1 test functions.

Now that the ``Functional`` object is created, we can use the
following functions to define integral terms (depending on their
dimensionality). Here, we use :math:`s` to denote the "source" term
(integrated against test functions), and :math:`f` to denote the 
"flux" term (integrated against test function gradients).

1. Integrals of the form:
   :math:`\displaystyle \iint_\Omega \psi \cdot s + \nabla \psi : f \; da`

   .. code-block:: cpp

    residual.AddAreaIntegral(
      DependsOn< ... >{},
      [](auto x, auto ... args){
      	auto s = ...;
      	auto f = ...;
      	return smith::tuple{s, f};
      }, 
      domain_of_integration
    );

2. Integrals of the form:
   :math:`\displaystyle \iiint_\Omega \psi \cdot s + \nabla \psi : f \; dv`

   .. code-block:: cpp

    residual.AddVolumeIntegral(
      DependsOn< ... >{},
      [](auto x, auto ... args){
      	auto s = ...;
      	auto f = ...;
      	return smith::tuple{s, f};
      }, 
      domain_of_integration
    );

3. Integrals of the form:
   :math:`\displaystyle \iint_{\partial \Omega} \psi \cdot s \; da`

   .. code-block:: cpp

    residual.AddSurfaceIntegral(
      DependsOn< ... >{},
      [](auto ... args){
      	auto s = ...;
      	return s;
      }, 
      domain_of_integration
    );	

Note: the first argument ``DependsOn< ... >{}`` is a way to specify
which of the trial spaces (if any) are required by that integral.
e.g. ``DependsOn< 1, 2 >{}`` will indicate that the values
from trial spaces 1 and 2 (zero-based indexing) will be passed in
to the provided q-function.

Going back to our example problem (since we assumed 3D earlier) we can make an
``Add****Integral()`` call for each of the integral terms in the
original residual. In each of these functions, the first argument tells
which trial spaces the calculation depends on, the second argument is the
integrand (a lambda function or functor returning :math:`\{s, f\}`),
and the third argument is the domain of integration. Let's start with
the stress response term:

.. code-block:: cpp

  // The integrand lambda function is passed the spatial position of the quadrature point,
  // as well as a {value, derivative} tuple for the trial space.
  residual.AddVolumeIntegral(

    // this calculation depends on the displacement field, which is the 0th trial space
    DependsOn<0>{}, 

    [](auto x, auto disp){
     
      // Here, we unpack the {value, derivative} tuple into separate variables
      auto [u, grad_u] = disp;
      
      // call some constitutive model for the material in this domain
      auto stress = material_model(grad_u); 
      
      // Functional::AddVolumeIntegral() expects us to return a tuple of the form {s, f},
      // but this integral has no term that get integrated against the test functions,
      // so the "source" term is just zero
      return smith::tuple{zero{}, stress};
     
    }, 
    mesh
  );

The other terms follow a similar pattern. For the body force:

.. code-block:: cpp

  residual.AddVolumeIntegral(

    // this calculation doesn't require values from any trial space
    // so there is nothing between the angle brackets
    DependsOn</* nothing in here */>{}, 

    [](auto x){    

      // evaluate the body force function at the location of the quadrature point
      auto body_force = b(x); 
     
      // Functional::AddVolumeIntegral() expects us to return a tuple of the form {s, f},
      // but this integral has no term that get integrated against the test function gradients,
      // so the "flux" term is just zero
      return std::tuple{body_force, zero{}}; 
     
    }, 
    mesh
  );

And finally, for the surface tractions:

.. code-block:: cpp

    // Functional::AddSurfaceIntegral() only expects us to return s, so we don't need a tuple
    residual.AddSurfaceIntegral(

      // this calculation doesn't require values from any trial space
      // so there is nothing between the angle brackets
      DependsOn</* nothing in here */>{}, 

      // evaluate the traction at the location of the quadrature point
      // note: the q-function for boundary integrals is also passed
      // the unit surface normal as the second argument
      [](auto x, auto n){ return t(x); }, 

      surface_mesh
    );

Now that we've finished describing all the integral terms that appear in
our residual, we can carry out the actual calculation by calling
``Functional::operator()``:

.. code-block:: cpp

   auto r = residual(displacements);

Putting these snippets together without the verbose comments, we have (note: the two AddVolumeIntegrals were fused into one):

.. code-block:: cpp

    using test = H1<order, dim>;
    using trial = H1<order, dim>;
    Functional< test(trial) > residual(test_fes, trial_fes);

    // note: the first two AddVolumeIntegral calls can be fused
    // into one, provided they share the same domain of integration
    residual.AddVolumeIntegral(
      DependsOn<0>{}, // depends on the displacement field
      [](auto x, auto disp){
        auto [u, grad_u] = disp;
        return smith::tuple{b(x), material_model(grad_u)};
      }, 
      mesh
    );

    residual.AddSurfaceIntegral([](auto x, auto disp /* unused */){ return traction(x); }, surface_mesh);

    auto r = residual(displacements);

So, in only a few lines of code, we can create optimized, custom finite
element kernels!

Quantities of Interest
----------------------

``Functional`` can also be used to represent scalar-valued integral expressions. These can be used
to represent objective functions, constraints, or other "quantities of interest". To make a ``Functional``
with a scalar-valued output, use ``double`` as the test space in its function signature:

.. code-block:: cpp

    using trial = H1<order, dim>;

    // this indicates that the calculation will 
    // return a scalar, rather than a residual vector
    using test = double; 

    Functional< test(trial) > qoi(&test_fes, {&trial_fes});
    
    ...

Like before, the actual integral calculations are defined by calling the following member functions:

1. Integrals of the form:
   :math:`\displaystyle \iint_\Omega s \; da`

   .. code-block:: cpp

    qoi.AddAreaIntegral(
      DependsOn< ... >{},
      [](auto x, auto ... args){
      	auto s = ...;
      	return s;
      }, 
      domain_of_integration
    );

2. Integrals of the form:
   :math:`\displaystyle \iiint_\Omega s \; dv`

   .. code-block:: cpp

    qoi.AddVolumeIntegral(
      DependsOn< ... >{},
      [](auto x, auto ... args){
      	auto s = ...;
      	return s;
      }, 
      domain_of_integration
    );

3. Integrals of the form:
   :math:`\displaystyle \iint_{\partial \Omega} s \; da`

   .. code-block:: cpp

    qoi.AddSurfaceIntegral(
      DependsOn< ... >{},
      [](auto ... args){
      	auto s = ...;
      	return s;
      }, 
      domain_of_integration
    );

Note: since there aren't really test functions in this case (or equivalently, :math:`\phi(x) = 1`), there
is never a "flux" term, so these q-functions all just return a scalar. Here's an example of how to
use ``Functional`` to implement a strain-energy calculation to accompany our solid mechanics example:

Strain energy:   :math:`\displaystyle \qquad U(u) = \frac{1}{2} \iiint_\Omega \sigma : \epsilon \; dv`

.. code-block:: cpp

    using displacement_field = H1<order,dim>

    Functional< double(displacement_field) > strain_energy(&test_fes, {&trial_fes});
    strain_energy.AddVolumeIntegral(
      DependsOn<0>{}, // depends on displacement
      [](auto x, auto displacement){
        auto [u, dudx] = displacement;
        auto epsilon = 0.5 * (transpose(dudx) + dudx);
        auto sigma = my_material_model(epsilon);
        auto strain_energy_density = 0.5 * double_dot(sigma, epsilon);
        return strain_energy_density;
      },
      mesh
    );

Implementation
--------------

For the most part, the ``Functional`` class is just a container of
``Integral`` objects, and some prolongation and restriction operators to
get the data they need:

.. code-block:: cpp

   template <typename test, typename trial>
   struct Functional<test(trial)> : public mfem::Operator {
     ...
     std::vector< Integral<test(trial)> > domain_integrals;
     std::vector< Integral<test(trial)> > boundary_integrals;
   };

The calls to ``Functional::Add****Integral`` forward the integrand and
mesh information to an ``Integral`` constructor and add it to the
appropriate list (either ``domain_integrals`` or
``boundary_integrals``). MFEM treats domain and boundary integrals
differently, so we maintain them in separate lists.

From there, the ``Integral`` constructor uses the integrand functor to
specialize a highly templated finite element kernel (simplified
implementation given below).

.. code-block:: cpp

   template < ::mfem::Geometry::Type g, typename test, typename trial, int geometry_dim, int spatial_dim, int Q,
              typename derivatives_type, typename lambda>
   void evaluation_kernel(const mfem::Vector& U, mfem::Vector& R, derivatives_type* derivatives_ptr,
                          const mfem::Vector& J_, const mfem::Vector& X_, int num_elements, lambda qf)
   {
     ...

     // for each element in the domain
     for (int e = 0; e < num_elements; e++) {
     
       // get the values for this particular element
       tensor u_elem = detail::Load<trial_element>(u, e);

       // this is where we will accumulate the element residual tensor
       element_residual_type r_elem{};

       // for each quadrature point in the element
       for (int q = 0; q < static_cast<int>(rule.size()); q++) {
         // get the position of this quadrature point in the parent and physical space,
         // and calculate the measure of that point in physical space.
         auto   xi  = rule.points[q];
         auto   dxi = rule.weights[q];
         auto   x_q = make_tensor<spatial_dim>([&](int i) { return X(q, i, e); });
         auto   J_q = make_tensor<spatial_dim, geometry_dim>([&](int i, int j) { return J(q, i, j, e); });
         double dx  = detail::Measure(J_q) * dxi;

         // evaluate the value/derivatives needed for the q-function at this quadrature point
         auto arg = detail::Preprocess<trial_element>(u_elem, xi, J_q);

         // evaluate the user-specified constitutive model
         //
         // note: make_dual(arg) promotes those arguments to dual number types
         // so that qf_output will contain values and derivatives
         auto qf_output = qf(x_q, make_dual(arg));

         // integrate qf_output against test space shape functions / gradients
         // to get element residual contributions
         r_elem += detail::Postprocess<test_element>(get_value(qf_output), xi, J_q) * dx;
         
       }

       // once we've finished the element integration loop, write our element residuals
       // out to memory, to be later assembled into global residuals by mfem
       detail::Add(r, r_elem, e);
     }
   }

Then, the call to that specialized finite element kernel is wrapped
inside a ``std::function`` object with the appropriate signature. This
``std::function`` is used to implement the action of ``Mult()``:

.. code-block:: cpp

   template < typename spaces > 
   struct Integral {

     ...
     
     template <int geometry_dim, int spatial_dim, typename lambda_type>
     Integral(...) {

       ...
       
       evaluation = [=](const mfem::Vector& U, mfem::Vector& R) {
         evaluation_kernel<geometry, test_space, trial_space, geometry_dim, spatial_dim, Q>(...);
       };
       
       ...
       
     };
     
     void Mult(const mfem::Vector& input, mfem::Vector& output) const { evaluation(input, output); }
     
     std::function<void(const mfem::Vector&, mfem::Vector&)> evaluation;
     
   }

Finally, when the user calls ``Functional::operator()``, it loops over the
domain and surface integrals, calling ``Integral::Mult()`` on each one
to compute the weighted residual contribution from each term.

Shape-Aware Functional
----------------------

.. note::
   This is only available for scalar and vector-valued :math:`H^1` and :math:`L_2` trial function spaces,
   scalar and vector-valued :math:`H^1`, :math:`L_2`, and ``double`` (quantity of interest) test function spaces, and 
   vector-valued :math:`H^1` shape displacement spaces.

For shape optimization problems, it is often useful to compute derivatives with respect to a shape displacement field
the modifies the nodal positions of the underlying computational mesh (for example, see `Swartz et al. <https://link.springer.com/article/10.1007/s00158-023-03684-9>`_).
This method takes a spatial position :math:`X` in the reference computational domain :math:`\Omega` and perturbs it 
by a spatial displacement vector :math:`p` such that the finite element computation occurs at the updated spatial
coordinate :math:`x = X + p` in the updated computational domain :math:`\Omega_p`. This implies for a volumetric
finite element integral with :math:`H^1` trial function :math:`u` and  :math:`H^1` test function :math:`\psi`:

.. math::

  \begin{align*}
  r(p, u) &= \int_{\Omega_p} \left( \psi \cdot s(x, u, \nabla_x u) + \nabla_x \psi : f(x, u, \nabla_x u) \right) \; dx \\
  &= \int_{\Omega} \left( \psi \cdot s\left(X + p, u, \frac{\partial u}{\partial x}\right) + \left(\frac{\partial \psi}{\partial x}\right) : f\left(X + p, u, \frac{\partial u}{\partial x}\right) \right) \; \mbox{det}\left(\frac{\partial x}{\partial X}\right)dX \\
  &= \int_{\Omega} \left( \psi \cdot s\left(X + p, u, \frac{\partial u}{\partial X}\frac{\partial X}{\partial x} \right) + \left(\frac{\partial \psi}{\partial X}\frac{\partial X}{\partial x}\right) : f\left(X + p, u, \frac{\partial u}{\partial X}\frac{\partial X}{\partial x}\right) \right) \; \mbox{det}\left(\frac{\partial x}{\partial X}\right)dX \\
  &= \int_{\Omega} \left( \psi \cdot s\left(X + p, u, \frac{\partial u}{\partial X}\left( I + \frac{\partial p}{\partial X}  \right)^{-1}\right) + \right. \\
  & \;\;\;\;\; \left. \left(\frac{\partial \psi}{\partial X}\left( I + \frac{\partial p}{\partial X}  \right)^{-1} \right) : f\left(X + p, u, \frac{\partial u}{\partial X}\left( I + \frac{\partial p}{\partial X}  \right)^{-1}\right) \right) \; \mbox{det}\left( I + \frac{\partial p}{\partial X}  \right)dX \\
  \end{align*}

For a boundary finite element integral with :math:`H^1` trial function :math:`u` and  :math:`H^1` test function :math:`\psi`:

.. math::

  \begin{align*}
  r(p, u) &= \int_{\Gamma_p}  t(x, u) \cdot \psi\, dS_x \\
  &= \int_{\Gamma_p} t(x, u)\cdot  \psi \left \Vert \frac{\partial x}{\partial \xi} \times  \frac{\partial x}{\partial \eta}  \right \Vert d\xi d\eta \\
  &= \int_{\Gamma} t(X + p, u)\cdot \psi \left \Vert \frac{\partial (X + p)}{\partial \xi} \times  \frac{\partial (X + p)}{\partial \eta}  \right \Vert d\xi d\eta \\
  \end{align*}

While simple, these transformations can become repetitive and error-prone when writing Q-functions for parameter-free shape optimization problems. To 
address this issue, we automated these transformations in ``ShapeAwareFunctional``. This object has the same interface and integral definition routines as 
``Functional``, but it takes the shape displacement field :math:`p` as a required trial parameter. 

The integrands captured in the ``Functional::Add****Integral`` methods do not need any further modifications to become "shape-aware" and produce the shape derivatives
needed for automated shape optimization. The integrands will have their arguments adjusted appropriately to reflect `x = X + p` and their output flux will
be automatically modified per the formulas above.

As a simple example, consider the implementation of two quantities of interest:

1. Average of a trial function :math:`u` in a domain with an underlying shape displacement field :math:`p`:

.. math::
  
  \mbox{average}(u) = \frac{\int_{\Omega_p} u\, dx }{\int_{\Omega_p} dx}  

.. literalinclude:: ../../../../src/smith/numerics/functional/tests/functional_qoi.cpp
   :start-after: _average_start
   :end-before: _average_end
   :language: C++

2. Average of the absolute value of the normal component of a vector-valued trial function :math:`\mathbf{u}` on a boundary with an underlying shape displacement field :math:`p`:

.. math::
  
  \mbox{boundary average}(| \mathbf{u} \cdot n |) = \frac{\int_{\Gamma_p} | \mathbf{u} \cdot n | \, dS }{\int_{\Gamma_p} dS}

.. literalinclude:: ../../../../src/smith/numerics/functional/tests/functional_qoi.cpp
 :start-after: _boundary_start
 :end-before: _boundary_end
 :language: C++
