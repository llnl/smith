.. ## Copyright (c) Lawrence Livermore National Security, LLC and
.. ## other Smith Project Developers. See the top-level COPYRIGHT file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)

.. _tensor-label:

============
Tensor Class
============

``tensor`` is a class template for doing arithmetic on small,
statically-sized vectors, matrices, and tensors. To create one, specify
the underlying type in the first template argument, followed by a list
of integers for the shape. For example, ``tensor<float,3,2,4>`` is
conceptually similar to the type ``float[3][2][4]``.

Here are some examples and features:

-  ``tensor`` has `value semantics <https://akrzemi1.wordpress.com/2012/02/03/value-semantics/>`_ (in contrast to C++ multidimensional
   arrays):

   .. code-block:: c++

      // c-style arrays are okay for storage, but can't do much else
      {
        double u[3] = {1.0, 2.0, 3.0};
        double v[3] = u; // does not compile 
        double * w  = u; // does compile, but size information is lost and w is a shallow copy
      }

      // tensors store their own data and can be copied, assigned, referenced, and transformed
      {
        tensor < double, 3 > u = {1.0, 2.0, 3.0};
        tensor < double, 3 > v = u; // make a copy of u
        tensor < double, 3 > & w = u; // make a reference to u
      }

-  ``tensor`` supports operator overloading:

   .. code-block:: cpp

      tensor < double, 3 > u = {1.0, 2.0, 3.0};
      tensor < double, 3 > v = {4.0, 5.0, 6.0};
      tensor < double, 3 > sum = u + v;
      tensor < double, 3 > weighted_sum = 3.0 * u - v / 7.0;
      weighted_sum += sum;

-  ``tensor`` supports many common mathematical functions:

   .. code-block:: cpp

      tensor < double, 3, 3 > I = Identity<3>();
      tensor < double, 3, 3 > grad_u = {...};
      tensor < double, 3, 3 > strain = 0.5 * (grad_u + transpose(grad_u)); // or, equivalently, sym(grad_u)
      tensor < double, 3, 3 > stress = lambda * tr(strain) * I + 2.0 * mu * strain;

      tensor < double, 3 > traction = dot(stress, normal);
      tensor < double, 2, 3 > J = {...};
      double dA = sqrt(det(dot(J, transpose(J))));
      tensor < double, 3 > force = traction * dA;

-  ``tensor`` supports useful shortcuts (like `class template argument deduction <https://devblogs.microsoft.com/cppblog/how-to-use-class-template-argument-deduction/>`_):

   .. code-block:: cpp

      // class template argument deduction: A is deduced to have type tensor<double,2,2>
      tensor A = {{{1.0, 2.0}, {2.0, 3.0}}}; 

      // create tensors from index notation expressions, B has type tensor<double,3,3,3,3>
      tensor B = make_tensor<3,3,3,3>([](int i, int j, int k, int l){
        return 0.5 * ((i == j) * (k == l) + (i == l) * (k == j));
      });

      // slicing: get and set specific subtensors
      tensor< double, 3, 3 > C = B[0][2];
      B[2][0][0] = C[1];

      // access by operator() or operator[]
      C[1][1] = 2.0;
      C(2, 2) = 3.0;

-   ``tensor`` arithmetic supports ``constexpr`` evaluation:

   .. code-block:: cpp

      constexpr tensor change_of_basis_matrix = {{
        { 3.0, -2.0, 1.7},
        { 2.0,  8.0, 1.7},
        {-1.0,  4.0, 6.7}
      }};

      // express a quantity in a new basis
      tensor v = dot(change_of_basis_matrix, u);

      // modify the components in the new basis
      v = f(v);

      // precompute the inverse basis transformation at compile time
      constexpr tensor inverse_change_of_basis_matrix = inv(change_of_basis_matrix);

      // convert the modified values back to the original basis
      u = dot(inverse_change_of_basis_matrix, v);

-  ``tensor`` only allows operations between operands of appropriate
   shapes

   .. code-block:: cpp

      tensor< double, 3, 2 > A{};
      tensor< double, 3 > u{};
      tensor< double, 2 > v{};

      auto uA = dot(u, A); // works, returns tensor< double, 2 >
      auto Av = dot(A, v); // works, returns tensor< double, 3 >
      auto Au = dot(A, u); // compile error: incompatible dimensions for dot product
      auto vA = dot(v, A); // compile error: incompatible dimensions for dot product

      auto w = u + v; // compile error: can't add tensors of different shapes

      A[0] = v; // works, assign a new value to the first row of A
      A[1] = u; // compile error: can't assign a vector with 3 components to a vector of 2 components

.. _dual-label:

Dual Number Class
-----------------

``dual`` is a class template that behaves like a floating point value,
but also stores information about derivatives. For example, say we have
a function, :math:`f(x) = \frac{x \sin(\exp(x) - 2)}{1 + x^2}`. In C++,
one might implement this function as:

.. code-block:: cpp

   auto f = [](auto x){ return (x * sin(exp(x) - 2.0)) / (1 + x*x); };

If :math:`f(x)` is used in a larger optimization or root-finding
problem, we will likely also need to be able to evaluate
:math:`f\;'(x)`. Historically, the two most common ways to get this
derivative information were

1. Finite Difference Stencil:

   .. code-block:: cpp

      static constexpr double epsilon = 1.0e-9;
      auto dfdx = [](double x) { return (f(x + epsilon) - f(x - epsilon)) / (2.0 * epsilon); }

   This approach is simple, but requires multiple function invocations
   and the accuracy suffers due to catastrophic cancellation in floating point arithmetic.

2. Derive the expression for :math:`f\;'(x)`, either by hand or with a
   computer algebra system, and manually implement the result. For
   example, using Mathematica we get

   .. math:: f\;'(x) = \frac{\exp(x) (x + x^3) \cos(2 - \exp(x)) - (x^2 - 1) \sin(2 - \exp(x))}{(1 + x^2)^2},

   which must then be manually implemented in C++ code:

   .. code-block:: cpp

      auto dfdx = [](double x) {
        return (exp(x) * (x + x*x*x) * cos(2 - exp(x)) - (x*x - 1) * exp(2 - sin(x))) / ((1 + x*x) * (1 + x*x)); 
      };

   This approach can give very accurate results, and allows the
   derivative implementations to be individually optimized for
   performance. The downside is that the symbolic differentiation and
   manual implementation steps can be error prone: mistakes in
   transcription, differentiation, or implementation can be hard to
   notice.

   To emphasize this point, the expression for :math:`f\;'(x)` given
   above is actually incorrect, and the subsequent C++ implementation of
   that incorrect expression for :math:`f \; '(x)` is itself incorrect.
   But if you only skimmed the content above, you likely didn't notice.

The ``dual`` class template provides a 3rd option that improves on the
accuracy and performance of finite difference stencil, without
sacrificing accuracy. In addition, it doesn't require the developer to
manually differentiate and write new code that might contain errors. An
example:

.. code-block:: cpp

   double answer = f(x); // evaluate f at x
   dual< double > answer_and_derivative = f(make_dual(x)); // evaluate f and f' at x
   double just_the_answer = answer.value;
   double just_the_gradient = answer.gradient;

Internally, the implementation is remarkably simple:

.. code-block:: cpp

   template <typename gradient_type>
   struct dual {
     double        value;
     gradient_type gradient;
   };

That is, ``dual`` just stores a ``double`` value and a specified type
for the gradient term. Then, the basic rules of differentiation are
encoded in the corresponding operator overloads:

.. math:: \frac{d}{dx}(a + b) = \frac{da}{dx} + \frac{db}{dx}

.. code-block:: cpp

   template <typename gradient_type_a, typename gradient_type_b>
   constexpr auto operator+(dual<gradient_type_a> a, dual<gradient_type_b> b)
   {
     return dual{a.value + b.value, a.gradient + b.gradient};
   }

.. math:: \frac{d}{dx}(a\;b) = \frac{da}{dx} \; b + a \frac{db}{dx}

.. code-block:: cpp

   template <typename gradient_type_a, typename gradient_type_b>
   constexpr auto operator*(dual<gradient_type_a> a, dual<gradient_type_b> b)
   {
     return dual{a.value * b.value, a.gradient * b.value + a.value * b.gradient};
   }

and so on. In this way, when a dual number is passed in to a function,
each of the intermediate values keep track of gradient information as
well. The downside to this approach is that doing that arithmetic to
track the gradients of intermediate values is more expensive than
manually writing code for the derivatives.

However, by supporting both manually-written derivatives and ``dual``
numbers, users can choose to calculate derivatives in whatever manner is
appropriate for their problem: manually-written gradients for
performance-critical codepaths, and automatic differentiation for
iterating quickly on prototypes and research.

Some additional resources on the theory and implementation of automatic differentiation
are given below:

`Slides on AD Theory <https://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/slides/lec10.pdf>`_

`Article demonstrating how AD applies to a computational graph <https://towardsdatascience.com/automatic-differentiation-explained-b4ba8e60c2ad>`_

`C++ tools and libraries for AD <http://www.autodiff.org/?module=Tools&language=C%2FC%2B%2B>`_


Using ``tensor`` and ``dual`` together
--------------------------------------

In the previous example, :math:`f` was a function with a scalar input
and scalar output. In practice, most of the functions we care about are
more interesting. For example, an isotropic linear elastic material in
solid mechanics has the following stress-strain relationship:

.. math:: \sigma = \lambda \; \text{tr}(\epsilon) \; \mathbf{I} + 2 \; \mu \; \epsilon

or, in C++:

.. code-block:: cpp

   double lambda = 2.0;
   double mu = 1.0;
   static constexpr auto I = Identity<3>();
   auto stress = [=](auto strain){ return lambda * tr(strain) * I + 2 * mu * strain; };

That is, ``stress()`` takes a ``tensor<double,3,3>`` as input, and
outputs a ``tensor<double, 3, 3>``:

.. code-block:: cpp

   tensor< double, 3, 3 > epsilon = {...};
   tensor< double, 3, 3 > sigma = stress(epsilon);

In general, each part of a function's output can depend on each part of
its inputs. So, in this example the gradient could potentially have up
to 81 components:

.. math:: \frac{\partial \sigma_{ij}}{\partial \epsilon_{kl}}, \qquad i,j,k,l \in {1,2,3}

If we promote the input argument to a tensor of dual numbers, we can
compute these derivatives automatically:

.. code-block:: cpp

   tensor< double, 3, 3 > epsilon = {...};
   tensor< dual< tensor< double, 3, 3 > >, 3, 3 > sigma = stress(make_dual(epsilon));

Now, ``sigma`` contains value and gradient information that can be
understood in the following way:

.. math:: \texttt{sigma[i][j].value} = \sigma_{ij} \qquad \texttt{sigma[i][j].gradient[k][l]} = \frac{\partial \sigma_{ij}}{\partial \epsilon_{kl}}

There are also convenience routines to extract all the values and
gradient terms into their own tensors of the appropriate shape:

.. code-block:: cpp

   // as before
   tensor< dual< tensor< double, 3, 3 > >, 3, 3 > sigma = stress(make_dual(epsilon));

   // extract the values
   tensor< double, 3, 3 > sigma_values = get_value(sigma);

   // extract the gradient
   tensor< double, 3, 3, 3, 3 > sigma_gradients = get_gradient(sigma);



Differentiating Functions with Multiple Inputs and Outputs
-----------------------------------------------------------

Now let's consider a function that has multiple inputs and multiple outputs:

.. code-block:: cpp

   double mu = 1.0;
   double rho = 2.0;
   static constexpr auto I = Identity<3>();
   auto f = [=](auto p, auto v, auto L){ 
      auto strain_rate = 0.5 * (L + transpose(L));
      auto stress = - p * I + 2 * mu * strain_rate;
      auto kinetic_energy_density = 0.5 * rho * dot(v, v);
      return smith::tuple{stress, kinetic_energy_density};
   };

Here, ``f`` calculates the stress, :math:`\sigma`, and local kinetic energy density, :math:`q`, of a fluid in terms of
the pressure ``p`` (scalar), velocity ``v`` (3-vector), and velocity gradient ``L`` (3x3 matrix).
So, there are 2 outputs and 3 inputs, resulting in potentially 6 derivatives with different order tensors:

.. math:: 

   \frac{\partial \sigma}{\partial p}, \frac{\partial \sigma}{\partial v}, \frac{\partial \sigma}{\partial L},
   \frac{\partial q}{\partial p}, \frac{\partial q}{\partial v}, \frac{\partial q}{\partial L}

All of these derivatives can be calculated in a single function invocation by following the same
pattern as before:

.. code-block:: cpp

   double p = ...;
   tensor<double,3> v = ...;
   tensor<double,3,3> L = ...;

   // promote the arguments to dual numbers with make_dual()
   tuple dual_args = make_dual(smith::tuple{p, v, L});

   // then call the function with the dual arguments
   auto outputs = apply(f, dual_args);

   // note: smith::apply is a way to pass an n-tuple to a function that expects n arguments 
   // 
   // i.e. the two following lines have the same effect
   // f(p, v, L);
   // smith::apply(f, smith::tuple{p, v, L});

Like before, ``outputs`` will now contain the actual output values, but also all gradient terms (6, in this case).
To get the gradient tensors, we call the same ``get_gradient()`` function:

.. code-block:: cpp

   auto gradients = get_gradient(outputs);

The 6 gradient terms for this example can be thought of in a "matrix" where the :math:`i,j` entry is
the derivative of the :math:`i^{th}` output with respect to the :math:`j^{th}` input:

.. math::

   \bigg[\frac{\partial f_i}{\partial x_j}\bigg]
   =
   \begin{bmatrix}
   \frac{\partial \sigma}{\partial p} & 
   \frac{\partial \sigma}{\partial v} & 
   \frac{\partial \sigma}{\partial L}
   \\
   \frac{\partial q}{\partial p} & 
   \frac{\partial q}{\partial v} & 
   \frac{\partial q}{\partial L}
   \end{bmatrix}

The type returned by ``get_gradient()`` reflects this structure: returning a ``smith::tuple`` of ``smith::tuple``.
So for this example, the return type will be of the form:

.. code-block:: cpp

  smith::tuple<
    smith::tuple< df1_dx1_type, df1_dx2_type, df1_dx2_type >, 
    smith::tuple< df2_dx1_type, df2_dx2_type, df2_dx2_type >
  >;

The individual blocks can be accessed by using ``smith::get()``.

Finally, if we look at the actual types contained in ``get_gradient(output)`` we see a few interesting details:

.. code-block:: cpp

   smith::tuple<
     smith::tuple<tensor<double, 3, 3>, zero,              tensor<double, 3, 3, 3, 3> >, 
     smith::tuple<zero,                 tensor<double, 3>, zero                       > 
   > gradients = get_gradient(outputs);

First, the tensor shapes of the individual blocks are are in agreement with what we expect (e.g. 
:math:`\frac{\partial \sigma}{\partial p}` is 3x3, :math:`\frac{\partial \sigma}{\partial L}` is 3x3x3x3, etc).

Second, some of the derivative blocks seem to be missing! 
Instead of actual tensors, a mysterious type ``zero`` appears in three of the blocks
of our derivative. What does that mean?

It means that if we look back at the original definition of our function, we see that the stress tensor does not depend on ``v`` at all.
Similarly, the kinetic energy density only depends on ``v``, while having no dependence on ``p`` or ``L``. The implementation of the
``tensor`` and ``dual`` class templates automatically detects and optimizes away unnecessary storage and calculations associated with
these derivative blocks that are identically zero.
