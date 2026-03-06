.. ## Copyright (c) Lawrence Livermore National Security, LLC and
.. ## other Smith Project Developers. See the top-level COPYRIGHT file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)

===========
Style Guide
===========

Code Style
----------

This project follows Google's `C++ Style Guide <https://google.github.io/styleguide/cppguide.html>`_
with the following amendments:

    1. ``camelCase`` should be used for function names
    #. ``ALL_CAPS`` should be used for constants (in addition to macros)

If a class/function could feasibly be upstreamed to MFEM or implements an MFEM interface, it should
be part of the ``smith::mfem_ext`` namespace and use MFEM's ``PascalCase`` naming convention.

The Google style guide is meant for style enforcement only. The design principles outlined in the 
`C++ Core Guidelines <http://isocpp.github.io/CppCoreGuidelines/>`_ should be followed.

Of particular importance are the guidelines proposed for managing object lifetime:

    1. Use raw pointers and references [only] to denote non-ownership (R.3 - R.4)
    #. Prefer ``unique_ptr`` to ``shared_ptr`` (R.21)

For example, if an object ``A`` creates a subobject ``B`` whose constructor requires a reference
to one of ``A``'s instance variables ``Foo f``, ``B`` should store a non-owning reference to ``f``,
either ``Foo&`` or ``Foo*``.  This should be ``const`` if at all possible.  In this case, shared ownership
is not required because the lifetime of ``B`` is entirely dependent on the lifetime of ``A``.

How to style your code
----------------------

We have two methods of enabling styling your code via ClangFormat. 

The first method is to use the `style` build target on a configuration with the correct version of
ClangFormat enabled. Here is an example on how to do it on LC's Ruby machine:

.. code-block:: bash

    $ ./config-build.py host-configs/ruby-toss_4_x86_64_ib-clang@14.0.6.cmake
    $ cd build-ruby-toss_4_x86_64_ib-clang@14.0.6-debug
    $ make style

The second method is to make a comment of ``/style`` on your open GitHub pull request. This will trigger
a GitHub Action that will automatically style your code and commit it to your branch. You will need to
`git pull` after it is finished to continue to work on that branch.

Documentation
-------------

Functions, structs, classes, and critical member variables should be annotated with `Doxygen <https://www.doxygen.nl/manual/>`_ 
comments.  These comments should be enclosed in `Javadoc-style <https://www.doxygen.nl/manual/docblocks.html#cppblock>`_ comment blocks.
For example, a variable can be documented as follows:

::

    /** 
     * The MPI communicator
     */
    MPI_Comm m_comm;

When annotating code, especially functions, Doxygen's `special commands <https://www.doxygen.nl/manual/commands.html>`_ 
can come in handy to provide additional information:

::

    /** 
     * Calculate du_dt = M^-1 (-Ku + f).
     * This is all that is needed for explicit methods
     * @param[in] u The state vector (input to the differentiation)
     * @param[out] du The derivative of @p u with respect to time
     * @see https://mfem.github.io/doxygen/html/classmfem_1_1TimeDependentOperator.html
     */
    virtual void Mult(const mfem::Vector &u, mfem::Vector &du_dt) const;

For non-``void`` functions, the ``@return`` command should be used to describe the return value.
