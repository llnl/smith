.. ## Copyright (c) Lawrence Livermore National Security, LLC and
.. ## other Smith Project Developers. See the top-level COPYRIGHT file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)

===================================
Frequently Used Modern C++ Features
===================================

Smith currently uses C++17.  Several modern C++ features and library components are used heavily throughout Smith.

Smart pointers are used to avoid directly using ``operator new`` and ``operator delete`` except when absolutely necessary.
``std::unique_ptr<T>`` is used to denote **exclusive** ownership of a pointer to ``T`` - see `this article <https://www.drdobbs.com/cpp/c11-uniqueptr/240002708>`__ for more info.
Because ``unique_ptr`` implies unique/exclusive ownership, instances cannot be copied.  For example, if a function has a ``unique_ptr`` argument, a caller must utilize
*move semantics* to transfer ownership at the call site.  The linked article provides an example of this, and move semantics are discussed in a more general sense 
`here <https://herbsutter.com/2020/02/17/move-simply/>`__.

``std::shared_ptr<T>`` is used to denote **shared** ownership of a pointer to ``T`` - see `this article <https://docs.microsoft.com/en-us/cpp/cpp/how-to-create-and-use-shared-ptr-instances?view=vs-2019>`_ for example uses.
``shared_ptr`` s should be used sparingly.  Often, when two objects need to share a resource, it is sufficient for only one of the objects to 
be responsible for the lifetime of the shared resource; the other object can store a reference to the resource.

``std::optional<T>`` is used to express the idea of ``Maybe T``, a.k.a. a nullable type.  An ``optional`` is optionally a ``T``,
which is useful as a return type for functions that can fail.  It is preferable to values that are implied to be invalid or 
represent failure, e.g., ``std::optional<int>`` should be used instead of -1 to represent an invalid array index.  It is also preferred
as an alternative to functions that return ``nullptr`` on failure.  You can read more about ``optional`` `here <https://www.bfilipek.com/2018/05/using-optional.html>`__.

``std::variant<T1, T2, T3, ...>`` is use to express the idea of ``Either T1 or T2 or T3 or ...``.  It is the type- and memory-safe
version of a ``union``.  `This article <https://arne-mertz.de/2018/05/modern-c-features-stdvariant-and-stdvisit/>`__ goes into more
detail, but typically this is used to "tie" together classes that are used in the same context but are not conducive to an
inheritance hierarchy.

Lambdas are also used frequently to declare small functions immediately before they are used, e.g., before they are passed to another function.
Lambdas are very useful with ``std::algorithm`` s (introduced `here <https://www.fluentcpp.com/2017/01/05/the-importance-of-knowing-stl-algorithms/>`_), 
which are often preferable to traditional ``for`` loops as they more clearly express intent.  Lambdas can also *capture* variables available
in the scope in which they are declared - see `this page <https://docs.microsoft.com/en-us/cpp/cpp/lambda-expressions-in-cpp>`__ for more info.

Finally, range-based ``for`` loops (described `here <https://en.cppreference.com/w/cpp/language/range-for>`__) should be used 
whenever possible instead of integer-iterator-based indexing.  This is supported for all standard library containers.

For a comprehensive overview of modern C++ (C++11 onwards), Scott Meyer's *Effective Modern C++* is quite useful.
