.. ## Copyright (c) Lawrence Livermore National Security, LLC and
.. ## other Smith Project Developers. See the top-level COPYRIGHT file for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)

===============
Memory Checking
===============

There are two commonly available memory checkers available to use with Smith on LC:
`AddressSanitizer <https://github.com/google/sanitizers/wiki/AddressSanitizer>`_
and `Valgrind <https://valgrind.org/>`_.

AddressSanitizer
----------------

AddressSanitizer (aka Asan) is memory error detection tool that is a part of LLVM.  It
very fast and easy to use but doesn't seem as robust as Valgrind.  It requires compile
and link flags which are enabled via the CMake option ``ENABLE_ASAN``.  Anything in our CMake
system will get those flags after that is enabled but our third-party libraries (like MFEM)
will not. After that just run your built executable and Asan will output a log to the screen
after your program is done running.  Asan's behavior can be modified with a set of
`environment variables <https://github.com/google/sanitizers/wiki/AddressSanitizerFlags>`_ .

.. note::
    Asan only works with the Clang and GCC compiler chains.  Our build system will throw
    an error if you try to build with anything else while ``ENABLE_ASAN`` is ``ON``.

Here is a recommended workflow:

.. code-block:: bash

    ./config-build.py -hc host-configs/ruby-toss_4_x86_64_ib-clang@14.0.6.cmake -DENABLE_ASAN=ON
    cd build-ruby-toss_4_x86_64_ib-clang@14.0.6-debug
    srun -N1 --exclusive --mpi-bind=off make -j
    LSAN_OPTIONS=suppressions=../suppressions.asan ASAN_OPTIONS=log_path=asan.out:log_exe_name=true srun -n2 bin/smith

This will output files in the current directory for each process that follow the pattern:
``asan.out.<exe name>.<pid>``.  It also sets your return code to a non-zero value if there
were any non-suppressed memory errors.

``LSAN_OPTIONS`` and ``ASAN_OPTIONS`` are delimited by ':'.

Here is an explanation of the given options (all should be added to ``ASAN_OPTIONS`` unless noted):

  * ``suppressions``: Location of memory leak suppression file (``LSAN_OPTIONS``)
  * ``log_path``: Logs to the given file instead of to the screen. This is very helpful
    to avoid intermingled lines on the screen from every process
  * ``log_exe_name``: Adds executable name to log_path

Helpful options:

  * ``fast_unwind_on_malloc=0``: This improves Asan's stack tracing ability but also greatly slows
    down the run
  * ``exitcode=0``: This stops Asan from returning a a non-zero exit code from your executable
    (defaults to 23) (``LSAN_OPTIONS``)
    
    
Debugging with Address Sanitizer enabled
========================================
    
If a program results in address sanitizer emitting an error (gcc example, here):

.. code-block:: bash

    ==45800==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x61a00000a1c0 at pc 0x5561996372b7 bp 0x7fff89f707e0 sp 0x7fff89f707d0
    READ of size 8 at 0x61a00000a1c0 thread T0
        #0 0x5561996372b6 in smith::Functional<double (smith::H1<2, 1>), (smith::ExecutionSpace)0>::Gradient::operator mfem::Vector() (/home/sam/code/smith/build/tests/functional_qoi+0x57e2b6)
        #1 0x55619962df8c in void check_gradient<double (smith::H1<2, 1>)>(smith::Functional<double (smith::H1<2, 1>), (smith::ExecutionSpace)0>&, mfem::Vector&) (/home/sam/code/smith/build/tests/functional_qoi+0x574f8c)
        #2 0x556199624144 in void functional_qoi_test<2, 2>(mfem::ParMesh&, smith::H1<2, 1>, smith::Dimension<2>) (/home/sam/code/smith/build/tests/functional_qoi+0x56b144)
        #3 0x556199614ee3 in main /home/sam/code/smith/src/smith/physics/utilities/functional/tests/functional_qoi.cpp:206

the information provided (e.g. invalid read at /home/sam/code/smith/build/tests/functional_qoi+0x57e2b6) doesn't precisely reveal where the error takes place.
We can set a breakpoint on the address sanitizer error function to find out exactly where the problem lies, but first we need to find out the name of that
error reporting symbol. One way to do this is to filter the symbols in the executable:

.. code-block:: bash

    sam@provolone:~/code/smith/build/tests$ nm -g ./functional_qoi | grep asan
                     U __asan_after_dynamic_init
                     U __asan_before_dynamic_init
                     U __asan_handle_no_return
                     U __asan_init
    0000000001646450 B __asan_option_detect_stack_use_after_return
                     U __asan_poison_stack_memory
                     U __asan_register_globals
                     U __asan_report_load1
                     U __asan_report_load16
                     U __asan_report_load2
                     U __asan_report_load4
                     U __asan_report_load8
                     U __asan_report_load_n
                     U __asan_report_store1
                     U __asan_report_store16
                     U __asan_report_store4
                     U __asan_report_store8
                     U __asan_report_store_n
                     U __asan_stack_free_5
                     U __asan_stack_free_6
                     U __asan_stack_free_7
                     U __asan_stack_free_8
                     U __asan_stack_free_9
                     U __asan_stack_malloc_0
                     U __asan_stack_malloc_1
                     U __asan_stack_malloc_2
                     U __asan_stack_malloc_3
                     U __asan_stack_malloc_4
                     U __asan_stack_malloc_5
                     U __asan_stack_malloc_6
                     U __asan_stack_malloc_7
                     U __asan_stack_malloc_8
                     U __asan_stack_malloc_9
                     U __asan_unpoison_stack_memory
                     U __asan_unregister_globals
                     U __asan_version_mismatch_check_v8
    00000000016466a1 B __odr_asan.mesh2D
    00000000016466a0 B __odr_asan.mesh3D
    00000000016466a3 B __odr_asan.myid
    00000000016466a2 B __odr_asan.nsamples
    00000000016466a4 B __odr_asan.num_procs

and select the one that corresponds to the the original error (read of size 8 => ``__asan_report_load8``).


Valgrind
--------

Valgrind is a very powerful set of tools that help with dynamic analysis tools.  We will
focus on `memcheck <https://valgrind.org/docs/manual/mc-manual.html>`_ which is a memory
error detection tool.

Unlike Asan, valgrind does not need any special compiler flags.  Just build your executable
and run your executable with ``valgrind``. Valgrind's suppression files are easily generated by
valgrind with ``--gen-suppressions=all`` and are more customizable than Asan's.

Here is a recommended workflow:

.. code-block:: bash

    ./config-build.py -hc host-configs/rzgenie-toss_3_x86_64_ib-gcc@8.1.0.cmake
    cd build-rzgenie-toss_3_x86_64_ib-gcc@8.1.0-debug
    srun -N1 --exclusive --mpi-bind=off make -j
    srun -n2 valgrind --tool=memcheck --log-file=valgrind.out --leak-check=yes --show-leak-kinds=all --num-callers=20 --suppressions=../suppressions.valgrind bin/smith

This will produce a file called ``valgrind.out`` in the current directory with a valgrind report.

Here is an explanation of the given options:

 * ``--tool=memcheck``: valgrind is a tool-suite so this runs the memcheck tool
 * ``--log-file=valgrind.out``: Logs report to the given file
 * ``--leak-check=yes``: Enables memory leak checks
 * ``--show-leak-kinds=all```: Enables showing all memory leak kinds
 * ``--num-callers=20``: Limits the size of the stack traces to 20
 * ``--suppressions=../suppressions.valgrind``: Location of memory leak suppression file
