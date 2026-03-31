# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

set(numerics_test_dependencies smith_numerics smith_boundary_conditions gtest)

set(numerics_serial_test_sources
    test_equationsolver.cpp
    test_operator.cpp
    test_odes.cpp
    test_block_preconditioner.cpp
    test_block_preconditioner_backend.cpp
    )

smith_add_tests( SOURCES       ${numerics_serial_test_sources}
                 DEPENDS_ON    ${numerics_test_dependencies}
                 NUM_MPI_TASKS 1)

if(PETSC_FOUND)
    set(petsc_solver_tests
        test_equationsolver_petsc.cpp
        )
    smith_add_tests(SOURCES       ${petsc_solver_tests}
                    DEPENDS_ON    ${numerics_test_dependencies}
                    NUM_MPI_TASKS 1)

    if(SLEPC_FOUND)
        set(slepc_solver_tests
            test_eigensolver.cpp
            test_trust_region_solver.cpp
            )
        smith_add_tests(SOURCES       ${slepc_solver_tests}
                        DEPENDS_ON    ${numerics_test_dependencies}
                        NUM_MPI_TASKS 2)
    endif()

endif()

# TODO re-enable test once segfault is fixed (on gcc14 docker container)
# [SIGNAL]: Received signal 11 (Segmentation fault), exiting
# double free or corruption (!prev)
# if(CONTINUATION_FOUND)
#     set(continuation_solver_qp_test_sources
#         qp_test_continuation_solver.cpp
#         )
    
#     smith_add_tests(SOURCES       ${continuation_solver_qp_test_sources}
#                     DEPENDS_ON    ${numerics_test_dependencies} continuation_solver
#                     NUM_MPI_TASKS 4 )
# endif()
