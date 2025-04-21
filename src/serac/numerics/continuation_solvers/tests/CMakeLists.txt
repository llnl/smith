# Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

set(continuation_solvers_test_depends gtest serac_numerics serac_continuation_solvers ${continuation_solvers_depends})

# This test is all constexpr-evaluated, so it doesn't
# actually need to run, if it compiles without error, the tests have passed
#blt_add_executable(NAME        tensor_unit_tests
#                   SOURCES     tensor_unit_tests.cpp
#                   OUTPUT_DIR  ${TEST_OUTPUT_DIRECTORY}
#                   DEPENDS_ON  ${continuation_solvers_test_depends}
#                   FOLDER      serac/tests)

# Then add the examples/tests
set(continuation_solvers_parallel_test_sources
    ipm_qp.cpp
    )

serac_add_tests(SOURCES       ${continuation_solvers_parallel_test_sources}
                DEPENDS_ON    ${continuation_solvers_test_depends}
                NUM_MPI_TASKS 4 )

