# Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

set(refactor_container_test_dependencies serac_refactor gtest)

set(refactor_container_test_sources
    accessor_tests.cpp
    ctor_tests.cpp
    reshape_tests.cpp
    stride_tests.cpp
    usage_tests.cpp
)

# TODO register CUDA tests

serac_add_tests( SOURCES       ${refactor_container_test_sources}
                 DEPENDS_ON    ${refactor_container_test_dependencies}
                 NUM_MPI_TASKS 1)
