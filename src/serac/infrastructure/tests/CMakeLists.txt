# Copyright (c) Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause) 

set(infrastructure_test_depends serac_infrastructure gtest)

set(infrastructure_test_sources
    application_manager.cpp
    debug_print.cpp
    input.cpp)

serac_add_tests( SOURCES    ${infrastructure_test_sources}
                 DEPENDS_ON ${infrastructure_test_depends})


# Infrastructure tests that require more than just infrastructure

serac_add_tests( SOURCES    error_handling.cpp
                 DEPENDS_ON serac_physics ${infrastructure_test_depends})

serac_add_tests( SOURCES    profiling.cpp
                 DEPENDS_ON serac_mesh_utils ${infrastructure_test_depends})
