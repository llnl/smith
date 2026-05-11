# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

smith_add_executable( NAME        composable_solid_mechanics
                      SOURCES     composable_solid_mechanics.cpp
                      OUTPUT_DIR ${EXAMPLE_OUTPUT_DIRECTORY}
                      DEPENDS_ON  smith
                      )

smith_add_executable( NAME        composable_cooks_membrane_fbar
                      SOURCES     composable_cooks_membrane_fbar.cpp
                      OUTPUT_DIR ${EXAMPLE_OUTPUT_DIRECTORY}
                      DEPENDS_ON  smith
                      )

install(
    FILES
        composable_solid_mechanics.cpp
        composable_cooks_membrane_fbar.cpp
    DESTINATION
        examples/smith/solid_mechanics
)

if(SMITH_ENABLE_TESTS)
    blt_add_test(NAME          composable_solid_mechanics
                 COMMAND       composable_solid_mechanics
                 NUM_MPI_TASKS 1 )
    blt_add_test(NAME          composable_cooks_membrane_fbar
                 COMMAND       composable_cooks_membrane_fbar
                 NUM_MPI_TASKS 1 )
endif()
