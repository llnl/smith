# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

smith_add_executable( NAME        composable_thermo_mechanics
                      SOURCES     composable_thermo_mechanics.cpp
                      OUTPUT_DIR ${EXAMPLE_OUTPUT_DIRECTORY}
                      DEPENDS_ON  smith
                      )

install(
    FILES
        composable_thermo_mechanics.cpp
    DESTINATION
        examples/smith/thermo_mechanics
)

if(SMITH_ENABLE_TESTS)
    blt_add_test(NAME          composable_thermo_mechanics
                 COMMAND       composable_thermo_mechanics
                 NUM_MPI_TASKS 1 )
endif()
