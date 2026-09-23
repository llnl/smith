# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

if (SMITH_ENABLE_CONTINUATION AND STRUMPACK_FOUND)
    smith_add_executable( NAME        obstacle_design_example
                          SOURCES     obstacle_design_example.cpp
                          OUTPUT_DIR ${EXAMPLE_OUTPUT_DIRECTORY}
                          DEPENDS_ON  smith continuation_solver
                          )
    smith_add_example_test(NAME            obstacle_design
                           COMMAND         obstacle_design_example
                           NUM_MPI_TASKS   4)

    install(
        FILES
            obstacle_design_example.cpp
        DESTINATION
            examples/smith/obstacle_design
    )
endif()
