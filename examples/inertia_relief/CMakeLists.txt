# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

if (SMITH_ENABLE_CONTINUATION AND STRUMPACK_DIR)
    smith_add_executable( NAME        inertia_relief_example
                          SOURCES     inertia_relief_example.cpp
                          OUTPUT_DIR ${EXAMPLE_OUTPUT_DIRECTORY}
                          DEPENDS_ON  smith continuation_solver
                          )
    smith_add_example_test(NAME            inertia_relief
                           COMMAND         inertia_relief_example
                           NUM_MPI_TASKS   4)

    install(
        FILES
            inertia_relief_example.cpp
        DESTINATION
            examples/smith/inertia_relief
    )
endif()
