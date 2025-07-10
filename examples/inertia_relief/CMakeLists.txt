# Copyright (c) Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

if (STRUMPACK_DIR)
    blt_add_executable( NAME        inertia_relief_example
                        SOURCES     inertia_relief_example.cpp
                        OUTPUT_DIR ${EXAMPLE_OUTPUT_DIRECTORY}
                        DEPENDS_ON  serac_physics continuation_solver
                        )

    serac_add_example_test(NAME            inertia_relief
                           COMMAND         inertia_relief_example
                           NUM_MPI_TASKS   1)

    install(
        FILES
            inertia_relief_example.cpp
        DESTINATION
            examples/serac/inertia_relief
    )
endif()
