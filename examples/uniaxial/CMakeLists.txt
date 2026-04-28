# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

if (STRUMPACK_DIR)
    smith_add_executable( NAME        uniaxial_tension
                          SOURCES     uniaxial_tension.cpp
                          OUTPUT_DIR ${EXAMPLE_OUTPUT_DIRECTORY}
                          DEPENDS_ON  smith
                          )

    smith_add_example_test(NAME            uniaxial_tension
                           COMMAND         uniaxial_tension
                           NUM_MPI_TASKS   4)

    install(
        FILES
            uniaxial_tension.cpp
        DESTINATION
            examples/smith/uniaxial
    )
endif()
