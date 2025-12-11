# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

smith_add_executable( NAME        simple_conduction
                      SOURCES     simple_conduction.cpp
                      OUTPUT_DIR ${EXAMPLE_OUTPUT_DIRECTORY}
                      DEPENDS_ON  smith
                      )

install(
    FILES
        simple_conduction.cpp
    DESTINATION
        examples/smith/simple_conduction
)

if(SMITH_ENABLE_TESTS)
    blt_add_test(NAME          simple_conduction
                 COMMAND       simple_conduction
                 NUM_MPI_TASKS 1 )
endif()
