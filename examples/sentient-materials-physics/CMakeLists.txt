# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

smith_add_executable( NAME        PNC_double_network_test
                      SOURCES     PNC_double_network_test.cpp
                      OUTPUT_DIR ${EXAMPLE_OUTPUT_DIRECTORY}
                      DEPENDS_ON  smith
                      )


smith_add_executable(NAME       extended_thermomechanics
                     SOURCES    extended_thermomechanics.cpp
                                helpers/extended_thermomechanics.hpp
                     OUTPUT_DIR ${EXAMPLE_OUTPUT_DIRECTORY}
                     DEPENDS_ON smith
                     )
install(
    FILES
        PNC_double_network_test.cpp
    DESTINATION
        examples/smith/PNC_double_network_test
)

if(SMITH_ENABLE_TESTS)
    blt_add_test(NAME          PNC_double_network_test
                 COMMAND       PNC_double_network_test
                 NUM_MPI_TASKS 1 )
endif()
