# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
add_custom_target(compile_commands
    COMMAND ${CMAKE_COMMAND} -E create_symlink
            "${CMAKE_BINARY_DIR}/compile_commands.json"
            "${CMAKE_SOURCE_DIR}/compile_commands.json"
    BYPRODUCTS "${CMAKE_SOURCE_DIR}/compile_commands.json"
    COMMENT "Symlink compile_commands.json into project root"
)

smith_add_executable( NAME        PNC_double_network_test
                      SOURCES     PNC_double_network_test.cpp
                      OUTPUT_DIR ${EXAMPLE_OUTPUT_DIRECTORY}
                      DEPENDS_ON  smith
                      )

smith_add_executable( NAME        PNC_double_network_test_visco
                      SOURCES     PNC_double_network_test_visco.cpp
                      OUTPUT_DIR ${EXAMPLE_OUTPUT_DIRECTORY}
                      DEPENDS_ON  smith
                      )

smith_add_executable( NAME        square_lattice_test
                      SOURCES     square_lattice_test.cpp
                      OUTPUT_DIR ${EXAMPLE_OUTPUT_DIRECTORY}
                      DEPENDS_ON  smith
                      )

smith_add_executable( NAME        PNC_double_network_simple
                      SOURCES     PNC_double_network_simple.cpp
                      OUTPUT_DIR ${EXAMPLE_OUTPUT_DIRECTORY}
                      DEPENDS_ON  smith
                      )

smith_add_executable(NAME       extended_thermomechanics
                     SOURCES    extended_thermomechanics.cpp
                                helpers/extended_thermomechanics.hpp
                     OUTPUT_DIR ${EXAMPLE_OUTPUT_DIRECTORY}
                     DEPENDS_ON smith
                     )

smith_add_executable(NAME       bibeam_test_2d
                     SOURCES    bibeam_test_2d.cpp
                     OUTPUT_DIR ${EXAMPLE_OUTPUT_DIRECTORY}
                     DEPENDS_ON smith
                     )
                     
smith_add_executable(NAME       bz_staggered_thermomechanics
                     SOURCES   BZ_staggered_thermomechanics.cpp
                     OUTPUT_DIR ${EXAMPLE_OUTPUT_DIRECTORY}
                     DEPENDS_ON smith
                     )

smith_add_executable(NAME       PNC_simple_new
                     SOURCES   PNC_simple_new.cpp
                     OUTPUT_DIR ${EXAMPLE_OUTPUT_DIRECTORY}
                     DEPENDS_ON smith
                     )
install(
    FILES
        PNC_double_network_test.cpp
    DESTINATION
        examples/smith/PNC_double_network_test
)

install(
    FILES
        PNC_double_network_test_visco.cpp
    DESTINATION
        examples/smith/PNC_double_network_test_visco
)

install(
    FILES
        PNC_double_network_simple.cpp
    DESTINATION
        examples/smith/PNC_double_network_simple
)

install(
    FILES
        square_lattice_test.cpp
    DESTINATION
        examples/smith/square_lattice_test
)

install(
    FILES
        PNC_simple_new.cpp
    DESTINATION
        examples/smith/PNC_simple_new
)

if(SMITH_ENABLE_TESTS)
    blt_add_test(NAME          PNC_double_network_test
                 COMMAND       PNC_double_network_test
                 NUM_MPI_TASKS 1 )
    blt_add_test(NAME          PNC_double_network_test_visco
                 COMMAND       PNC_double_network_test_visco
                 NUM_MPI_TASKS 1 )
    blt_add_test(NAME          square_lattice_test
                 COMMAND       square_lattice_test
                 NUM_MPI_TASKS 1 )
    blt_add_test(NAME          PNC_double_network_simple
                 COMMAND       PNC_double_network_simple
                 NUM_MPI_TASKS 1 )
    blt_add_test(NAME          PNC_simple_new
                 COMMAND       PNC_simple_new
                 NUM_MPI_TASKS 1 )
endif()
