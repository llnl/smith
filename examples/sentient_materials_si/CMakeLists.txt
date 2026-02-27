# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

smith_add_executable(NAME       thermomechanics
                     SOURCES    thermomechanics.cpp
                     OUTPUT_DIR ${EXAMPLE_OUTPUT_DIRECTORY}
                     DEPENDS_ON smith
                     )

smith_add_executable(NAME       extended_thermomechanics
                     SOURCES    extended_thermomechanics.cpp
                                extended_thermomechanics.hpp
                     OUTPUT_DIR ${EXAMPLE_OUTPUT_DIRECTORY}
                     DEPENDS_ON smith
                     )

install(
    FILES
        thermomechanics.cpp
        extended_thermomechanics.cpp
        extended_thermomechanics.hpp
    DESTINATION
        examples/smith/sentient_materials_si
)

if(SMITH_ENABLE_TESTS)
    smith_add_example_test(NAME          thermomechanics
                           COMMAND       thermomechanics
                           NUM_MPI_TASKS 1)
    smith_add_example_test(NAME          extended_thermomechanics
                           COMMAND       extended_thermomechanics
                           NUM_MPI_TASKS 1)
endif()
