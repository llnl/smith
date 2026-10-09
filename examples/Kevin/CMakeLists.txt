# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

smith_add_executable(NAME       prescribed_temperature_mechanics
                     SOURCES    prescribed_temperature_mechanics.cpp
                                thermoelastic_material.hpp
                     OUTPUT_DIR ${EXAMPLE_OUTPUT_DIRECTORY}
                     DEPENDS_ON smith)

smith_add_executable(NAME       dynamic_temperature_quasistatic_mechanics
                     SOURCES    dynamic_temperature_quasistatic_mechanics.cpp
                                thermoelastic_material.hpp
                     OUTPUT_DIR ${EXAMPLE_OUTPUT_DIRECTORY}
                     DEPENDS_ON smith)

install(
    FILES
        prescribed_temperature_mechanics.cpp
        dynamic_temperature_quasistatic_mechanics.cpp
        thermoelastic_material.hpp
    DESTINATION
        examples/smith/Kevin
)

if(SMITH_ENABLE_TESTS)
    blt_add_test(NAME          prescribed_temperature_mechanics
                 COMMAND       prescribed_temperature_mechanics
                 NUM_MPI_TASKS 1)
    blt_add_test(NAME          dynamic_temperature_quasistatic_mechanics
                 COMMAND       dynamic_temperature_quasistatic_mechanics
                 NUM_MPI_TASKS 1)
endif()
