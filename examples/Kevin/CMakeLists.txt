# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

smith_add_executable(
    NAME        material_identifiability
    SOURCES     MaterialIdentifiability.cpp
    OUTPUT_DIR  ${EXAMPLE_OUTPUT_DIRECTORY}
    DEPENDS_ON  smith
    )

smith_add_executable(
    NAME        material_identifiability_fixed_poisson
    SOURCES     MaterialIdentifiabilityFixedPoisson.cpp
    OUTPUT_DIR  ${EXAMPLE_OUTPUT_DIRECTORY}
    DEPENDS_ON  smith
    )

smith_add_executable(
    NAME        material_identifiability_fixed_poisson_50
    SOURCES     MaterialIdentifiabilityFixedPoisson50.cpp
    OUTPUT_DIR  ${EXAMPLE_OUTPUT_DIRECTORY}
    DEPENDS_ON  smith
    )

smith_add_executable(
    NAME        material_identifiability_designed_experiments
    SOURCES     MaterialIdentifiabilityDesignedExperiments.cpp
    OUTPUT_DIR  ${EXAMPLE_OUTPUT_DIRECTORY}
    DEPENDS_ON  smith
    )

smith_add_executable(
    NAME        gent_material_identifiability
    SOURCES     GentMaterialIdentifiability.cpp
    OUTPUT_DIR  ${EXAMPLE_OUTPUT_DIRECTORY}
    DEPENDS_ON  smith
    )

smith_add_example_test(
    NAME          material_identifiability
    COMMAND       material_identifiability
    NUM_MPI_TASKS 2
    )

smith_add_example_test(
    NAME          material_identifiability_fixed_poisson
    COMMAND       material_identifiability_fixed_poisson
    NUM_MPI_TASKS 2
    )

smith_add_example_test(
    NAME          material_identifiability_fixed_poisson_50
    COMMAND       material_identifiability_fixed_poisson_50
    NUM_MPI_TASKS 2
    )

smith_add_example_test(
    NAME          material_identifiability_designed_experiments
    COMMAND       material_identifiability_designed_experiments
    NUM_MPI_TASKS 2
    )

smith_add_example_test(
    NAME          gent_material_identifiability
    COMMAND       gent_material_identifiability
    NUM_MPI_TASKS 2
    )

install(
    FILES
        MaterialIdentifiability.cpp
        MaterialIdentifiabilityDesignedExperiments.cpp
        MaterialIdentifiabilityFixedPoisson.cpp
        MaterialIdentifiabilityFixedPoisson50.cpp
        GentMaterialIdentifiability.cpp
    DESTINATION
        examples/smith/Kevin
    )

install(
    DIRECTORY   ${CMAKE_CURRENT_SOURCE_DIR}/
    DESTINATION examples/smith/Kevin
    FILES_MATCHING
        PATTERN "*.g"
    )
