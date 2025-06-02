# Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

blt_add_executable( NAME        first_driver
                    SOURCES     first_driver.cpp
                    OUTPUT_DIR ${EXAMPLE_OUTPUT_DIRECTORY}
                    DEPENDS_ON  serac_physics serac_mesh
                    )

blt_add_executable( NAME        curing_study_driver
                    SOURCES     curing_study_driver.cpp
                    OUTPUT_DIR ${EXAMPLE_OUTPUT_DIRECTORY}
                    DEPENDS_ON  serac_physics serac_mesh
                    )

blt_add_executable( NAME        curing_study_driver_nonlinear
                    SOURCES     curing_study_driver_nonlinear.cpp
                    OUTPUT_DIR ${EXAMPLE_OUTPUT_DIRECTORY}
                    DEPENDS_ON  serac_physics serac_mesh
                    )
install(
    FILES
        first_driver.cpp,
        curing_study_driver.cpp,
        curing_study_driver_nonlinear.cpp
    DESTINATION
        /g/g90/barrera/runs/digitalTwins/results
)
