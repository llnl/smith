# Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

# blt_add_executable( NAME        first_driver
#                     SOURCES     first_driver.cpp
#                     OUTPUT_DIR ${EXAMPLE_OUTPUT_DIRECTORY}
#                     DEPENDS_ON  serac_physics
#                     )

blt_add_executable( NAME        curing_study_driver
                    SOURCES     curing_study_driver.cpp
                    OUTPUT_DIR ${EXAMPLE_OUTPUT_DIRECTORY}
                    DEPENDS_ON  serac_physics
                    )

blt_add_executable( NAME        curing_study_driver_nonlinear
                    SOURCES     curing_study_driver_nonlinear.cpp
                    OUTPUT_DIR ${EXAMPLE_OUTPUT_DIRECTORY}
                    DEPENDS_ON  serac_physics
                    )

blt_add_executable( NAME        curing_study_driver_nonlinear_with_contact
                    SOURCES     curing_study_driver_nonlinear_with_contact.cpp
                    OUTPUT_DIR ${EXAMPLE_OUTPUT_DIRECTORY}
                    DEPENDS_ON  serac_physics
                    )
 
blt_add_executable( NAME        indentation_with_contact
                    SOURCES     indentation_with_contact.cpp
                    OUTPUT_DIR ${EXAMPLE_OUTPUT_DIRECTORY}
                    DEPENDS_ON  serac_physics
                    )

blt_add_executable( NAME        indentation_with_contact_nonlinear
                    SOURCES     indentation_with_contact_nonlinear.cpp
                    OUTPUT_DIR ${EXAMPLE_OUTPUT_DIRECTORY}
                    DEPENDS_ON  serac_physics
                    )

# blt_add_executable( NAME        mesh_concatenation_with_axom
#                     SOURCES     mesh_concatenation_with_axom.cpp
#                     OUTPUT_DIR ${EXAMPLE_OUTPUT_DIRECTORY}
#                     DEPENDS_ON  serac_physics
#                     )

# blt_add_executable( NAME        indentation_with_contact_nonlinear_two_meshes
#                     SOURCES     indentation_with_contact_nonlinear_two_meshes.cpp
#                     OUTPUT_DIR ${EXAMPLE_OUTPUT_DIRECTORY}
#                     DEPENDS_ON  serac_physics
#                     )
# Install the executables to a specific directory
install(
    FILES
        first_driver.cpp,
        curing_study_driver.cpp,
        curing_study_driver_nonlinear.cpp,
        curing_study_driver_nonlinear_with_contact.cpp,
        indentation_with_contact_nonlinear.cpp
    DESTINATION
        /g/g90/barrera/runs_lustre/digital_twins_SI/serac_runs/
)
