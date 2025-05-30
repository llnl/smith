# Copyright (c) Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

if(TRIBOL_FOUND AND PETSC_FOUND)
    blt_add_executable( NAME        buckling_cylinder
                        SOURCES     cylinder.cpp
                        OUTPUT_DIR ${EXAMPLE_OUTPUT_DIRECTORY}
                        DEPENDS_ON  serac_physics serac_mesh
                        )

    install(
        FILES
            cylinder.cpp
        DESTINATION
            examples/serac/buckling
    )

endif()
