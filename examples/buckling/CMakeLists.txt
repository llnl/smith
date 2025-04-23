# Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
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

    blt_add_test(NAME            buckling_cylinder
                 COMMAND         buckling_cylinder
                 NUM_MPI_TASKS   1)

    install(
        FILES
            cylinder.cpp
        DESTINATION
            examples/serac/buckling
    )

endif()
