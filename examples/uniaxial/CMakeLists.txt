# Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

blt_add_executable( NAME        uniaxial_tension
                    SOURCES     uniaxial_tension.cpp
                    OUTPUT_DIR ${EXAMPLE_OUTPUT_DIRECTORY}
                    DEPENDS_ON  serac_physics serac_mesh
                    )

blt_add_test(NAME            uniaxial_tension
             COMMAND         uniaxial_tension
             NUM_MPI_TASKS   4)

install(
    FILES
        uniaxial_tension.cpp
    DESTINATION
        examples/serac/uniaxial
)
