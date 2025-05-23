# Copyright (c) Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

blt_add_executable( NAME        simple_conduction
                    SOURCES     simple_conduction.cpp
                    OUTPUT_DIR ${EXAMPLE_OUTPUT_DIRECTORY}
                    DEPENDS_ON  serac_physics
                    )

install(
    FILES
        simple_conduction.cpp
    DESTINATION
        examples/serac/simple_conduction
)

if(SERAC_ENABLE_TESTS)
    blt_add_test(NAME          simple_conduction
                 COMMAND       simple_conduction
                 NUM_MPI_TASKS 1 )
endif()
