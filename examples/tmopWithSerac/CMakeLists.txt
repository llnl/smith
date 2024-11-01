# Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

blt_add_executable( NAME        tmop_with_serac
                    SOURCES     tmopWithSeracInitialTesting.cpp
                    OUTPUT_DIR ${EXAMPLE_OUTPUT_DIRECTORY}
                    DEPENDS_ON  serac_physics serac_mesh serac_functional
                    )

blt_add_executable( NAME        tmop_multi_element
                    SOURCES     tmop_multi_element.cpp
                    OUTPUT_DIR ${EXAMPLE_OUTPUT_DIRECTORY}
                    DEPENDS_ON  serac_physics serac_mesh serac_functional
                    )

if(SERAC_ENABLE_TESTS)
    blt_add_test(NAME          tmop_with_serac
                COMMAND        tmop_with_serac
                NUM_MPI_TASKS 1 )
endif()