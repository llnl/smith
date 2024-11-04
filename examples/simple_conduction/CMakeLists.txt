# Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

blt_add_executable( NAME        simple_conduction_without_input_file
                    SOURCES     without_input_file.cpp
                    OUTPUT_DIR ${EXAMPLE_OUTPUT_DIRECTORY}
                    DEPENDS_ON  serac_physics serac_mesh serac_functional
                    )

if(SERAC_ENABLE_TESTS)
    blt_add_test(NAME          simple_conduction_without_input_file
                 COMMAND       simple_conduction_without_input_file
                 NUM_MPI_TASKS 1 )     
endif()
