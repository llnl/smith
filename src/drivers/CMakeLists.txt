# Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

blt_add_executable( NAME        serac_driver
                    SOURCES     serac.cpp
                    DEPENDS_ON  serac_physics serac_mesh
                    OUTPUT_NAME serac
                    )

if (SERAC_ENABLE_TESTS)
    set(input_files_dir ${CMAKE_CURRENT_SOURCE_DIR}/../../data/input_files/tests)

    # Run basic test for the Serac driver
    blt_add_test(NAME          serac_driver_solid
                 COMMAND       ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/serac -o driver_solid -i ${input_files_dir}/solid/dyn_solve.lua
                 NUM_MPI_TASKS 1 )

    blt_add_test(NAME          serac_driver_heat_transfer
                 COMMAND       ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/serac -o driver_heat_transfer -i ${input_files_dir}/heat_transfer/static_solve.lua
                 NUM_MPI_TASKS 1 )

    blt_add_test(NAME          serac_driver_help
                 COMMAND       ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/serac --help 
                 NUM_MPI_TASKS 1 )

    blt_add_test(NAME          serac_driver_docs
                 COMMAND       ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/serac -o docs -d -i ${input_files_dir}/solid/qs_linear.lua
                 NUM_MPI_TASKS 1 )
endif()

install( TARGETS serac_driver
         RUNTIME DESTINATION bin
         )
