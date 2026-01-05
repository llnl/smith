# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

#smith_add_executable( NAME        smith_driver
#                      SOURCES     smith.cpp
#                      DEPENDS_ON  smith_physics
#                      OUTPUT_NAME smith
#                      )
#
#if (SMITH_ENABLE_TESTS)
#    set(input_files_dir ${CMAKE_CURRENT_SOURCE_DIR}/../../data/input_files/tests)
#
#    # Run basic test for the Smith driver
#    blt_add_test(NAME          smith_driver_solid
#                 COMMAND       ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/smith -o driver_solid -i ${input_files_dir}/solid/dyn_solve.lua
#                 NUM_MPI_TASKS 1 )
#
#    blt_add_test(NAME          smith_driver_heat_transfer
#                 COMMAND       ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/smith -o driver_heat_transfer -i ${input_files_dir}/heat_transfer/static_solve.lua
#                 NUM_MPI_TASKS 1 )
#
#    blt_add_test(NAME          smith_driver_help
#                 COMMAND       ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/smith --help 
#                 NUM_MPI_TASKS 1 )
#
#    blt_add_test(NAME          smith_driver_docs
#                 COMMAND       ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/smith -o docs -d -i ${input_files_dir}/solid/qs_linear.lua
#                 NUM_MPI_TASKS 1 )
#endif()
#
#install( TARGETS smith_driver
#         RUNTIME DESTINATION bin
#         )
