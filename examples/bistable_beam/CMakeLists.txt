# Copyright (c) Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

blt_add_executable( NAME        bistable
                    SOURCES     bistable.cpp
                    OUTPUT_DIR ${EXAMPLE_OUTPUT_DIRECTORY}
                    DEPENDS_ON  serac_physics
                    )

if(SERAC_ENABLE_HIP)
    blt_add_target_compile_flags(TO bistable FLAGS "-fgpu-rdc")
    blt_add_target_link_flags(TO bistable FLAGS "-fgpu-rdc")
endif()

serac_add_example_test(NAME           bistable
                       COMMAND        bistable
                       NUM_MPI_TASKS  1)

install(
    FILES
        bistable.cpp
    DESTINATION
        examples/serac/bistable_beam
)
