# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

smith_add_executable( NAME        buckling_cylinder
                      SOURCES     cylinder.cpp
                      OUTPUT_DIR ${EXAMPLE_OUTPUT_DIRECTORY}
                      DEPENDS_ON  smith
                      )

if(SMITH_ENABLE_HIP)
    blt_add_target_compile_flags(TO buckling_cylinder FLAGS "-fgpu-rdc")
    blt_add_target_link_flags(TO buckling_cylinder FLAGS "-fgpu-rdc")
endif()

smith_add_example_test(NAME           buckling_cylinder_no_contact
                       COMMAND        buckling_cylinder --no-contact
                       NUM_MPI_TASKS  4)

if(TRIBOL_FOUND)
    blt_add_test(NAME           buckling_cylinder_fast
                 COMMAND        buckling_cylinder --fast
                 NUM_MPI_TASKS  4)

    smith_add_example_test(NAME           buckling_cylinder
                           COMMAND        buckling_cylinder
                           NUM_MPI_TASKS  4)
endif()

install(
    FILES
        cylinder.cpp
    DESTINATION
        examples/smith/buckling
)
