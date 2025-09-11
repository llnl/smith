# Copyright (c) Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

if(ENZYME_FOUND)
    set(EXPLICIT_DYNAMICS_EXAMPLES_SOURCES
        dfem_gpu.cpp
    )

    foreach(filename ${EXPLICIT_DYNAMICS_EXAMPLES_SOURCES})
        get_filename_component(example_name ${filename} NAME_WE)

        blt_add_executable(NAME        explicit_${example_name}
                           SOURCES     ${filename}
                           OUTPUT_DIR  ${EXAMPLE_OUTPUT_DIRECTORY}
                           DEPENDS_ON  serac_physics serac_mesh_utils)

        serac_add_example_test(NAME           explicit_${example_name}
                               COMMAND        explicit_${example_name}
                               NUM_MPI_TASKS  4)
        
    endforeach()

    install(
        FILES
            ${EXPLICIT_DYNAMICS_EXAMPLES_SOURCES}
        DESTINATION
            examples/serac/explicit
    )

endif()
