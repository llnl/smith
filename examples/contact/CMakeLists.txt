# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

if(TRIBOL_FOUND AND STRUMPACK_DIR)
    set(CONTACT_EXAMPLES_SOURCES
        beam_bending.cpp
        ironing.cpp
        sphere.cpp
        twist.cpp
        constraint_twist.cpp
    )

    foreach(filename ${CONTACT_EXAMPLES_SOURCES})
        get_filename_component(example_name ${filename} NAME_WE)

        serac_add_executable(NAME        contact_${example_name}
                             SOURCES     ${filename}
                             OUTPUT_DIR  ${EXAMPLE_OUTPUT_DIRECTORY}
                             DEPENDS_ON  serac_physics serac_mesh_utils)

        serac_add_example_test(NAME           contact_${example_name}
                               COMMAND        contact_${example_name}
                               NUM_MPI_TASKS  4)

        if(SERAC_ENABLE_HIP)
            blt_add_target_compile_flags(TO contact_${example_name} FLAGS "-fgpu-rdc")
            blt_add_target_link_flags(TO contact_${example_name} FLAGS "-fgpu-rdc")
        endif()
        
    endforeach()

    install(
        FILES
            ${CONTACT_EXAMPLES_SOURCES}
        DESTINATION
            examples/serac/contact
    )

endif()
