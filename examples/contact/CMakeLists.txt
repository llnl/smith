# Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

if(TRIBOL_FOUND)
    set(CONTACT_EXAMPLES_SOURCES
        beam_bending.cpp
        ironing.cpp
        sphere.cpp
        twist.cpp
    )

    foreach(filename ${CONTACT_EXAMPLES_SOURCES})
        get_filename_component(example_name ${filename} NAME_WE)

        blt_add_executable(NAME        contact_${example_name}
                           SOURCES     ${filename}
                           OUTPUT_DIR  ${EXAMPLE_OUTPUT_DIRECTORY}
                           DEPENDS_ON  serac_physics serac_mesh)
    endforeach()

    install(
        FILES
            ${CONTACT_EXAMPLES_SOURCES}
        DESTINATION
            examples/serac/contact
    )

endif()
