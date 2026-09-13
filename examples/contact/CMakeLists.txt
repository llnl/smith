# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

if(TRIBOL_FOUND AND STRUMPACK_FOUND)
    set(CONTACT_EXAMPLES_SOURCES
        beam_bending.cpp
        ironing.cpp
        sphere.cpp
        twist.cpp
    )

    if(ENZYME_FOUND)
        list(APPEND CONTACT_EXAMPLES_SOURCES
            c_shape.cpp
            ironing_2D.cpp
        )
    endif()

    foreach(filename ${CONTACT_EXAMPLES_SOURCES})
        get_filename_component(example_name ${filename} NAME_WE)

        smith_add_executable(NAME        contact_${example_name}
                             SOURCES     ${filename}
                             OUTPUT_DIR  ${EXAMPLE_OUTPUT_DIRECTORY}
                             DEPENDS_ON  smith)

        if(example_name STREQUAL "c_shape")
            smith_add_example_test(NAME           contact_c_shape_three_interactions
                                   COMMAND        contact_${example_name}
                                                  --contact-interactions 3
                                                  --num-x-elements 4
                                                  --num-y-elements 4
                                                  --arm-thickness-elements 1
                                                  --residual-gap 0.05
                                                  --num-steps 5
                                                  --penalty 100
                                                  --traction 0.05
                                                  --binning-proximity-scale 10
                                                  --normal-smoothing-start-angle 90
                                   NUM_MPI_TASKS  4)
            smith_add_example_test(NAME           contact_c_shape_one_interaction
                                   COMMAND        contact_${example_name}
                                                  --contact-interactions 1
                                                  --num-x-elements 4
                                                  --num-y-elements 4
                                                  --arm-thickness-elements 1
                                                  --residual-gap 0.05
                                                  --num-steps 5
                                                  --penalty 100
                                                  --traction 0.05
                                                  --binning-proximity-scale 10
                                                  --normal-smoothing-start-angle 90
                                   NUM_MPI_TASKS  4)
        elseif(example_name STREQUAL "ironing_2D")
            smith_add_example_test(NAME           contact_ironing_2D_square
                                   COMMAND        contact_${example_name} --case square --num-steps 3
                                   NUM_MPI_TASKS  4)
            smith_add_example_test(NAME           contact_ironing_2D_circle
                                   COMMAND        contact_${example_name} --case circle --num-steps 3
                                   NUM_MPI_TASKS  4)
            smith_add_example_test(NAME           contact_ironing_2D_twisted
                                   COMMAND        contact_${example_name} --case twisted --num-steps 3
                                   NUM_MPI_TASKS  4)
        else()
            smith_add_example_test(NAME           contact_${example_name}
                                   COMMAND        contact_${example_name}
                                   NUM_MPI_TASKS  4)
        endif()

        if(SMITH_ENABLE_HIP)
            blt_add_target_compile_flags(TO contact_${example_name} FLAGS "-fgpu-rdc")
            blt_add_target_link_flags(TO contact_${example_name} FLAGS "-fgpu-rdc")
        endif()
        
    endforeach()

    install(
        FILES
            ${CONTACT_EXAMPLES_SOURCES}
        DESTINATION
            examples/smith/contact
    )

endif()
