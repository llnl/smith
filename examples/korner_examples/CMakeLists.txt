# Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
function (create_symlinks)
    # Do nothing if building in-source
    if (${CMAKE_CURRENT_BINARY_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
        return()
    endif()

    foreach (path_file ${ARGN})
        get_filename_component(folder ${path_file} PATH)

        # Create REAL folder
        file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${folder}")

        # Delete symlink if it exists
        file(REMOVE "${CMAKE_CURRENT_BINARY_DIR}/${path_file}")

        # Get OS dependent path to use in `execute_process`
        file(TO_NATIVE_PATH "${CMAKE_CURRENT_BINARY_DIR}/${path_file}" link)
        file(TO_NATIVE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/${path_file}" target)


        if (UNIX)
            set(command ln -s ${target} ${link})
        else()
            set(command cmd.exe /c mklink ${link} ${target})
        endif()

        execute_process(COMMAND ${command}
                        RESULT_VARIABLE result
                        ERROR_VARIABLE output)

        if (NOT ${result} EQUAL 0)
            message(FATAL_ERROR "Could not create symbolic link for: ${target} --> ${output}")
        endif()

    endforeach(path_file)
endfunction(create_symlinks)

if(TRIBOL_FOUND)
                    set(SourceFiles
                    # Rate_Dependent_Viscous_Lattice
                    double_ring_tensile
                    logpile_lattice
                    multi_tensile
                    )
                    foreach (source IN LISTS SourceFiles)

                        blt_add_executable( NAME       ${source} 
                                            SOURCES    ${source}.cpp
                                            OUTPUT_DIR ${EXAMPLE_OUTPUT_DIRECTORY}
                                            DEPENDS_ON  serac_physics serac_mesh
                                            )
endforeach()
    # blt_add_executable( NAME       active_cantilever_metrics 
    #                     SOURCES    active_cantilever_metrics.cpp
    #                     OUTPUT_DIR ${EXAMPLE_OUTPUT_DIRECTORY}
    #                     DEPENDS_ON  serac_physics serac_mesh
    #                     )
    # blt_add_executable( NAME       lattice_compression_contact 
    #                     SOURCES    lattice_compression_contact.cpp
    #                     OUTPUT_DIR ${EXAMPLE_OUTPUT_DIRECTORY}
    #                     DEPENDS_ON  serac_physics serac_mesh
    #                     )
    # blt_add_executable( NAME       half_lattice_compression_contact 
    #                     SOURCES    half_lattice_compression_contact.cpp
    #                     OUTPUT_DIR ${EXAMPLE_OUTPUT_DIRECTORY}
    #                     DEPENDS_ON  serac_physics serac_mesh
    #                     )
    # blt_add_executable( NAME        
    #                     SOURCES    half_lattice_compression_contact.cpp
    #                     OUTPUT_DIR ${EXAMPLE_OUTPUT_DIRECTORY}
    #                     DEPENDS_ON  serac_physics serac_mesh
    #                     )
endif()

file(GLOB_RECURSE files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.py *.dat *.xml)
create_symlinks(${files})
