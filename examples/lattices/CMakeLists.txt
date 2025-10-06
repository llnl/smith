# Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
# set(py2cpp_DIR "/g/g20/korner1/test/py2cpp/install/lib64/cmake/py2cpp/" CACHE STRING "")
# message(${py2cpp_DIR})

# find_package(py2cpp CONFIG REQUIRED)
# include(FetchContent)
# FetchContent_Declare(
#     py2cpp
#     GIT_REPOSITORY ssh://git@czgitlab.llnl.gov:7999/korner1/py2cpp.git
#     GIT_TAG main
# )
# FetchContent_MakeAvailable(py2cpp)
# target_compile_options(py2cpp PRIVATE -Wno-old-style-cast -Wno-unused-variable)


    # blt_import_library(
    #     NAME          py2cpp
    #     INCLUDES      ${SERAC_SOURCE_DIR}/py2cpp/install/include
    #     LIBRARIES     ${SERAC_SOURCE_DIR}/py2cpp/install/lib64/libpy2cpp.a
    #     TREAT_INCLUDES_AS_SYSTEM ON
    #     EXPORTABLE    ON)
    #     target_link_libraries(py2cpp INTERFACE Python3::Python Python3::NumPy)
    #     target_include_directories(py2cpp SYSTEM INTERFACE
    #       ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})

    # blt_import_library(
    #     NAME          sol2
    #     INCLUDES      ${SERAC_SOURCE_DIR}/sol2/include
    #     LIBRARIES     ${SERAC_SOURCE_DIR}/sol2/sol.hpp
    #     TYPE          INTERFACE
    #     TREAT_INCLUDES_AS_SYSTEM ON
    #     EXPORTABLE    ON)
    #     message("SOL2 Dir: ${SOL2_DIR}")
    include(FetchContent)
    FetchContent_Declare(sol2
  GIT_REPOSITORY https://github.com/ThePhD/sol2.git
  GIT_TAG        develop
)

FetchContent_Populate(sol2)
    add_library(sol2 INTERFACE)
    target_include_directories( sol2 INTERFACE ${sol2_SOURCE_DIR}/include)

    # blt_add_library(
    #     NAME sol2 
    #     TYPE INTERFACE
    #     TREAT_INCLUDES_AS_SYSTEM ON
    #     )
    # target_include_directories(sol2 INTERFACE ${SERAC_SOURCE_DIR}/sol2/include)
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
            hole_lattice
            plastic_hole_lattice
	    # thermomech
            thermomech_test
            )
            foreach (source IN LISTS SourceFiles)

                blt_add_executable( NAME       ${source} 
                                    SOURCES    ${source}.cpp
                                    OUTPUT_DIR ${EXAMPLE_OUTPUT_DIRECTORY}
                                    DEPENDS_ON  serac_physics serac_mesh_utils sol2
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
