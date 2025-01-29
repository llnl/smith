# Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

if(TRIBOL_FOUND)
                    set(SourceFiles
                                        # active_cantilever_metrics
                                        # lattice_compression_contact
                                        # half_lattice_compression_contact
                                        # floating_structure_test
					# floating_self_contact_test
                    tupek_regularization
                    floating_regularized_self_contact
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
