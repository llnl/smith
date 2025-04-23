# Copyright (c) Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

blt_add_doxygen_target( serac_doxygen )

# Required directory for doxygen to generate output to 
add_custom_target( build_doxygen_output_dir 
                   ALL 
                   COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/src/docs/html/doxygen
                 ) 

add_dependencies( serac_doxygen build_doxygen_output_dir )

