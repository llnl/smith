# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

find_package(Doxygen COMPONENTS dot)

if(DOXYGEN_DOT_FOUND)
    set(DOXYGEN_HAVE_DOT "YES")
else()
    set(DOXYGEN_HAVE_DOT "NO")
endif()

# Check for pdflatex for latex generation
find_program(PDFLATEX_EXECUTABLE pdflatex)
if(PDFLATEX_EXECUTABLE)
    set(DOXYGEN_GENERATE_LATEX "YES")
else()
    set(DOXYGEN_GENERATE_LATEX "NO")
endif()

blt_add_doxygen_target( smith_doxygen )

# Required directory for doxygen to generate output to 
add_custom_target( build_doxygen_output_dir 
                   ALL 
                   COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/src/docs/html/doxygen
                 ) 

add_dependencies( smith_doxygen build_doxygen_output_dir )

