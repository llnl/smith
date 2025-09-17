# Copyright (c) Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

if (TRIBOL_FOUND AND STRUMPACK_DIR AND SERAC_ENABLE_CONTINUATION)
    blt_add_executable( NAME       constraint_twist_example
  	                SOURCES    constraint_twist.cpp
                        OUTPUT_DIR ${EXAMPLE_OUTPUT_DIRECTORY}
                        DEPENDS_ON serac_physics serac_mesh_utils continuation_solver
		      )
    serac_add_example_test(NAME          constraint_twist
                           COMMAND       constraint_twist_example
                           NUM_MPI_TASKS 4)
    install(
	FILES
	    constraint_twist.cpp
	DESTINATION
	    examples/serac/constraint_twist
    )
endif()
