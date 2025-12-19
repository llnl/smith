# Copyright (c) Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

if (TRIBOL_FOUND AND STRUMPACK_DIR AND SMITH_ENABLE_CONTINUATION)
    blt_add_executable( NAME       two_blocks_example
                        SOURCES    two_blocks.cpp
                        OUTPUT_DIR ${EXAMPLE_OUTPUT_DIRECTORY}
                        DEPENDS_ON smith continuation_solver
		      )
    smith_add_example_test(NAME          two_blocks
                           COMMAND       two_blocks_example
                           NUM_MPI_TASKS 4)
    install(
        FILES
            two_blocks.cpp
        DESTINATION
            examples/smith/two_blocks
    )
endif()
