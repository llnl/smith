# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause) 

set(mesh_test_depends smith_numerics smith_mesh_utils smith_boundary_conditions gtest)

set(mesh_parallel_test_sources
    mesh_test.cpp
    )

smith_add_tests( SOURCES       ${mesh_parallel_test_sources}
                 DEPENDS_ON    ${mesh_test_depends}
                 NUM_MPI_TASKS 2)

set(mesh_serial_test_sources
    mesh_generation.cpp
    )

smith_add_tests( SOURCES       ${mesh_serial_test_sources}
                 DEPENDS_ON    ${mesh_test_depends}
                 NUM_MPI_TASKS 1)
