# Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

set(refactor_test_dependencies serac_refactor serac_boundary_conditions gtest)

set(refactor_serial_test_sources
    h1_evaluation_tests.cpp
    hcurl_evaluation_tests.cpp

    integrate_residual_H1_flux_tests.cpp
    integrate_residual_H1_source_tests.cpp
    integrate_residual_H1v_flux_tests.cpp
    integrate_residual_Hcurl_flux_tests.cpp
    integrate_residual_Hcurl_source_tests.cpp

    nodal_coordinates_and_directions_tests.cpp
)

serac_add_tests( SOURCES       ${refactor_serial_test_sources}
                 DEPENDS_ON    ${refactor_test_dependencies}
                 NUM_MPI_TASKS 1)
