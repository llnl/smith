# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

set(differentiable_numerics_test_headers
    differentiable_test_utils.hpp
    )

smith_add_library(
    NAME        differentiable_numerics_test_utils
    HEADERS     ${differentiable_numerics_test_headers}
    )

set(differentiable_numerics_test_depends smith_physics smith_differentiable_numerics differentiable_numerics_test_utils gtest)

set(differentiable_numerics_test_source
    test_field_state.cpp
    test_explicit_dynamics.cpp
    test_solid_mechanics_state_advancer.cpp
    differentiable_test_utils.cpp
    )

smith_add_tests( SOURCES       ${differentiable_numerics_test_source}
                 DEPENDS_ON    ${differentiable_numerics_test_depends}
                 NUM_MPI_TASKS 1)
