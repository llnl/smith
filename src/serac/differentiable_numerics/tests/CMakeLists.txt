# Copyright (c) 2019-2025, Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

set(differentiable_numerics_test_depends serac_physics serac_gretl serac_differentiable_numerics gtest)

set(differentiable_numerics_tests
    test_explicit_dynamics.cpp
    test_field_state.cpp
    paraview_helper.hpp
    )

serac_add_tests( SOURCES       ${differentiable_numerics_tests}
                 DEPENDS_ON    ${differentiable_numerics_test_depends}
                 NUM_MPI_TASKS 1)

