# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

set(differentiable_numerics_test_depends smith_physics smith_differentiable_numerics gtest)

set(differentiable_numerics_test_source
    test_solver_convergence.cpp
    test_field_state.cpp
    test_solid_dynamics.cpp
    test_explicit_dynamics.cpp
    test_porous_heat_sink.cpp
    test_thermo_mechanics.cpp
    test_thermal_static.cpp
    test_solid_static_with_internal_vars.cpp
    test_multiphysics_time_integrator.cpp
    )

smith_add_tests( SOURCES       ${differentiable_numerics_test_source}
                 DEPENDS_ON    ${differentiable_numerics_test_depends}
                 NUM_MPI_TASKS 1)
