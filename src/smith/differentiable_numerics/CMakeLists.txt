# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

set(differentiable_numerics_sources
    field_state.cpp
    differentiable_physics.cpp
    lumped_mass_explicit_newmark_state_advancer.cpp
    solid_mechanics_state_advancer.cpp
    differentiable_solver.cpp
    nonlinear_solve.cpp
    evaluate_objective.cpp
    dirichlet_boundary_conditions.cpp
    )

set(differentiable_numerics_headers
    field_state.hpp
    state_advancer.hpp
    reaction.hpp
    differentiable_solver.hpp
    differentiable_physics.hpp
    timestep_estimator.hpp
    explicit_dynamic_solve.hpp
    lumped_mass_weak_form.hpp
    lumped_mass_explicit_newmark_state_advancer.hpp
    solid_mechanics_state_advancer.hpp
    nonlinear_solve.hpp
    evaluate_objective.hpp
    dirichlet_boundary_conditions.hpp
    time_integration_rule.hpp
    time_discretized_weak_form.hpp
    differentiable_solid_mechanics.hpp
    paraview_writer.hpp
    differentiable_test_utils.hpp
    )

set(differentiable_numerics_depends
    gretl
    smith_state
    smith_boundary_conditions
    smith_contact
    smith_numerics
    smith_physics
    )

blt_add_library(
    NAME        smith_differentiable_numerics
    SOURCES     ${differentiable_numerics_sources}
    HEADERS     ${differentiable_numerics_headers}
    DEPENDS_ON  ${differentiable_numerics_depends}
    )

smith_write_unified_header(
    NAME    differentiable_numerics
    HEADERS ${differentiable_numerics_headers}
    )

install(FILES ${differentiable_numerics_headers} DESTINATION include/smith/differentiable_numerics )

install(TARGETS              smith_differentiable_numerics
        EXPORT               smith-targets
        DESTINATION          lib
        )

if(SMITH_ENABLE_TESTS)
    add_subdirectory(tests)
endif()
