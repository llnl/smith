# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

set(differentiable_numerics_sources
    field_state.cpp
    differentiable_utils.cpp
    differentiable_physics.cpp
    lumped_mass_explicit_newmark_state_advancer.cpp
    )

set(differentiable_numerics_headers
    field_state.hpp
    state_advancer.hpp
    timestep_estimator.hpp
    explicit_dynamic_solve.hpp
    lumped_mass_weak_form.hpp
    differentiable_utils.hpp
    differentiable_physics.hpp
    lumped_mass_explicit_newmark_state_advancer.hpp
    )

set(differentiable_numerics_depends
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
    