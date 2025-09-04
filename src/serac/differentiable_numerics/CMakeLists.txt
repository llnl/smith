# Copyright (c) Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

set(differentiable_numerics_sources
    state_advancer.cpp
    field_state.cpp
    mechanics.cpp
    )

set(differentiable_numerics_headers
    field_state.hpp
    state_advancer.hpp
    timestep_estimator.hpp
    mechanics.hpp
    explicit_dynamic_solve.hpp
    lumped_mass_weak_form.hpp
    differentiable_utils.hpp
    )

set(differentiable_numerics_depends
    serac_state
    serac_boundary_conditions
    serac_contact
    serac_numerics
    serac_physics
    serac_gretl
    )

blt_add_library(
    NAME        serac_differentiable_numerics
    SOURCES     ${differentiable_numerics_sources}
    HEADERS     ${differentiable_numerics_headers}
    DEPENDS_ON  ${differentiable_numerics_depends}
    )

serac_write_unified_header(
    NAME    differentiable_numerics
    HEADERS ${differentiable_numerics_headers}
    )

install(FILES ${differentiable_numerics_headers} DESTINATION include/serac/differentiable_numerics )

install(TARGETS              serac_differentiable_numerics
        EXPORT               serac-targets
        DESTINATION          lib
        )

if(SERAC_ENABLE_TESTS)
    add_subdirectory(tests)
endif()
    