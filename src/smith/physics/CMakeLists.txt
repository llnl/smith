# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

add_subdirectory(materials)
add_subdirectory(state)
add_subdirectory(boundary_conditions)
add_subdirectory(contact)

set(physics_sources
    base_physics.cpp
    mesh.cpp
    solid_mechanics.cpp
    heat_transfer_input.cpp
    solid_mechanics_input.cpp
    thermomechanics_input.cpp
    )

set(physics_headers
    base_physics.hpp
    mesh.hpp
    common.hpp
    dfem_mass_weak_form.hpp
    dfem_solid_weak_form.hpp
    dfem_weak_form.hpp
    field_types.hpp
    fit.hpp
    heat_transfer.hpp
    heat_transfer_input.hpp
    solid_mechanics.hpp
    solid_mechanics_contact.hpp
    solid_mechanics_input.hpp
    thermomechanics.hpp
    thermomechanics_input.hpp
    thermomechanics_monolithic.hpp
    weak_form.hpp
    functional_weak_form.hpp
    solid_weak_form.hpp
    heat_transfer_weak_form.hpp
    scalar_objective.hpp
    functional_objective.hpp
    constraint.hpp
    contact_constraint.hpp
    )

set(physics_depends
    smith_state
    smith_boundary_conditions
    smith_contact
    smith_numerics
    smith_physics_materials
    )

smith_add_library(
    NAME        smith_physics
    SOURCES     ${physics_sources}
    HEADERS     ${physics_headers}
    DEPENDS_ON  ${physics_depends}
    )

smith_write_unified_header(
    NAME    physics
    HEADERS ${physics_headers}
    )

install(FILES ${physics_headers} DESTINATION include/smith/physics )

install(TARGETS              smith_physics
        EXPORT               smith-targets
        DESTINATION          lib
        )

if(SMITH_ENABLE_TESTS)
    add_subdirectory(tests)
endif()
    
if(SMITH_ENABLE_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
