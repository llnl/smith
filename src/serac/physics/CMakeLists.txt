# Copyright (c) Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
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
    common.hpp
    mesh.hpp
    fit.hpp
    heat_transfer.hpp
    heat_transfer_input.hpp
    solid_mechanics.hpp
    solid_mechanics_contact.hpp
    solid_mechanics_input.hpp
    thermomechanics.hpp
    thermomechanics_input.hpp
    thermomechanics_monolithic.hpp
    residual.hpp
    functional_residual.hpp
    solid_residual.hpp
    objective.hpp
    motion_objectives.hpp
    )

set(physics_depends
    serac_state
    serac_boundary_conditions
    serac_contact
    serac_numerics
    serac_physics_materials
    )

blt_add_library(
    NAME        serac_physics
    SOURCES     ${physics_sources}
    HEADERS     ${physics_headers}
    DEPENDS_ON  ${physics_depends}
    )

install(FILES ${physics_headers} DESTINATION include/serac/physics )

install(TARGETS              serac_physics
        EXPORT               serac-targets
        DESTINATION          lib
        )

if(SERAC_ENABLE_TESTS)
    add_subdirectory(tests)
endif()
    
if(SERAC_ENABLE_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
