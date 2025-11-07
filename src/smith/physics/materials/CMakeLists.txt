# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

set(materials_sources
    hardening_input.cpp
    thermal_material_input.cpp
    solid_material_input.cpp)

set(materials_headers
    green_saint_venant_thermoelastic.hpp
    hardening_input.hpp
    liquid_crystal_elastomer.hpp
    material_verification_tools.hpp
    parameterized_solid_material.hpp
    parameterized_thermal_material.hpp
    parameterized_thermoelastic_material.hpp
    solid_material_input.hpp
    solid_material.hpp
    thermal_material_input.hpp
    thermal_material.hpp
    )

set(materials_depends
    smith_infrastructure
    smith_functional
    )

smith_add_library(
   NAME        smith_physics_materials
   SOURCES     ${materials_sources}
   HEADERS     ${materials_headers}
   DEPENDS_ON  ${materials_depends}
   )

smith_write_unified_header(
    NAME    materials
    HEADERS ${materials_headers}
    )

install(FILES ${materials_headers} DESTINATION include/smith/physics/materials )

install(TARGETS              smith_physics_materials
        EXPORT               smith-targets
        DESTINATION          lib
        )

if(SMITH_ENABLE_TESTS)
    add_subdirectory(tests)
endif()
