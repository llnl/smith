# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

set(boundary_conditions_headers
    boundary_condition.hpp
    boundary_condition_helper.hpp
    boundary_condition_manager.hpp
    components.hpp
    )

set(boundary_conditions_sources
    boundary_condition.cpp
    boundary_condition_helper.cpp
    boundary_condition_manager.cpp
    )

set(boundary_conditions_depends smith_infrastructure smith_state)

smith_add_library(
    NAME        smith_boundary_conditions
    HEADERS     ${boundary_conditions_headers}
    SOURCES     ${boundary_conditions_sources}
    DEPENDS_ON  ${boundary_conditions_depends}
    )

# Unified header is not needed in this component

install(FILES ${boundary_conditions_headers} DESTINATION include/smith/physics/boundary_conditions )

install(TARGETS              smith_boundary_conditions
        EXPORT               smith-targets
        DESTINATION          lib
        )

if(SMITH_ENABLE_TESTS)
    add_subdirectory(tests)
endif()
