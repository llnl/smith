# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

set(state_headers
    finite_element_vector.hpp
    finite_element_state.hpp
    finite_element_dual.hpp
    state_manager.hpp
    )

set(state_sources
    finite_element_vector.cpp
    finite_element_state.cpp
    state_manager.cpp
    )

set(state_depends smith_infrastructure smith_functional)

smith_add_library(
    NAME        smith_state
    HEADERS     ${state_headers}
    SOURCES     ${state_sources}
    DEPENDS_ON  ${state_depends}
    )

smith_write_unified_header(
    NAME    state
    HEADERS ${state_headers}
    )

install(FILES ${state_headers} DESTINATION include/smith/physics/state )

install(TARGETS              smith_state
        EXPORT               smith-targets
        DESTINATION          lib
        )

if(SMITH_ENABLE_TESTS)
    add_subdirectory(tests)
endif()
