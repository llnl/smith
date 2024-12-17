# Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

set(refactor_containers_headers
    ndarray.hpp
    stack_array.hpp
)

set(refactor_containers_sources
    memory.cpp
    relative_error.cpp
)

set(refactor_containers_depends serac_infrastructure)

blt_add_library(
    NAME        serac_refactor_containers
    HEADERS     ${refactor_containers_headers}
    SOURCES     ${refactor_containers_sources}
    DEPENDS_ON  ${refactor_containers_depends}
    )

install(FILES ${refactor_containers_headers} DESTINATION include/serac/refactor/containers )

install(TARGETS              serac_refactor_containers
        EXPORT               serac-targets
        DESTINATION          lib
        )

if(SERAC_ENABLE_TESTS)
    add_subdirectory(tests)
endif()
