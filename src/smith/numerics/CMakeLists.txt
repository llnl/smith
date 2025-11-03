# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

add_subdirectory(functional)

set(numerics_headers
    equation_solver.hpp
    odes.hpp
    solver_config.hpp
    stdfunction_operator.hpp
    petsc_solvers.hpp
    trust_region_solver.hpp
    dense_petsc.hpp
    )

set(numerics_sources
    equation_solver.cpp
    trust_region_solver.cpp
    odes.cpp
    petsc_solvers.cpp
    )

set(numerics_depends smith_infrastructure smith_functional smith_boundary_conditions)

smith_add_library(
    NAME        smith_numerics
    HEADERS     ${numerics_headers}
    SOURCES     ${numerics_sources}
    DEPENDS_ON  ${numerics_depends}
    )

smith_write_unified_header(
    NAME    numerics
    HEADERS solver_config.hpp
    )

install(FILES ${numerics_headers} DESTINATION include/smith/numerics )

install(TARGETS              smith_numerics
        EXPORT               smith-targets
        DESTINATION          lib
        )

if(SMITH_ENABLE_TESTS)
    add_subdirectory(tests)
endif()
