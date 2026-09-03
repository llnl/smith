# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

add_subdirectory(functional)
set(numerics_headers
    equation_solver.hpp
    nonlinear_convergence.hpp
    odes.hpp
    solver_config.hpp
    state_dependent_solver.hpp
    solver_with_preconditioner.hpp
    stdfunction_operator.hpp
    petsc_solvers.hpp
    steihaug_toint_cg.hpp
    trust_region_solver.hpp
    trust_region_subspace_cache.hpp
    block_preconditioner.hpp
    )

set(numerics_sources
    equation_solver.cpp
    steihaug_toint_cg.cpp
    mfem_trust_region_subspace.cpp
    trust_region_subspace_cache.cpp
    nonlinear_convergence.cpp
    odes.cpp
    petsc_solvers.cpp
    block_preconditioner.cpp
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
