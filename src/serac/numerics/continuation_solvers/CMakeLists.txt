# Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

# Add the library first
set(continuation_solvers_sources
  problems/NLMCProblems.cpp
  problems/OptProblems.cpp
  solvers/HomotopySolver.cpp
  solvers/IPSolver.cpp
  utilities.cpp)

set(continuation_solvers_headers
  problems/NLMCProblems.hpp
  problems/OptProblems.hpp
  problems/Problems.hpp
  solvers/Solvers.hpp
  solvers/HomotopySolver.hpp
  solvers/IPSolver.hpp
  utilities.hpp)

set(continuation_solvers_depends
  blt::mpi
  mfem
  )

blt_add_library(
  NAME       serac_continuation_solvers
  SOURCES    ${continuation_solvers_sources}
  HEADERS    ${continuation_solvers_headers}
  DEPENDS_ON ${continuation_solvers_depends})

#target_include_directories(continuation_solvers PUBLIC
#  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
#  $<INSTALL_INTERFACE:include/continuation_solvers>)

set(continuation_solvers_problem_headers
  problems/NLMCProblems.hpp
  problems/OptProblems.hpp
  problems/Problems.hpp)

install(FILES ${continuation_solvers_problem_headers} DESTINATION include/serac/numerics/continuation_solvers/problems)

set(continuation_solvers_solver_headers
  solvers/Solvers.hpp
  solvers/HomotopySolver.hpp
  solvers/IPSolver.hpp)

install(FILES ${continuation_solvers_solver_headers} DESTINATION include/serac/numerics/continuation_solvers/solvers)

install(FILES utilities.hpp DESTINATION include/serac/numerics/continuation_solvers)

install(TARGETS     serac_continuation_solvers
        EXPORT      serac-targets
        DESTINATION lib
        )
