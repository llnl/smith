# Copyright (c) 2019-2025, Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

set(differentiable_numerics_test_depends serac_physics serac_gretl serac_differentiable_numerics gtest)

set(differentiable_numerics_tests
    test_smith_mechanics.cpp
    test_smith_hydro.cpp
    serac_qoi_integrators.hpp
    )

serac_add_tests( SOURCES       ${parallel_solver_tests}
                 DEPENDS_ON    ${physics_parallel_test_depends}
                 NUM_MPI_TASKS 2)

set(physics_serial_test_depends
    smith_helmholtz_filter_serial.cpp
    smith_porous_finite_diff.cpp)

serac_add_tests( SOURCES       ${serial_solver_tests}
                 DEPENDS_ON    ${physics_serial_test_depends}
                 NUM_MPI_TASKS 1)
