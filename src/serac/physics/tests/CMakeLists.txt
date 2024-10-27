# Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause) 

set(physics_test_depends serac_physics serac_mesh gtest)

set(physics_serial_test_sources
    beam_bending.cpp
    fit_test.cpp
    thermal_finite_diff.cpp
    thermal_statics_patch.cpp
    thermal_dynamics_patch.cpp
    solid_finite_diff.cpp
    solid_statics_patch.cpp
    solid_dynamics_patch.cpp
    dynamic_solid_adjoint.cpp
    quasistatic_solid_adjoint.cpp
    finite_element_vector_set_over_domain.cpp
    solid_multi_material.cpp
    )

serac_add_tests(SOURCES       ${physics_serial_test_sources}
                DEPENDS_ON    ${physics_test_depends}
                NUM_MPI_TASKS 1)

set(physics_parallel_test_sources
    lce_Brighenti_tensile.cpp
    lce_Bertoldi_lattice.cpp
    parameterized_thermomechanics_example.cpp
    parameterized_thermal.cpp
    solid.cpp
    solid_periodic.cpp
    solid_shape.cpp
    solid_robin_condition.cpp
    thermal_shape.cpp
    thermal_mechanics.cpp
    thermal_robin_condition.cpp
    dynamic_thermal_adjoint.cpp
    solid_reaction_adjoint.cpp
    thermal_nonlinear_solve.cpp
    )
blt_list_append(TO physics_parallel_test_sources ELEMENTS contact_patch.cpp contact_beam.cpp IF TRIBOL_FOUND)

serac_add_tests(SOURCES       ${physics_parallel_test_sources}
                DEPENDS_ON    ${physics_test_depends}
                NUM_MPI_TASKS 2)
