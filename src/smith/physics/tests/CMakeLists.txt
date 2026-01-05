# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause) 

set(physics_test_headers
    physics_test_utils.hpp
    )
    
smith_add_library(
    NAME        smith_physics_test_utils
    HEADERS     ${physics_test_headers}
    )
      
set(physics_test_depends smith_physics smith_physics_test_utils gtest)

set(physics_serial_test_sources
    beam_bending.cpp
    fit_test.cpp
    test_mesh_wrapper.cpp
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
    test_solid_weak_form.cpp
    test_heat_weak_form.cpp
    test_functional_weak_form.cpp
    thermomech_finite_diff.cpp
    thermomech_statics_patch.cpp
    test_kinematic_objective.cpp
    )

if (TRIBOL_FOUND)
    set(physics_serial_tribol_test_sources 
        test_contact_constraint.cpp
        contact_solid_adjoint.cpp)
    if (ENZYME_FOUND)
        list(APPEND physics_serial_tribol_test_sources 
                    contact_finite_diff.cpp
                    tribol_finite_diff.cpp)
    endif()
    list(APPEND physics_serial_test_sources
        ${physics_serial_tribol_test_sources})
endif()

if (SMITH_USE_DFEM)
    list(APPEND physics_serial_test_sources
                test_dfem_explicit_dynamics.cpp)
endif()

smith_add_tests(SOURCES       ${physics_serial_test_sources}
                DEPENDS_ON    ${physics_test_depends}
                NUM_MPI_TASKS 1)

if(SMITH_ENABLE_HIP AND TRIBOL_FOUND)
    foreach(filename ${physics_serial_tribol_test_sources})
        get_filename_component(test_name ${filename} NAME_WE)
        blt_add_target_compile_flags(TO ${test_name} FLAGS "-fgpu-rdc")
        blt_add_target_link_flags(TO ${test_name} FLAGS "-fgpu-rdc")
    endforeach()
endif()

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
set(physics_parallel_tribol_test_sources
    contact_patch.cpp
    contact_patch_tied.cpp
    contact_beam.cpp
    )
blt_list_append(TO       physics_parallel_test_sources
                ELEMENTS ${physics_parallel_tribol_test_sources}
                IF       TRIBOL_FOUND)

smith_add_tests(SOURCES       ${physics_parallel_test_sources}
                DEPENDS_ON    ${physics_test_depends}
                NUM_MPI_TASKS 2)

if(SMITH_ENABLE_HIP AND TRIBOL_FOUND)
    foreach(filename ${physics_parallel_tribol_test_sources})
        get_filename_component(test_name ${filename} NAME_WE)
        blt_add_target_compile_flags(TO ${test_name} FLAGS "-fgpu-rdc")
        blt_add_target_link_flags(TO ${test_name} FLAGS "-fgpu-rdc")
    endforeach()
endif()
