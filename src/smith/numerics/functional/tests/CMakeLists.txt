# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

set(functional_test_depends gtest smith_functional ${functional_depends})

# This test is all constexpr-evaluated, so it doesn't
# actually need to run, if it compiles without error, the tests have passed
smith_add_executable(NAME        tensor_unit_tests
                     SOURCES     tensor_unit_tests.cpp
                     OUTPUT_DIR  ${TEST_OUTPUT_DIRECTORY}
                     DEPENDS_ON  ${functional_test_depends}
                     FOLDER      smith/tests)

# Then add the examples/tests
set(functional_serial_test_sources
    functional_shape_derivatives.cpp
    simplex_basis_function_unit_tests.cpp
    element_restriction_tests.cpp
    dg_restriction_operators.cpp
    bug_boundary_qoi.cpp
    domain_tests.cpp
    geometric_factors_tests.cpp
    hcurl_unit_tests.cpp
    functional_tet_quality.cpp
    test_tensor_ad.cpp
    tuple_arithmetic_unit_tests.cpp
    test_newton.cpp)

smith_add_tests(SOURCES    ${functional_serial_test_sources}
                DEPENDS_ON ${functional_test_depends})

set_source_files_properties(functional_tet_quality PROPERTIES LANGUAGE CXX)

# Then add the examples/tests
set(functional_parallel_test_sources
    #functional_basic_hcurl.cpp
    functional_with_domain.cpp
    functional_basic_h1_scalar.cpp
    functional_basic_h1_vector.cpp
    functional_basic_dg.cpp
    functional_multiphysics.cpp
    functional_qoi.cpp
    functional_nonlinear.cpp
    functional_boundary_test.cpp
    functional_comparisons.cpp
    functional_comparison_L2.cpp
    dg_ghost_face_index.cpp
    )

smith_add_tests(SOURCES       ${functional_parallel_test_sources}
                DEPENDS_ON    ${functional_test_depends}
                NUM_MPI_TASKS 4 )

foreach(filename ${functional_parallel_test_sources})
    set_source_files_properties(${filename} PROPERTIES LANGUAGE CXX)
endforeach()

target_link_libraries(bug_boundary_qoi PUBLIC smith_physics)

if(SMITH_ENABLE_CUDA)

    set(functional_cuda_test_sources 
        tensor_unit_tests_cuda.cu 
#        some of the GPU functionality is temporarily disabled to 
#        help incrementally roll-out the variadic implementation of Functional
#        TODO: re-enable GPU kernels in a follow-up PR
#        functional_comparisons_cuda.cu
       )

    smith_add_tests( SOURCES    ${functional_cuda_test_sources}
                     DEPENDS_ON ${functional_test_depends})

endif()
