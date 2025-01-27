# Copyright (c) 2019-2024, Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

set(functional_depends serac_mesh ${serac_device_depends})

# Add the library first
set(functional_headers
    differentiate_wrt.hpp
    boundary_integral_kernels.hpp
    element_restriction.hpp
    geometry.hpp
    geometric_factors.hpp
    domain.hpp
    domain_integral_kernels.hpp
    dual.hpp
    finite_element.hpp
    functional.hpp
    function_signature.hpp
    functional_qoi.inl
    integral.hpp
    interior_face_integral_kernels.hpp
    isotropic_tensor.hpp
    polynomials.hpp
    quadrature.hpp
    quadrature_data.hpp
    shape_aware_functional.hpp
    tensor.hpp
    tuple.hpp
    tuple_tensor_dual_functions.hpp
    typedefs.hpp
    )

set(functional_sources 
    domain.cpp 
    element_restriction.cpp 
    geometric_factors.cpp 
    quadrature_data.cpp)

set(functional_detail_headers
    detail/hexahedron_H1.inl
    detail/hexahedron_Hcurl.inl
    detail/hexahedron_L2.inl
    detail/metaprogramming.hpp
    detail/qoi.inl
    detail/quadrilateral_H1.inl
    detail/quadrilateral_Hcurl.inl
    detail/quadrilateral_L2.inl
    detail/segment_H1.inl
    detail/segment_Hcurl.inl
    detail/segment_L2.inl
    detail/tetrahedron_H1.inl
    detail/tetrahedron_L2.inl
    detail/triangle_H1.inl
    detail/triangle_L2.inl
    )

set(functional_cuda_headers
    boundary_integral_kernels.cuh
    domain_integral_kernels.cuh
    )

blt_list_append(TO functional_headers ELEMENTS ${functional_cuda_headers} IF SERAC_ENABLE_CUDA)

blt_add_library(
      NAME        serac_functional
      HEADERS     ${functional_headers} ${functional_detail_headers} 
      SOURCES     ${functional_sources}
      DEPENDS_ON  ${functional_depends}
      )


install(FILES ${functional_headers} DESTINATION include/serac/numerics/functional )
install(FILES ${functional_detail_headers} DESTINATION include/serac/numerics/functional/detail )

install(TARGETS              serac_functional
        EXPORT               serac-targets
        DESTINATION          lib
        )

if(SERAC_ENABLE_TESTS)
   add_subdirectory(tests)
endif()
