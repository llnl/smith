# Copyright (c) Lawrence Livermore National Security, LLC and
# other Serac Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

set(serac_gretl_sources
      data_store.cpp
      state_base.cpp
      vector_state.cpp
    )

set(serac_gretl_headers
      checkpoint.hpp
      data_store.hpp
      test_utils.hpp
      state_base.hpp
      state.hpp
      create_state.hpp
      upstream_state.hpp
      double_state.hpp
      vector_state.hpp
      print_utils.hpp
    )

blt_add_library(
    NAME        serac_gretl
    SOURCES     ${serac_gretl_sources}
    HEADERS     ${serac_gretl_headers}
    )

target_include_directories(serac_gretl PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
    $<INSTALL_INTERFACE:include>
    )

install(FILES ${serac_gretl_headers} DESTINATION include/serac/gretl )

install(TARGETS              serac_gretl
        EXPORT               serac-targets
        DESTINATION          lib
        )

if(SERAC_ENABLE_TESTS)
    add_subdirectory(tests)
endif()   
    
