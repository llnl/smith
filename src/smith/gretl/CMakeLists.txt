# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

set(smith_gretl_sources
      data_store.cpp
      state_base.cpp
      vector_state.cpp
    )

set(smith_gretl_headers
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
    NAME        smith_gretl
    SOURCES     ${smith_gretl_sources}
    HEADERS     ${smith_gretl_headers}
    )

target_include_directories(smith_gretl PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
    $<INSTALL_INTERFACE:include>
    )

install(FILES ${smith_gretl_headers} DESTINATION include/smith/gretl )

install(TARGETS              smith_gretl
        EXPORT               smith-targets
        DESTINATION          lib
        )

if(SMITH_ENABLE_TESTS)
    add_subdirectory(tests)
endif()   
    
