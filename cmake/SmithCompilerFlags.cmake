# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

if(ENABLE_ASAN)
    message(STATUS "AddressSanitizer is ON (ENABLE_ASAN)")
    foreach(_flagvar CMAKE_C_FLAGS CMAKE_CXX_FLAGS CMAKE_EXE_LINKER_FLAGS)
        string(APPEND ${_flagvar} " -fsanitize=address -fno-omit-frame-pointer")
    endforeach()
endif()

# Need to add symbols to dynamic symtab in order to be visible from stacktraces
string(APPEND CMAKE_EXE_LINKER_FLAGS " -rdynamic")

# ROCm's amdclang++ can drop its implicit C++ standard library on HIP links
# once Fortran runtime libraries are introduced. On this platform the shared
# libstdc++ linker script still points at the old system runtime, so use the
# compiler's GCC 13 static archives instead.
if(ENABLE_HIP AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER MATCHES "amdclang\\+\\+")
    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libstdc++.a
        OUTPUT_VARIABLE _smith_libstdcxx_static
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libstdc++_nonshared.a
        OUTPUT_VARIABLE _smith_libstdcxx_nonshared
        OUTPUT_STRIP_TRAILING_WHITESPACE)

    if(EXISTS "${_smith_libstdcxx_static}" AND EXISTS "${_smith_libstdcxx_nonshared}")
        string(APPEND CMAKE_CXX_STANDARD_LIBRARIES
            " ${_smith_libstdcxx_static} ${_smith_libstdcxx_nonshared}")
    endif()
endif()

# Apple ld warns about duplicate -l flags when the same library is reachable
# via multiple dependency paths (common with Spack-built CMake targets that use
# raw -l strings instead of imported targets). Suppress the spurious warning.
if(APPLE)
    string(APPEND CMAKE_EXE_LINKER_FLAGS " -Wl,-no_warn_duplicate_libraries")
endif()

# Prevent unused -Xlinker arguments on Lassen Clang-10
if(DEFINED ENV{SYS_TYPE} AND "$ENV{SYS_TYPE}" STREQUAL "blueos_3_ppc64le_ib_p9")
    string(APPEND CMAKE_EXE_LINKER_FLAGS " -Wno-unused-command-line-argument")
endif()

# Enable warnings for things not covered by -Wall -Wextra
set(_extra_flags "-Wshadow -Wdouble-promotion -Wconversion -Wundef -Wnull-dereference -Wold-style-cast")
blt_append_custom_compiler_flag(FLAGS_VAR CMAKE_CXX_FLAGS DEFAULT ${_extra_flags})

# Clang specific warnings
# Note: pedantic is a gcc flag but throws a false positive in src/smith/numerics/petsc_solvers.cpp
blt_append_custom_compiler_flag(FLAGS_VAR CMAKE_CXX_FLAGS CLANG "-Wpedantic -Wunused-private-field")
