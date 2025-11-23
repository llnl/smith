# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

##------------------------------------------------------------------------------
## smith_add_executable(
##                     NAME        <name>
##                     SOURCES     [source1 [source2 ...]]
##                     HEADERS     [header1 [header2 ...]]
##                     INCLUDES    [dir1 [dir2 ...]]
##                     DEFINES     [define1 [define2 ...]]
##                     DEPENDS_ON  [dep1 [dep2 ...]]
##                     OUTPUT_DIR  [dir]
##                     OUTPUT_NAME [name]
##                     FOLDER      [name])
##
## Wrapper around blt_add_executable
##------------------------------------------------------------------------------
macro(smith_add_executable)

    set(options )
    set(singleValueArgs NAME OUTPUT_DIR OUTPUT_NAME FOLDER)
    set(multiValueArgs HEADERS SOURCES INCLUDES DEFINES DEPENDS_ON)

    # Parse the arguments to the macro
    cmake_parse_arguments(arg
        "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN})

    blt_add_executable(NAME        ${arg_NAME}
                       SOURCES     ${arg_SOURCES}
                       HEADERS     ${arg_HEADERS}
                       INCLUDES    ${arg_INCLUDES}
                       DEFINES     ${arg_DEFINES}
                       DEPENDS_ON  ${arg_DEPENDS_ON} ${smith_device_depends}
                       OUTPUT_DIR  ${arg_OUTPUT_DIR}
                       OUTPUT_NAME ${arg_OUTPUT_NAME}
                       FOLDER      ${arg_FOLDER})

endmacro(smith_add_executable)

##------------------------------------------------------------------------------
## smith_add_library(
##                  NAME         <libname>
##                  SOURCES      [source1 [source2 ...]]
##                  HEADERS      [header1 [header2 ...]]
##                  INCLUDES     [dir1 [dir2 ...]]
##                  DEFINES      [define1 [define2 ...]]
##                  DEPENDS_ON   [dep1 ...]
##                  OUTPUT_NAME  [name]
##                  OUTPUT_DIR   [dir]
##                  SHARED       [TRUE | FALSE]
##                  OBJECT       [TRUE | FALSE]
##                  CLEAR_PREFIX [TRUE | FALSE]
##                  FOLDER       [name])
##
## Wrapper around blt_add_library, to conveniently add device dependencies.
##------------------------------------------------------------------------------
macro(smith_add_library)

  set(options)
  set(singleValueArgs
      NAME
      OUTPUT_NAME
      OUTPUT_DIR
      SHARED
      OBJECT
      CLEAR_PREFIX
      FOLDER)
  set(multiValueArgs SOURCES HEADERS INCLUDES DEFINES DEPENDS_ON)

  cmake_parse_arguments(arg "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN})

  blt_add_library(
    NAME ${arg_NAME}
    SOURCES ${arg_SOURCES}
    HEADERS ${arg_HEADERS}
    INCLUDES ${arg_INCLUDES}
    DEFINES ${arg_DEFINES}
    DEPENDS_ON ${arg_DEPENDS_ON} ${smith_device_depends}
    OUTPUT_DIR ${arg_OUTPUT_DIR}
    OUTPUT_NAME ${arg_OUTPUT_NAME}
    FOLDER ${arg_FOLDER})

endmacro(smith_add_library)

##------------------------------------------------------------------------------
## smith_add_example_test(NAME              [name]
##                        COMMAND           [command]
##                        NUM_MPI_TASKS     [n]
##                        NUM_OMP_THREADS   [n]
##                        CONFIGURATIONS    [config1 [config2...]]
##                        WORKING_DIRECTORY [dir]
##                        TIMEOUT           [time_seconds])
##
## Wrapper for blt_add_test designed to run lengthier example tests.
##
## To run examples, use `make run_examples` custom CMake target.
##
## Unlike smith_add_tests, this macro takes in a pre-existing executable.
##------------------------------------------------------------------------------
macro(smith_add_example_test)

    set(options)
    set(singleValueArgs NAME NUM_MPI_TASKS NUM_OMP_THREADS WORKING_DIRECTORY)
    set(multiValueArgs COMMAND CONFIGURATIONS)

    # Parse the arguments to the macro
    cmake_parse_arguments(arg
        "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN} )

    if (NOT DEFINED arg_TIMEOUT)
        set(_timeout 4800)
    else()
        set(_timeout ${arg_TIMEOUT})
    endif()

    # The 'CONFIGURATIONS Example' line excludes examples
    # from the general list of tests
    blt_add_test(NAME              ${arg_NAME}
                 COMMAND           ${arg_COMMAND}
                 NUM_MPI_TASKS     ${arg_NUM_MPI_TASKS}
                 NUM_OMP_THREADS   ${arg_NUM_OMP_THREADS}
                 CONFIGURATIONS    Example ${arg_CONFIGURATIONS}
                 WORKING_DIRECTORY ${arg_WORKING_DIRECTORY})

    set_tests_properties(${arg_NAME} PROPERTIES TIMEOUT ${_timeout})

    # The 'LABELS Example' prevents regular tests from
    # running when running example custom target
    set_tests_properties(${arg_NAME} PROPERTIES LABELS "Example")

    unset(_timeout)

endmacro(smith_add_example_test)

##------------------------------------------------------------------------------
## Adds code checks for all cpp/hpp files recursively under the current directory
## that regex match INCLUDES and excludes any files that regex match EXCLUDES
## 
## This creates the following parent build targets:
##  check - Runs a non file changing style check and CppCheck
##  style - In-place code formatting
##
## Creates various child build targets that follow this pattern:
##  smith_<check|style>
##  smith_<cppcheck|clangformat>_<check|style>
##
## This also creates targets for running clang-tidy on the src/ and test/
## directories, with a more permissive set of checks for the tests,
## called smith_guidelines_check and smith_guidelines_check_tests, respectively
##------------------------------------------------------------------------------
macro(smith_add_code_checks)

    set(options)
    set(singleValueArgs PREFIX)
    set(multiValueArgs)

    # Parse the arguments to the macro
    cmake_parse_arguments(arg
         "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN})

    # Create file globbing expressions that only include directories that contain source
    set(_base_dirs "tests" "src" "examples")
    # Note: any extensions added here should also be added to BLT's lists in CMakeLists.txt
    set(_ext_expressions "*.cpp" "*.hpp" "*.inl" "*.cuh" "*.cu" "*.cpp.in" "*.hpp.in")

    set(_glob_expressions)
    foreach(_exp ${_ext_expressions})
        foreach(_base_dir ${_base_dirs})
            list(APPEND _glob_expressions "${PROJECT_SOURCE_DIR}/${_base_dir}/${_exp}")
        endforeach()
    endforeach()

    # Glob for list of files to run code checks on
    set(_sources)
    file(GLOB_RECURSE _sources ${_glob_expressions})

    blt_add_code_checks(PREFIX          ${arg_PREFIX}
                        SOURCES         ${_sources}
                        CLANGFORMAT_CFG_FILE ${PROJECT_SOURCE_DIR}/.clang-format
                        CPPCHECK_FLAGS  --enable=all --inconclusive)


    set(_src_sources)
    file(GLOB_RECURSE _src_sources "src/*.cpp" "src/*.hpp" "src/*.inl")
    list(FILTER _src_sources EXCLUDE REGEX ".*/tests/.*pp")

    blt_add_clang_tidy_target(NAME              ${arg_PREFIX}_guidelines_check
                              CHECKS            "clang-analyzer-*,clang-analyzer-cplusplus*,cppcoreguidelines-*"
                              SRC_FILES         ${_src_sources})

    # Create list of recursive test directory glob expressions
    # NOTE: GLOB operator ** did not appear to be supported by cmake and did not recursively find test subdirectories
    # NOTE: Do not include all directories at root (for example: blt)

    file(GLOB_RECURSE _test_sources "${PROJECT_SOURCE_DIR}/src/*.cpp" "${PROJECT_SOURCE_DIR}/tests/*.cpp")
    list(FILTER _test_sources INCLUDE REGEX ".*/tests/.*pp")

    blt_add_clang_tidy_target(NAME              ${arg_PREFIX}_guidelines_check_tests
                              CHECKS            "clang-analyzer-*,clang-analyzer-cplusplus*,cppcoreguidelines-*,-cppcoreguidelines-avoid-magic-numbers"
                              SRC_FILES         ${_test_sources})
                                  
    if (ENABLE_COVERAGE)
        blt_add_code_coverage_target(NAME   ${arg_PREFIX}_coverage
                                     RUNNER ${CMAKE_MAKE_PROGRAM} test
                                     SOURCE_DIRECTORIES ${PROJECT_SOURCE_DIR}/src )
    endif()

endmacro(smith_add_code_checks)


##------------------------------------------------------------------------------
## smith_assert_is_directory(DIR_VARIABLE <variable that holds the prefix>)
##
## Asserts that the given DIR_VARIABLE's value is a directory and exists.
## Fails with a helpful message when it doesn't.
##------------------------------------------------------------------------------
macro(smith_assert_is_directory)

    set(options)
    set(singleValueArgs DIR_VARIABLE)
    set(multiValueArgs)

    # Parse the arguments to the macro
    cmake_parse_arguments(arg
         "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT EXISTS "${${arg_DIR_VARIABLE}}")
        message(FATAL_ERROR "Given ${arg_DIR_VARIABLE} does not exist: ${${arg_DIR_VARIABLE}}")
    endif()

    if (NOT IS_DIRECTORY "${${arg_DIR_VARIABLE}}")
        message(FATAL_ERROR "Given ${arg_DIR_VARIABLE} is not a directory: ${${arg_DIR_VARIABLE}}")
    endif()

endmacro(smith_assert_is_directory)


##------------------------------------------------------------------------------
## smith_assert_find_succeeded(PROJECT_NAME <project name>
##                             TARGET       <found target>
##                             DIR_VARIABLE <variable that holds the prefix>)
##
## Asserts that the given PROJECT_NAME's TARGET exists.
## Fails with a helpful message when it doesn't.
##------------------------------------------------------------------------------
macro(smith_assert_find_succeeded)

    set(options)
    set(singleValueArgs DIR_VARIABLE PROJECT_NAME TARGET)
    set(multiValueArgs)

    # Parse the arguments to the macro
    cmake_parse_arguments(arg
         "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN})

    message(STATUS "Checking for expected ${arg_PROJECT_NAME} target '${arg_TARGET}'")
    if (NOT TARGET ${arg_TARGET})
        message(FATAL_ERROR "${arg_PROJECT_NAME} failed to load: ${${arg_DIR_VARIABLE}}")
    else()
        message(STATUS "${arg_PROJECT_NAME} loaded: ${${arg_DIR_VARIABLE}}")
    endif()

endmacro(smith_assert_find_succeeded)


##------------------------------------------------------------------------------
## smith_convert_to_native_escaped_file_path( path output )
##
## This macro converts a cmake path to a platform specific string literal
## usable in C++.  (For example, on windows C:/Path will be come C:\\Path)
##------------------------------------------------------------------------------
macro(smith_convert_to_native_escaped_file_path path output)
    file(TO_NATIVE_PATH ${path} ${output})
    string(REPLACE "\\" "\\\\"  ${output} "${${output}}")
endmacro(smith_convert_to_native_escaped_file_path)


##------------------------------------------------------------------------------
## smith_add_tests( SOURCES         [source1 [source2 ...]]
##                  USE_CUDA        [use CUDA if set]
##                  DEPENDS_ON      [dep1 [dep2 ...]]
##                  NUM_MPI_TASKS   [num tasks]
##                  NUM_OMP_THREADS [num threads])
##
## Creates an executable per given source and then adds the test to CTest
## If USE_CUDA is set, this macro will compile a CUDA enabled version of
## of each unit test.
##------------------------------------------------------------------------------
macro(smith_add_tests)

    set(options )
    set(singleValueArgs NUM_MPI_TASKS NUM_OMP_THREADS USE_CUDA)
    set(multiValueArgs SOURCES DEPENDS_ON)

    # Parse the arguments to the macro
    cmake_parse_arguments(arg
        "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN})

    if ( NOT DEFINED arg_NUM_MPI_TASKS )
        set( arg_NUM_MPI_TASKS 1 )
    endif()

    if ( NOT DEFINED arg_NUM_OMP_THREADS )
        set( arg_NUM_OMP_THREADS 1 )
    endif()

    foreach(filename ${arg_SOURCES})
        get_filename_component(test_name ${filename} NAME_WE)
        if (DEFINED arg_USE_CUDA)
            set(test_name "${test_name}_cuda")
        endif()

        smith_add_executable(NAME        ${test_name}
                             SOURCES     ${filename}
                             OUTPUT_DIR  ${TEST_OUTPUT_DIRECTORY}
                             DEPENDS_ON  ${arg_DEPENDS_ON}
                             FOLDER      smith/tests )

        if (DEFINED arg_USE_CUDA)
            target_compile_definitions(${test_name} PUBLIC SMITH_USE_CUDA_KERNEL_EVALUATION)
        endif()

        blt_add_test(NAME            ${test_name}
                     COMMAND         ${test_name}
                     NUM_MPI_TASKS   ${arg_NUM_MPI_TASKS}
                     NUM_OMP_THREADS ${arg_NUM_OMP_THREADS} )
    endforeach()

endmacro(smith_add_tests)


##------------------------------------------------------------------------------
## smith_configure_file
##
## This macro is a thin wrapper over the builtin configure_file command.
## It has the same arguments/options as configure_file but introduces an
## intermediate file that is only copied to the target file if the target differs
## from the intermediate.
##------------------------------------------------------------------------------
macro(smith_configure_file _source _target)
    set(_tmp_target ${_target}.tmp)
    configure_file(${_source} ${_tmp_target} ${ARGN})
    execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_tmp_target} ${_target})
    execute_process(COMMAND ${CMAKE_COMMAND} -E remove ${_tmp_target})
endmacro(smith_configure_file)


##------------------------------------------------------------------------------
## smith_remove_string_prefix
##
## This macro removes a string prefix from a given string.
##
## PREFIX - String prefix to be removed
## INPUT - String with possible prefix to be removed
## OUTPUT_VAR - Possibly altered output string
##
##------------------------------------------------------------------------------
macro(smith_remove_string_prefix)
    set(options)
    set(singleValueArgs PREFIX INPUT OUTPUT_VAR)
    set(multiValueArgs)

    # Parse the arguments to the macro
    cmake_parse_arguments(arg
         "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT DEFINED arg_PREFIX)
         message(FATAL_ERROR "PREFIX is required")
    endif()

    if(NOT DEFINED arg_INPUT)
         message(FATAL_ERROR "INPUT is required")
    endif()

    if(NOT DEFINED arg_OUTPUT_VAR)
         message(FATAL_ERROR "OUTPUT_VAR is required")
    endif()

    string(LENGTH "${arg_PREFIX}" _prefix_len)
    string(SUBSTRING "${arg_INPUT}" 0 ${_prefix_len} _actual_prefix)

    if(_actual_prefix STREQUAL "${arg_PREFIX}")
        string(REPLACE "${arg_PREFIX}" "" _stripped_path "${arg_INPUT}")
        set(${arg_OUTPUT_VAR} "${_stripped_path}")
    else()
        set(${arg_OUTPUT_VAR} "${arg_INPUT}")
    endif()
endmacro()


##------------------------------------------------------------------------------
## smith_remove_string_prefix
##
## This macro fills OUTPUT_VAR with the subdirectory relative to 
## "<lower case project name>/src"
##
## OUTPUT_VAR - Possibly altered output string
##
##------------------------------------------------------------------------------
macro(smith_get_src_subdirectory)

    set(options)
    set(singleValueArgs OUTPUT_VAR)
    set(multiValueArgs)

    # Parse the arguments to the macro
    cmake_parse_arguments(arg
         "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT DEFINED arg_OUTPUT_VAR)
         message(FATAL_ERROR "OUTPUT_VAR is required")
    endif()

    string(TOLOWER ${PROJECT_NAME} _project_name)

    # Convert both paths to absolute
    file(REAL_PATH "${PROJECT_SOURCE_DIR}/src/${_project_name}" abs_base_dir)
    file(REAL_PATH "${CMAKE_CURRENT_SOURCE_DIR}" abs_full_path)

    # Check if full path starts with the base directory
    string(FIND "${abs_full_path}" "${abs_base_dir}" found_pos)

    if(NOT found_pos EQUAL 0)
        message(FATAL_ERROR "The full path '${abs_full_path}' does not start with the base directory '${abs_base_dir}'")
    endif()

    # Strip base directory from full path
    smith_remove_string_prefix(PREFIX "${abs_base_dir}/"
                               INPUT "${abs_full_path}"
                               OUTPUT_VAR ${arg_OUTPUT_VAR})
endmacro(smith_get_src_subdirectory)


##------------------------------------------------------------------------------
## smith_write_unified_header
##
## This macro writes the unified header (<lowered PROJECT_NAME/<lowered NAME>.hpp)
## to the build directory for the given NAME with the given HEADERS included
## inside of it.
##
## NAME - The name of the unified header.
## HEADERS - Headers to be included in the header.
## NO_PATH_MODIFICATION - ON/OFF(default) Stops include path modification if on,
##    used for the project-level unified header
##
##------------------------------------------------------------------------------
# List to hold all unified headers to be later used to create a master unified header
set(_${PROJECT_NAME}_unified_headers "" CACHE STRING "" FORCE)
macro(smith_write_unified_header)

    set(options)
    set(singleValueArgs NAME NO_PATH_MODIFICATION)
    set(multiValueArgs HEADERS EXCLUDE)

    # Parse the arguments to the macro
    cmake_parse_arguments(arg
         "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT DEFINED arg_NO_PATH_MODIFICATION)
        set(arg_NO_PATH_MODIFICATION OFF)
    endif()

    string(TOLOWER ${arg_NAME} _unified_header_name)
    string(TOLOWER ${PROJECT_NAME} _project_name)
    set(_header ${PROJECT_BINARY_DIR}/include/${_project_name}/${_unified_header_name}.hpp)
    set(_tmp_header ${_header}.tmp)
    set(_src_subdir)
    if(NOT arg_NO_PATH_MODIFICATION)
        smith_get_src_subdirectory(OUTPUT_VAR _src_subdir)
    endif()

    file(WRITE ${_tmp_header} "\/\/ Copyright Lawrence Livermore National Security, LLC and
\/\/ other ${PROJECT_NAME} Project Developers. See the top-level LICENSE file for details.
\/\/
\/\/ SPDX-License-Identifier: (BSD-3-Clause)
\n
")

    file(APPEND ${_tmp_header} "#pragma once\n\n")

    file(APPEND ${_tmp_header} "#include \"${_project_name}\/${_project_name}_config.hpp\"\n\n")

    foreach(_file ${arg_HEADERS})
        if("${_file}" IN_LIST arg_EXCLUDE)
            continue()
        elseif("${_file}" MATCHES "\\.inl$")
            continue()
        elseif("${_file}" MATCHES "(\/detail\/)|(\/internal\/)")
            continue()
        elseif(arg_NO_PATH_MODIFICATION)
            file(APPEND ${_tmp_header} "#include \"${_file}\"\n")
        else()
            set(_headerPath)
            smith_remove_string_prefix(PREFIX "${PROJECT_BINARY_DIR}\/"
                                       INPUT "${_file}"
                                       OUTPUT_VAR _headerPath)
            smith_remove_string_prefix(PREFIX "include\/${_project_name}\/${arg_NAME}\/"
                                       INPUT "${_headerPath}"
                                       OUTPUT_VAR _headerPath)
            set(_headerPath "${_project_name}\/${_src_subdir}\/${_headerPath}")
            file(APPEND ${_tmp_header} "#include \"${_headerPath}\"\n")
        endif()
    endforeach()

    file(APPEND ${_tmp_header} "\n")

    execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_tmp_header} ${_header})
    execute_process(COMMAND ${CMAKE_COMMAND} -E remove ${_tmp_header})

    install(FILES       ${_header}
            DESTINATION include/${_project_name})

    # Add this component's unified header to the list to be added to the project specific unified header
    set(_component_header "${_project_name}/${_unified_header_name}.hpp")
    if("${_${PROJECT_NAME}_unified_headers}" STREQUAL "")
        set(_${PROJECT_NAME}_unified_headers "${_component_header}" CACHE STRING "" FORCE)
    else()
        set(_${PROJECT_NAME}_unified_headers "${_${PROJECT_NAME}_unified_headers};${_component_header}" CACHE STRING "" FORCE)
    endif()
endmacro(smith_write_unified_header)
