cmake_minimum_required(VERSION 3.14)

project(smith_example LANGUAGES C CXX)

message(STATUS "CMake Version: ${CMAKE_VERSION}")

if (ENABLE_CUDA)
    enable_language(CUDA)
endif()

if (ENABLE_HIP)
    enable_language(HIP)
endif()

include(CMakeFindDependencyMacro)
find_dependency(smith REQUIRED NO_DEFAULT_PATH PATHS "${SMITH_DIR}/lib/cmake")

add_executable(smith_example smith_example.cpp)
target_link_libraries(smith_example smith::smith)

# Compile as HIP source if enabled
if (ENABLE_HIP)
    set_source_files_properties(smith_example.cpp PROPERTIES LANGUAGE HIP)
endif()

if (ENABLE_CUDA)
    set_source_files_properties(smith_example.cpp PROPERTIES LANGUAGE CUDA)
endif()

enable_testing()
add_test(NAME    smith_example
         COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ./smith_example)
