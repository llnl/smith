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

#---------------------------------------------------------------------------
# Remove non-existant INTERFACE_INCLUDE_DIRECTORIES from imported targets
# to work around CMake error
#---------------------------------------------------------------------------
set(_imported_targets
   axom
   axom::mfem
   conduit
   conduit::conduit_mpi
   conduit::conduit
   conduit_relay_mpi
   conduit_relay_mpi_io
   conduit_blueprint
   conduit_blueprint_mpi
   tribol::mfem
   smith::mfem
   mfem)

foreach(_target ${_imported_targets})
   if(TARGET ${_target})
      message(STATUS "Removing non-existant include directories from target[${_target}]")

      get_target_property(_dirs ${_target} INTERFACE_INCLUDE_DIRECTORIES)
      set(_existing_dirs)
      foreach(_dir ${_dirs})
         if (EXISTS "${_dir}")
            list(APPEND _existing_dirs "${_dir}")
         endif()
      endforeach()
      if (_existing_dirs)
         set_target_properties(${_target} PROPERTIES
                               INTERFACE_INCLUDE_DIRECTORIES "${_existing_dirs}" )
      endif()
   endif()
endforeach()

add_executable(smith_example smith_example.cpp)
target_link_libraries(smith_example smith::smith)

# Compile as HIP source if enabled
if (ENABLE_HIP)
    set_source_files_properties(smith_example.cpp PROPERTIES LANGUAGE HIP)
endif()

enable_testing()
add_test(NAME    smith_example 
         COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ./smith_example)
