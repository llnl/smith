cmake_minimum_required(VERSION 3.14)

project(serac_example LANGUAGES C CXX)

message(STATUS "CMake Version: ${CMAKE_VERSION}")

include(CMakeFindDependencyMacro)
find_dependency(serac REQUIRED NO_DEFAULT_PATH PATHS "${SERAC_DIR}/lib/cmake")

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
   serac::mfem
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

add_executable(serac_example serac_example.cpp)
target_link_libraries(serac_example serac::serac)

enable_testing()
add_test(NAME    serac_example 
         COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ./serac_example)
