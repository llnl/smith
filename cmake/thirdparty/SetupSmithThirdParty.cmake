# Copyright (c) Lawrence Livermore National Security, LLC and
# other Smith Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

if (NOT SMITH_THIRD_PARTY_LIBRARIES_FOUND)
    # Prevent this file from being called twice in the same scope
    set(SMITH_THIRD_PARTY_LIBRARIES_FOUND TRUE)

    # At this time we do not want to profile any libraries. Temporarily unset
    # Caliper and Adiak directories so our upstream libraries don't enable that
    # behavior. Restoring them at bottom of file.
    set(_adiak_dir ${ADIAK_DIR})
    set(_caliper_dir ${CALIPER_DIR})
    unset(ADIAK_DIR CACHE)
    unset(CALIPER_DIR CACHE)

    #------------------------------------------------------------------------------
    # CUDA
    #------------------------------------------------------------------------------
    if(SMITH_ENABLE_CUDA)
        # Manually set includes as system includes
        foreach(_target cuda_runtime cuda)
            get_target_property(_dirs ${_target} INTERFACE_INCLUDE_DIRECTORIES)
            set_property(TARGET ${_target} 
                         APPEND PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
                         "${_dirs}")
        endforeach()
    endif()

    # Policy to use <PackageName>_ROOT variable in find_<Package> commands
    # Policy added in 3.12+
    if(POLICY CMP0074)
        set(CMAKE_POLICY_DEFAULT_CMP0074 NEW)
    endif()

    include(CMakeFindDependencyMacro)

    #------------------------------------------------------------------------------
    # Create global variable to toggle between GPU targets
    #------------------------------------------------------------------------------
    if(SMITH_ENABLE_CUDA)
        # CUDAToolkit required to find cublasLt library
        # Can be removed once this BLT PR is merged https://github.com/LLNL/blt/pull/585 (?)
        find_package(CUDAToolkit REQUIRED)
        set(smith_device_depends blt::cuda CUDA::cublasLt CACHE STRING "" FORCE)
    elseif(SMITH_ENABLE_HIP)
        set(smith_device_depends blt::hip CACHE STRING "" FORCE)
    else()
        set(smith_device_depends "" CACHE STRING "" FORCE)
    endif()

    #------------------------------------------------------------------------------
    # Camp
    #------------------------------------------------------------------------------
    if (NOT CAMP_DIR)
        message(FATAL_ERROR "CAMP_DIR is required.")
    endif()

    smith_assert_is_directory(DIR_VARIABLE CAMP_DIR)

    find_dependency(camp REQUIRED PATHS "${CAMP_DIR}")

    smith_assert_find_succeeded(PROJECT_NAME Camp
                                TARGET       camp
                                DIR_VARIABLE CAMP_DIR)
    message(STATUS "Camp support is ON")
    set(CAMP_FOUND TRUE)

    #------------------------------------------------------------------------------
    # Umpire
    #------------------------------------------------------------------------------
    if(UMPIRE_DIR)
        smith_assert_is_directory(DIR_VARIABLE UMPIRE_DIR)
        find_dependency(umpire REQUIRED PATHS "${UMPIRE_DIR}")

        smith_assert_find_succeeded(PROJECT_NAME Umpire
                                    TARGET       umpire
                                    DIR_VARIABLE UMPIRE_DIR)

        message(STATUS "Umpire support is ON")
        set(UMPIRE_FOUND TRUE)
    else()
        message(STATUS "Umpire support is OFF")
        set(UMPIRE_FOUND FALSE)
    endif()

    #------------------------------------------------------------------------------
    # RAJA
    #------------------------------------------------------------------------------
    if(RAJA_DIR)
        smith_assert_is_directory(DIR_VARIABLE RAJA_DIR)
        find_dependency(RAJA REQUIRED PATHS ${RAJA_DIR})
        smith_assert_find_succeeded(PROJECT_NAME RAJA
                                    TARGET       RAJA
                                    DIR_VARIABLE RAJA_DIR)
        message(STATUS "RAJA support is ON")
        set(RAJA_FOUND TRUE)
    else()
        message(STATUS "RAJA support is OFF")
        set(RAJA_FOUND FALSE)
    endif()

    #------------------------------------------------------------------------------
    # Conduit (required by Axom)
    #------------------------------------------------------------------------------
    if(NOT CONDUIT_DIR)
        MESSAGE(FATAL_ERROR "Could not find Conduit. Conduit requires explicit CONDUIT_DIR.")
    endif()

    smith_assert_is_directory(DIR_VARIABLE CONDUIT_DIR)

    set(_conduit_config "${CONDUIT_DIR}/lib/cmake/conduit/ConduitConfig.cmake")
    if(NOT EXISTS ${_conduit_config})
        MESSAGE(FATAL_ERROR "Could not find Conduit CMake include file ${_conduit_config}")
    endif()

    find_dependency(Conduit REQUIRED
                    PATHS "${CONDUIT_DIR}"
                          "${CONDUIT_DIR}/lib/cmake/conduit")

    smith_assert_find_succeeded(PROJECT_NAME Conduit
                                TARGET       conduit::conduit
                                DIR_VARIABLE CONDUIT_DIR)
    message(STATUS "Conduit support is ON")
    set(CONDUIT_FOUND TRUE)

    # Manually set includes as system includes
    get_target_property(_dirs conduit::conduit INTERFACE_INCLUDE_DIRECTORIES)
    set_property(TARGET conduit::conduit 
                 APPEND PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
                 "${_dirs}")

    #------------------------------------------------------------------------------
    # Sundials
    #------------------------------------------------------------------------------
    if (SUNDIALS_DIR)
        # Note: Sundials is currently only used via MFEM and MFEM's target contains it's information
        smith_assert_is_directory(DIR_VARIABLE SUNDIALS_DIR)
        set(SMITH_USE_SUNDIALS ON CACHE BOOL "")
        
        # Note: MFEM sets SUNDIALS_FOUND itself
        if (NOT SMITH_ENABLE_CODEVELOP)
            set(SUNDIALS_FOUND TRUE)
        endif()
    else()
        set(SMITH_USE_SUNDIALS OFF CACHE BOOL "")
        set(SUNDIALS_FOUND FALSE)
    endif()
    message(STATUS "Sundials support is ${SMITH_USE_SUNDIALS}")

    #------------------------------------------------------------------------------
    # PETSc
    #------------------------------------------------------------------------------
    if (PETSC_DIR)
        smith_assert_is_directory(DIR_VARIABLE PETSC_DIR)
        # NOTE: PETSc is built and used through MFEM
        set(SMITH_USE_PETSC ON CACHE BOOL "")
        
        # Note: MFEM *does not* set PETSC_FOUND itself, likely because we skip petsc build tests
        set(PETSC_FOUND TRUE)
    else()
        set(SMITH_USE_PETSC OFF CACHE BOOL "")
        set(PETSC_FOUND FALSE)
    endif()
    message(STATUS "PETSc support is ${SMITH_USE_PETSC}")

    #------------------------------------------------------------------------------
    # SLEPc
    #------------------------------------------------------------------------------
    if (SLEPC_DIR AND SMITH_USE_PETSC)
        smith_assert_is_directory(DIR_VARIABLE SLEPC_DIR)
        # NOTE: SLEPc is built and used through MFEM
        set(SMITH_USE_SLEPC ON CACHE BOOL "")
        
        # Note: MFEM sets SLEPC_FOUND itself
        if (NOT SMITH_ENABLE_CODEVELOP)
            set(SLEPC_FOUND TRUE)
        endif()
    else()
        set(SMITH_USE_SLEPC OFF CACHE BOOL "")
        set(SLEPC_FOUND FALSE)
    endif()
    message(STATUS "SLEPc support is ${SMITH_USE_SLEPC}")

    #------------------------------------------------------------------------------
    # ARPACK
    #------------------------------------------------------------------------------
    if (ARPACK_DIR AND SMITH_USE_SLEPC)
        smith_assert_is_directory(DIR_VARIABLE ARPACK_DIR)
        include(${CMAKE_CURRENT_LIST_DIR}/FindARPACK.cmake)
    else()
        set(ARPACK_FOUND FALSE)
    endif()
    message(STATUS "ARPACK support is ${ARPACK_FOUND}")

    #------------------------------------------------------------------------------
    # Enzyme
    #------------------------------------------------------------------------------
    if (ENZYME_DIR)
        smith_assert_is_directory(DIR_VARIABLE ENZYME_DIR)
        set(Enzyme_ROOT ${ENZYME_DIR} CACHE PATH "")
        find_dependency(Enzyme REQUIRED)

        smith_assert_find_succeeded(PROJECT_NAME Enzyme
                                    TARGET       ClangEnzymeFlags
                                    DIR_VARIABLE ENZYME_DIR)

        message(STATUS "Checking for Target 'ClangEnzymeFlags' plugin target exists..")
        get_target_property(_clangenzyme_opts ClangEnzymeFlags INTERFACE_COMPILE_OPTIONS)
        if("${_clangenzyme_opts}" MATCHES "\\$<TARGET_FILE:([^>]+)>")
            set(_enzyme_target "${CMAKE_MATCH_1}")

            # Check if the extracted target exists
            if(TARGET "${_enzyme_target}")
                message(STATUS "Found 'ClangEnzymeFlags' plugin target: ${_enzyme_target}")
            else()
                message(FATAL_ERROR "'ClangEnzymeFlags' plugin target '${_enzyme_target}' referenced in INTERFACE_COMPILE_OPTIONS does not exist.")
            endif()
        else()
            message(STATUS "Skipped check. `ClangEnzymeFlags` target does not reference another target")
        endif()

        message(STATUS "Enzyme support is ON")
        set(ENZYME_FOUND TRUE)
    else()
        message(STATUS "Enzyme support is OFF")
        set(ENZYME_FOUND FALSE)
    endif()

    #------------------------------------------------------------------------------
    # MFEM
    #------------------------------------------------------------------------------
    if(NOT SMITH_ENABLE_CODEVELOP)
        message(STATUS "Using installed MFEM")
        smith_assert_is_directory(DIR_VARIABLE MFEM_DIR)
        include(${CMAKE_CURRENT_LIST_DIR}/FindMFEM.cmake)
        smith_assert_find_succeeded(PROJECT_NAME MFEM
                                    TARGET       mfem
                                    DIR_VARIABLE MFEM_DIR)

        if (SMITH_ENABLE_HIP AND STRUMPACK_DIR)
            string(APPEND MFEM_LIBRARIES " -lrocblas -lrocsolver")
            target_link_libraries(mfem INTERFACE rocblas rocsolver)
        endif()
    else()
        message(STATUS "Using MFEM submodule")

        #### Store Data that MFEM clears
        set(tpls_to_save ADIAK AMGX AXOM CALIPER CAMP CONDUIT ENZYME HDF5
                         HYPRE LUA METIS MFEM NETCDF PARMETIS PETSC RAJA 
                         SLEPC SUPERLU_DIST STRUMPACK SUNDIALS TRIBOL
                         UMPIRE)
        foreach(_tpl ${tpls_to_save})
            set(${_tpl}_DIR_SAVE "${${_tpl}_DIR}")
        endforeach()

        #### MFEM "Use" Options

        # Assumes that we have AMGX if we have CUDA
        set(MFEM_USE_AMGX ${SMITH_ENABLE_CUDA} CACHE BOOL "")
        set(MFEM_USE_CALIPER ${CALIPER_FOUND} CACHE BOOL "")
        # We don't use MFEM's Conduit/Axom support
        set(MFEM_USE_CONDUIT OFF CACHE BOOL "")
        set(MFEM_USE_CUDA ${SMITH_ENABLE_CUDA} CACHE BOOL "")
        set(MFEM_USE_HIP ${SMITH_ENABLE_HIP} CACHE BOOL "")
        set(MFEM_USE_LAPACK ON CACHE BOOL "")
        # mfem+mpi requires metis
        set(MFEM_USE_METIS ${SMITH_ENABLE_MPI} CACHE BOOL "")
        set(MFEM_USE_METIS_5 ${SMITH_ENABLE_MPI} CACHE BOOL "")
        set(MFEM_USE_MPI ${SMITH_ENABLE_MPI} CACHE BOOL "")
        if(NETCDF_DIR)
            smith_assert_is_directory(DIR_VARIABLE NETCDF_DIR)
            set(MFEM_USE_NETCDF ON CACHE BOOL "")
        endif()
        # mfem+mpi also needs parmetis
        if(SMITH_ENABLE_MPI)
            smith_assert_is_directory(DIR_VARIABLE PARMETIS_DIR)
            # Slightly different naming convention
            set(ParMETIS_DIR ${PARMETIS_DIR} CACHE PATH "")
        endif()
        set(MFEM_USE_OPENMP ${SMITH_ENABLE_OPENMP} CACHE BOOL "")
        if(PETSC_DIR)
            set(MFEM_USE_PETSC ON CACHE BOOL "")
            set(PETSC_ARCH "" CACHE STRING "")
            set(PETSC_EXECUTABLE_RUNS "ON" CACHE BOOL "")
            if(SLEPC_DIR)
                set(MFEM_USE_SLEPC ON CACHE BOOL "")
                set(SLEPC_ARCH "" CACHE STRING "")
                set(SLEPC_VERSION_OK "TRUE" CACHE BOOL "")
            endif()
        else()
            set(MFEM_USE_PETSC OFF CACHE BOOL "")
            set(MFEM_USE_SLEPC OFF CACHE BOOL "")
        endif()
        set(MFEM_USE_SUNDIALS ${SMITH_USE_SUNDIALS} CACHE BOOL "")
        if(SUPERLUDIST_DIR)
            smith_assert_is_directory(DIR_VARIABLE SUPERLUDIST_DIR)
            # MFEM uses a slightly different naming convention
            set(SuperLUDist_DIR ${SUPERLUDIST_DIR} CACHE PATH "")
            set(MFEM_USE_SUPERLU ${SMITH_ENABLE_MPI} CACHE BOOL "")
        endif()
        if(STRUMPACK_DIR)
            smith_assert_is_directory(DIR_VARIABLE STRUMPACK_DIR)
            set(MFEM_USE_STRUMPACK ON CACHE BOOL "")
            # Since we manually find strumpack before MFEM, we must manually find hip-related packages
            if (SMITH_ENABLE_HIP)
                find_package(hipblas REQUIRED)
                find_package(rocblas REQUIRED)
                find_package(rocsolver REQUIRED)
                find_package(hipsparse REQUIRED)
                find_package(rocthrust REQUIRED)
            endif()
            find_dependency(strumpack CONFIG
                            PATHS "${STRUMPACK_DIR}"
                                  "${STRUMPACK_DIR}/lib/cmake/STRUMPACK"
                                  "${STRUMPACK_DIR}/lib64/cmake/STRUMPACK")
            set(STRUMPACK_REQUIRED_PACKAGES "MPI" "MPI_Fortran" "ParMETIS" "METIS"
                "ScaLAPACK" CACHE STRING
                "Additional packages required by STRUMPACK.")
            set(STRUMPACK_TARGET_NAMES STRUMPACK::strumpack CACHE STRING "")
        endif()
        set(MFEM_USE_ZLIB ON CACHE BOOL "")
        if(ENZYME_DIR)
            smith_assert_is_directory(DIR_VARIABLE ENZYME_DIR)
            set(MFEM_USE_ENZYME ON CACHE BOOL "")
        else()
            set(MFEM_USE_ENZYME OFF CACHE BOOL "")
        endif()

        # MFEM uses Raja/ Umpire if GPU enabled
        if (SMITH_ENABLE_HIP OR SMITH_ENABLE_CUDA)
            set(MFEM_USE_RAJA ON CACHE BOOL "")
            set(MFEM_USE_UMPIRE ON CACHE BOOL "")
        else()
            set(MFEM_USE_RAJA OFF CACHE BOOL "")
            set(MFEM_USE_UMPIRE OFF CACHE BOOL "")
        endif()

        #### MFEM Configuration Options

        # Prefix the "check" targets
        set(MFEM_CUSTOM_TARGET_PREFIX "mfem_" CACHE STRING "")

        # Tweaks needed after Spack converted to the HDF5 CMake build system
        # NOTE: we check if an hdf5 target is namespaced or not, since some versions
        #       of hdf5 do not namespace their targets and others do
        set(HDF5_TARGET_NAMES "" CACHE STRING "")
        if(TARGET hdf5::hdf5-static)
            list(APPEND HDF5_TARGET_NAMES hdf5::hdf5-static)
        else()
            list(APPEND HDF5_TARGET_NAMES hdf5-static)
        endif()
        if(TARGET hdf5::hdf5-shared)
            list(APPEND HDF5_TARGET_NAMES hdf5::hdf5-shared)
        else()
            list(APPEND HDF5_TARGET_NAMES hdf5-shared)
        endif()

        if(TARGET hdf5::hdf5_hl-static)
            set(HDF5_C_LIBRARY_hdf5_hl hdf5::hdf5_hl-static CACHE STRING "")
        else()
            set(HDF5_C_LIBRARY_hdf5_hl hdf5_hl-static CACHE STRING "")
        endif()

        set(HDF5_IMPORT_CONFIG "RELEASE" CACHE STRING "")

        # Add missing include dir to var that MFEM uses
        if (CALIPER_FOUND)
            get_target_property(CALIPER_INCLUDE_DIRS caliper INTERFACE_INCLUDE_DIRECTORIES)
        endif()

        # Disable tests + examples
        set(MFEM_ENABLE_TESTING  OFF CACHE BOOL "")
        set(MFEM_ENABLE_EXAMPLES OFF CACHE BOOL "")
        set(MFEM_ENABLE_MINIAPPS OFF CACHE BOOL "")

        # Build MFEM shared if Smith is being built shared
        set(MFEM_SHARED_BUILD ${BUILD_SHARED_LIBS} CACHE BOOL "")

        # Unset runtime output directory to prevent duplication issue that occurs when using Ninja
        # https://github.com/LLNL/blt/issues/695
        set(tmp_cmake_runtime_output_directory ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
        unset(CMAKE_RUNTIME_OUTPUT_DIRECTORY CACHE)

        # MFEM sets CMAKE_CXX_STANDARD if it is not CACHE variable
        set(CMAKE_CXX_STANDARD ${CMAKE_CXX_STANDARD} CACHE STRING "")

        if(EXISTS "${PROJECT_SOURCE_DIR}/smith/mfem")
            add_subdirectory(${PROJECT_SOURCE_DIR}/smith/mfem  ${CMAKE_BINARY_DIR}/mfem)
        else()
            add_subdirectory(${PROJECT_SOURCE_DIR}/mfem  ${CMAKE_BINARY_DIR}/mfem)
        endif()

        # Restore previous runtime output directory
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${tmp_cmake_runtime_output_directory} CACHE PATH "" FORCE)
 
        set(MFEM_FOUND TRUE CACHE BOOL "" FORCE)

        # Patch the mfem target with the correct include directories
        get_target_property(_mfem_includes mfem INCLUDE_DIRECTORIES)
        target_include_directories(mfem SYSTEM INTERFACE ${_mfem_includes})
        target_include_directories(mfem SYSTEM INTERFACE $<BUILD_INTERFACE:${SMITH_SOURCE_DIR}>)
        target_include_directories(mfem SYSTEM INTERFACE $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/mfem>)

        #### Restore previously stored data
        foreach(_tpl ${tpls_to_save})
            set(${_tpl}_DIR "${${_tpl}_DIR_SAVE}" CACHE PATH "" FORCE)
        endforeach()

        set(MFEM_BUILT_WITH_CMAKE TRUE)
    endif()

    #------------------------------------------------------------------------------
    # ContinuationSolvers
    #------------------------------------------------------------------------------
    message(STATUS "Smith Enable Continuation: ${SMITH_ENABLE_CONTINUATION}")
    
    if(SMITH_ENABLE_CONTINUATION)
        # Allow homotopy solver as a non-submodule
        if (DEFINED CONTINUATION_SOURCE_DIR)
            if(NOT EXISTS "${CONTINUATION_SOURCE_DIR}/CMakeLists.txt")
                message(FATAL_ERROR "Given CONTINUATION_SOURCE_DIR does not contain CMakeLists.txt")
            endif()
        else()
            if(EXISTS "${PROJECT_SOURCE_DIR}/smith/ContinuationSolvers")
                set(CONTINUATION_SOURCE_DIR "${PROJECT_SOURCE_DIR}/smith/ContinuationSolvers" CACHE PATH "")
            else()
                set(CONTINUATION_SOURCE_DIR "${PROJECT_SOURCE_DIR}/ContinuationSolvers" CACHE PATH "")
            endif()

            if (NOT EXISTS "${CONTINUATION_SOURCE_DIR}/CMakeLists.txt")
                message(FATAL_ERROR
                    "The continuationsolver repo is not present. "
                    "Either run the following command in your git repository: \n"
                    "    git submodule update --init --recursive\n"
                    "Or add -DCONTINUATION_SOURCE_DIR=/path/to/ContinuationSolvers to your CMake command." )
            endif()
        endif()

        # Add MUMPS direct solver to MFEM codevelop (we have to do this before Smith's MFEM library is added)
        if (SMITH_ENABLE_CODEVELOP AND MUMPS_DIR)
            set(MFEM_USE_MUMPS ON CACHE BOOL "")
        endif()

        set(CONTINUATION_FOUND TRUE)
        add_subdirectory("${CONTINUATION_SOURCE_DIR}" ${CMAKE_BINARY_DIR}/ContinuationSolvers)
    endif()

    #------------------------------------------------------------------------------
    # Axom
    #------------------------------------------------------------------------------
    if(NOT SMITH_ENABLE_CODEVELOP)
        message(STATUS "Using installed Axom")
        smith_assert_is_directory(DIR_VARIABLE AXOM_DIR)

        find_dependency(axom REQUIRED PATHS "${AXOM_DIR}/lib/cmake")

        smith_assert_find_succeeded(PROJECT_NAME Axom
                                    TARGET       axom
                                    DIR_VARIABLE AXOM_DIR)
        message(STATUS "Axom support is ON")

        #
        # Check for optional Axom headers that are required for Smith
        #

        # sol.hpp
        find_path(
            _sol_found sol.hpp
            PATHS ${AXOM_DIR}/include/axom
            NO_DEFAULT_PATH
            NO_CMAKE_ENVIRONMENT_PATH
            NO_CMAKE_PATH
            NO_SYSTEM_ENVIRONMENT_PATH
            NO_CMAKE_SYSTEM_PATH
        )
        if (NOT _sol_found)
            message(FATAL_ERROR "Given AXOM_DIR did not contain a required header: axom/sol.hpp"
                                "\nTry building Axom with '-DBLT_CXX_STD=c++14' or higher\n ")
        endif()

        # LuaReader.hpp
        find_path(
            _luareader_found LuaReader.hpp
            PATHS ${AXOM_DIR}/include/axom/inlet
            NO_DEFAULT_PATH
            NO_CMAKE_ENVIRONMENT_PATH
            NO_CMAKE_PATH
            NO_SYSTEM_ENVIRONMENT_PATH
            NO_CMAKE_SYSTEM_PATH
        )
        if (NOT _luareader_found)
            message(FATAL_ERROR "Given AXOM_DIR did not contain a required header: axom/inlet/LuaReader.hpp"
                                "\nTry building Axom with '-DLUA_DIR=path/to/lua/install'\n ")
        endif()
        set(LUA_FOUND TRUE CACHE BOOL "")

        # MFEMSidreDataCollection.hpp
        find_path(
            _mfemdatacollection_found MFEMSidreDataCollection.hpp
            PATHS ${AXOM_DIR}/include/axom/sidre/core
            NO_DEFAULT_PATH
            NO_CMAKE_ENVIRONMENT_PATH
            NO_CMAKE_PATH
            NO_SYSTEM_ENVIRONMENT_PATH
            NO_CMAKE_SYSTEM_PATH
        )
        if (NOT _mfemdatacollection_found)
            message(FATAL_ERROR "Given AXOM_DIR did not contain a required header: axom/sidre/core/MFEMSidreDataCollection.hpp"
                                "\nTry building Axom with '-DAXOM_ENABLE_MFEM_SIDRE_DATACOLLECTION=ON'\n ")
        endif()

    else()
        set(ENABLE_FORTRAN OFF CACHE BOOL "" FORCE)
        # Otherwise we use the submodule
        message(STATUS "Using Axom submodule")
        if(NOT LUA_DIR)
            message(FATAL_ERROR "LUA_DIR is required to use the Axom submodule"
                                "\nTry running CMake with '-DLUA_DIR=path/to/lua/install'\n ")
        endif()
        set(AXOM_ENABLE_EXAMPLES OFF CACHE BOOL "")
        set(AXOM_ENABLE_TESTS    OFF CACHE BOOL "")
        set(AXOM_ENABLE_DOCS     OFF CACHE BOOL "")
        set(AXOM_ENABLE_TOOLS    OFF CACHE BOOL "")

        # Used for the doxygen target
        set(AXOM_CUSTOM_TARGET_PREFIX "axom_" CACHE STRING "" FORCE)
        if(EXISTS "${PROJECT_SOURCE_DIR}/smith/axom/src")
            add_subdirectory(${PROJECT_SOURCE_DIR}/smith/axom/src  ${CMAKE_BINARY_DIR}/axom)
        else()
            add_subdirectory(${PROJECT_SOURCE_DIR}/axom/src ${CMAKE_BINARY_DIR}/axom)
        endif()
        set(AXOM_FOUND TRUE CACHE BOOL "" FORCE)

        add_library(axom::cli11 ALIAS cli11)
        add_library(axom::fmt ALIAS fmt)

        if (STRUMPACK_DIR)
            target_link_libraries(sidre PUBLIC STRUMPACK::strumpack)
        endif()

        if(SMITH_ENABLE_OPENMP)
            target_link_libraries(core INTERFACE blt::openmp)
        endif()

        blt_convert_to_system_includes(TARGET core)

        set(ENABLE_FORTRAN ON CACHE BOOL "" FORCE)
    endif()

    #------------------------------------------------------------------------------
    # Gretl
    #------------------------------------------------------------------------------
    message(STATUS "Smith Enable Gretl: ${SMITH_ENABLE_GRETL}")
    
    if(SMITH_ENABLE_GRETL)
        if (NOT DEFINED GRETL_SOURCE_DIR)
            set(GRETL_SOURCE_DIR "${PROJECT_SOURCE_DIR}/gretl" CACHE PATH "")
        endif()

        # check if gretl exists in the GRETL_SOURCE_DIR, if not, try looking through the smith submodule
        if (NOT EXISTS "${GRETL_SOURCE_DIR}/CMakeLists.txt")
            set(GRETL_SOURCE_DIR "${PROJECT_SOURCE_DIR}/smith/gretl" CACHE PATH "" FORCE)
        endif()

        if (NOT EXISTS "${GRETL_SOURCE_DIR}/CMakeLists.txt")
            message(FATAL_ERROR
                "The gretl repo is not present. "
                "Either run the following command in your git repository: \n"
                "    git submodule update --init --recursive\n"
                "Or add -DGRETL_SOURCE_DIR=/path/to/gretl to your CMake command." )
        endif()

        add_subdirectory("${GRETL_SOURCE_DIR}" ${CMAKE_BINARY_DIR}/gretl)
        set(GRETL_FOUND TRUE)
    endif()


    #------------------------------------------------------------------------------
    # Tribol
    #------------------------------------------------------------------------------
    if (NOT SMITH_ENABLE_CODEVELOP)
        if(TRIBOL_DIR)
            smith_assert_is_directory(DIR_VARIABLE TRIBOL_DIR)

            find_dependency(tribol REQUIRED PATHS "${TRIBOL_DIR}/lib/cmake")

            smith_assert_find_succeeded(PROJECT_NAME Tribol
                                        TARGET       tribol
                                        DIR_VARIABLE TRIBOL_DIR)
            blt_convert_to_system_includes(TARGET tribol)
            set(TRIBOL_FOUND ON)
        else()
            set(TRIBOL_FOUND OFF)
        endif()
        
        message(STATUS "Tribol support is " ${TRIBOL_FOUND})
    else()
        set(ENABLE_FORTRAN OFF CACHE BOOL "" FORCE)
        # Otherwise we use the submodule
        message(STATUS "Using Tribol submodule")
        set(BUILD_REDECOMP ${SMITH_ENABLE_MPI} CACHE BOOL "")
        set(TRIBOL_USE_MPI ${SMITH_ENABLE_MPI} CACHE BOOL "")
        set(TRIBOL_ENABLE_TESTS OFF CACHE BOOL "")
        set(TRIBOL_ENABLE_EXAMPLES OFF CACHE BOOL "")
        set(TRIBOL_ENABLE_DOCS OFF CACHE BOOL "")

        if(EXISTS "${PROJECT_SOURCE_DIR}/smith/tribol")
            set(tribol_repo_dir "${PROJECT_SOURCE_DIR}/smith/tribol")
        else()
            set(tribol_repo_dir "${PROJECT_SOURCE_DIR}/tribol")
        endif()

        add_subdirectory(${tribol_repo_dir}  ${CMAKE_BINARY_DIR}/tribol)
        
        target_include_directories(redecomp PUBLIC
            $<BUILD_INTERFACE:${tribol_repo_dir}/src>
        )
        target_include_directories(tribol PUBLIC
            $<BUILD_INTERFACE:${tribol_repo_dir}/src>
            $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/tribol/include>
            $<INSTALL_INTERFACE:include>
        )
        target_include_directories(tribol_shared PUBLIC
            $<BUILD_INTERFACE:${tribol_repo_dir}/src>
            $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/tribol/include>
            $<INSTALL_INTERFACE:include>
        )
        
        set(TRIBOL_FOUND TRUE CACHE BOOL "" FORCE)
        set(ENABLE_FORTRAN ON CACHE BOOL "" FORCE)
    endif()

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
        tribol::mfem)

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

    # When both MFEM and Strumpack are on, MFEM adds the Fortran
    # MPI library/directory but in some cases Spack cannot determine
    # the correct MPI lib directory. This guards against that.
    # https://github.com/spack/spack/issues/24685
    set(_mfem_targets
        mfem
        axom::mfem
        tribol::mfem)
    if(STRUMPACK_DIR)
        list(GET MPI_C_LIBRARIES 0 _first_mpi_lib)
        get_filename_component(_mpi_lib_dir ${_first_mpi_lib} DIRECTORY)
    
        foreach(_target ${_mfem_targets})
            if(TARGET ${_target})
                message(STATUS "Adding MPI link directory to target [${_target}]")
                target_link_directories(${_target} BEFORE INTERFACE ${_mpi_lib_dir})
            endif()
        endforeach()
    endif()

    # Add missing ARPACK flags needed by SLEPc by injecting them into the MFEM targets.
    # https://github.com/mfem/mfem/issues/4364
    if (ARPACK_FOUND)
        foreach(_target ${_mfem_targets})
            if(TARGET ${_target})
                message(STATUS "Adding arpack libraries and include dirs to target [${_target}]")
                target_include_directories(${_target} INTERFACE ${ARPACK_INCLUDE_DIRS})
                target_link_libraries(${_target} INTERFACE ${ARPACK_LIBRARIES})
            endif()
        endforeach()
    endif()
    unset(_mfem_targets)

    # Restore cleared Adiak/Caliper directories, reason at top of file.
    set(ADIAK_DIR ${_adiak_dir} CACHE PATH "" FORCE)
    set(CALIPER_DIR ${_caliper_dir} CACHE PATH "" FORCE)

    #------------------------------------------------------------------------------
    # Adiak
    #------------------------------------------------------------------------------
    if(SMITH_ENABLE_PROFILING AND NOT ADIAK_DIR)
        message(FATAL_ERROR "SMITH_ENABLE_PROFILING cannot be ON without ADIAK_DIR defined. Either specify a host \
                             config with ADIAK_DIR, or rebuild Smith TPLs with +profiling variant.")
    endif()

    if(ADIAK_DIR AND SMITH_ENABLE_PROFILING)
        smith_assert_is_directory(DIR_VARIABLE ADIAK_DIR)

        find_dependency(adiak REQUIRED PATHS "${ADIAK_DIR}")
        smith_assert_find_succeeded(PROJECT_NAME Adiak
                                    TARGET       adiak::adiak
                                    DIR_VARIABLE ADIAK_DIR)
        message(STATUS "Adiak support is ON")
        set(ADIAK_FOUND TRUE)
    else()
        message(STATUS "Adiak support is OFF")
        set(ADIAK_FOUND FALSE)
    endif()

    #------------------------------------------------------------------------------
    # Caliper
    #------------------------------------------------------------------------------
    if(SMITH_ENABLE_PROFILING AND NOT CALIPER_DIR)
        message(FATAL_ERROR "SMITH_ENABLE_PROFILING cannot be ON without CALIPER_DIR defined. Either specify a host \
                             config with CALIPER_DIR, or rebuild Smith TPLs with +profiling variant.")
    endif()

    if(CALIPER_DIR AND SMITH_ENABLE_PROFILING)
        smith_assert_is_directory(DIR_VARIABLE CALIPER_DIR)

        # Should this logic be in the Caliper CMake package?
        # If CMake version doesn't support CUDAToolkit the libraries
        # are just "baked in"
        if(SMITH_ENABLE_CUDA)
            if(CMAKE_VERSION VERSION_LESS 3.17)
                message(FATAL_ERROR "Smith+Caliper+CUDA requires CMake > 3.17.")
            else()
                find_package(CUDAToolkit REQUIRED)
            endif() 
        endif()

        find_dependency(caliper REQUIRED PATHS "${CALIPER_DIR}")
        smith_assert_find_succeeded(PROJECT_NAME Caliper
                                    TARGET       caliper
                                    DIR_VARIABLE CALIPER_DIR)
        message(STATUS "Caliper support is ON")
        set(CALIPER_FOUND TRUE)
    else()
        message(STATUS "Caliper support is OFF")
        set(CALIPER_FOUND FALSE)
    endif()

endif()
