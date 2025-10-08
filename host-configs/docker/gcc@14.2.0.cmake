#------------------------------------------------------------------------------
# !!!! This is a generated file, edit at own risk !!!!
#------------------------------------------------------------------------------
# CMake executable path: /usr/local/bin/cmake
#------------------------------------------------------------------------------

set(CMAKE_PREFIX_PATH "/home/serac/serac_tpls/gcc-14.2.0/tribol-0.1.0.21-g7ql3j6epwksrs4mtuuuwogor3f7ps3c;/home/serac/serac_tpls/gcc-14.2.0/axom-0.11.0.1-2j3rkjd4uppms7n7xjvos2dsfkh5gxnx;/home/serac/serac_tpls/gcc-14.2.0/conduit-0.9.2-6unbdvaphhnzvpjzblqlml4ok7slto5j;/home/serac/serac_tpls/gcc-14.2.0/mfem-4.9.0.1-mgm5d2yvvh3yxf7qwnzsdop6r4dxmgkc;/home/serac/serac_tpls/gcc-14.2.0/raja-2024.07.0-hbxa5g4eqneaq3e2pnyaouk2mnrk47zc;/home/serac/serac_tpls/gcc-14.2.0/umpire-2024.07.0-yvluhnikhhswrlvuhfwbgcbur55jtxlu;/home/serac/serac_tpls/gcc-14.2.0/netcdf-c-4.7.4-n7hweek4e4l7z424tytkvm7zs7xkz2y5;/home/serac/serac_tpls/gcc-14.2.0/slepc-3.21.2-3tsbvlf2xmyxypn2ztnzv3j4qtnm7sif;/home/serac/serac_tpls/gcc-14.2.0/sundials-6.7.0-ipet5qmt2fqqa4dqvottg36xs2hiap7e;/home/serac/serac_tpls/gcc-14.2.0/camp-2024.07.0-r6yrhlse676pzry225kxim6w2ri4ywxs;/home/serac/serac_tpls/gcc-14.2.0/fmt-11.0.2-ht2idt2ah5wkaf2qxcobbccu6lzlmb33;/home/serac/serac_tpls/gcc-14.2.0/hdf5-1.8.23-crboya2gv7d3qwkz7l5sznsqorucooqn;/home/serac/serac_tpls/gcc-14.2.0/arpack-ng-3.9.1-zv5giatrest6jgysp75rbupcekcrce47;/home/serac/serac_tpls/gcc-14.2.0/petsc-3.21.6-fswv5hfi23yypboxi7aoj4djfqokwu6q;/home/serac/serac_tpls/gcc-14.2.0/hypre-2.26.0-plqmrslk2ihnpunce3spetsjijxohri7;/home/serac/serac_tpls/gcc-14.2.0/strumpack-8.0.0-yiev5sryvprb4xiygtweai6s4cx5bway;/home/serac/serac_tpls/gcc-14.2.0/superlu-dist-8.1.2-t6kjmsntmut4iomecuxm352cje7sx3nc;/home/serac/serac_tpls/gcc-14.2.0/netlib-scalapack-2.2.2-zss2cpatiaznwykad36zjybixqxo3qrg;/home/serac/serac_tpls/gcc-14.2.0/parmetis-4.0.3-tfdkub4nkbzr4wnooukdhra7wb5ewneh;/home/serac/serac_tpls/gcc-14.2.0/metis-5.1.0-ypp5h6p4n6evogx2wng2eqnp74a4losv;/home/serac/serac_tpls/none-none/gcc-runtime-14.2.0-732djrphgfjipas5vfbmlvwfci4ui4za;/home/serac/serac_tpls/none-none/compiler-wrapper-1.0-adr4m722sut3yibwgl2ehvjbtuzeozzm" CACHE STRING "")

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH "ON" CACHE STRING "")

set(CMAKE_BUILD_RPATH "/home/serac/serac_tpls/gcc-14.2.0/serac-develop-3jgsab7opjmspoacysvq5rnq2oa42ihi/lib;/home/serac/serac_tpls/gcc-14.2.0/serac-develop-3jgsab7opjmspoacysvq5rnq2oa42ihi/lib64;;" CACHE STRING "")

set(CMAKE_INSTALL_RPATH "/home/serac/serac_tpls/gcc-14.2.0/serac-develop-3jgsab7opjmspoacysvq5rnq2oa42ihi/lib;/home/serac/serac_tpls/gcc-14.2.0/serac-develop-3jgsab7opjmspoacysvq5rnq2oa42ihi/lib64;;" CACHE STRING "")

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "")

#------------------------------------------------------------------------------
# Compilers
#------------------------------------------------------------------------------
# Compiler Spec: gcc@14.2.0/na7vd37xl45hztvg4h4fojpnqlfxgd2r
#------------------------------------------------------------------------------
if(DEFINED ENV{SPACK_CC})

  set(CMAKE_C_COMPILER "/home/serac/serac_tpls/none-none/compiler-wrapper-1.0-adr4m722sut3yibwgl2ehvjbtuzeozzm/libexec/spack/gcc/gcc" CACHE PATH "")

  set(CMAKE_CXX_COMPILER "/home/serac/serac_tpls/none-none/compiler-wrapper-1.0-adr4m722sut3yibwgl2ehvjbtuzeozzm/libexec/spack/gcc/g++" CACHE PATH "")

  set(CMAKE_Fortran_COMPILER "/home/serac/serac_tpls/none-none/compiler-wrapper-1.0-adr4m722sut3yibwgl2ehvjbtuzeozzm/libexec/spack/gcc/gfortran" CACHE PATH "")

else()

  set(CMAKE_C_COMPILER "/usr/bin/gcc-14" CACHE PATH "")

  set(CMAKE_CXX_COMPILER "/usr/bin/g++-14" CACHE PATH "")

  set(CMAKE_Fortran_COMPILER "/usr/bin/gfortran-14" CACHE PATH "")

endif()

set(CMAKE_C_FLAGS "-fPIC -pthread" CACHE STRING "")

set(CMAKE_CXX_FLAGS "-fPIC -pthread" CACHE STRING "")

set(CMAKE_Fortran_FLAGS "-fPIC -pthread" CACHE STRING "")

#------------------------------------------------------------------------------
# MPI
#------------------------------------------------------------------------------

set(MPI_C_COMPILER "/usr/bin/mpicc" CACHE PATH "")

set(MPI_CXX_COMPILER "/usr/bin/mpicxx" CACHE PATH "")

set(MPI_Fortran_COMPILER "/usr/bin/mpif90" CACHE PATH "")

set(MPIEXEC_EXECUTABLE "/usr/bin/mpirun" CACHE PATH "")

set(MPIEXEC_NUMPROC_FLAG "-np" CACHE STRING "")

set(ENABLE_MPI ON CACHE BOOL "")

#------------------------------------------------------------------------------
# Hardware
#------------------------------------------------------------------------------

set(ENABLE_OPENMP ON CACHE BOOL "")

#------------------------------------------------------------------------------
# TPLs
#------------------------------------------------------------------------------

set(TPL_ROOT "/home/serac/serac_tpls/gcc-14.2.0" CACHE PATH "")

set(AXOM_DIR "${TPL_ROOT}/axom-0.11.0.1-2j3rkjd4uppms7n7xjvos2dsfkh5gxnx" CACHE PATH "")

set(CAMP_DIR "${TPL_ROOT}/camp-2024.07.0-r6yrhlse676pzry225kxim6w2ri4ywxs" CACHE PATH "")

set(CONDUIT_DIR "${TPL_ROOT}/conduit-0.9.2-6unbdvaphhnzvpjzblqlml4ok7slto5j" CACHE PATH "")

set(LUA_DIR "/usr" CACHE PATH "")

set(MFEM_DIR "${TPL_ROOT}/mfem-4.9.0.1-mgm5d2yvvh3yxf7qwnzsdop6r4dxmgkc" CACHE PATH "")

set(HDF5_DIR "${TPL_ROOT}/hdf5-1.8.23-crboya2gv7d3qwkz7l5sznsqorucooqn" CACHE PATH "")

set(HYPRE_DIR "${TPL_ROOT}/hypre-2.26.0-plqmrslk2ihnpunce3spetsjijxohri7" CACHE PATH "")

set(METIS_DIR "${TPL_ROOT}/metis-5.1.0-ypp5h6p4n6evogx2wng2eqnp74a4losv" CACHE PATH "")

set(PARMETIS_DIR "${TPL_ROOT}/parmetis-4.0.3-tfdkub4nkbzr4wnooukdhra7wb5ewneh" CACHE PATH "")

set(NETCDF_DIR "${TPL_ROOT}/netcdf-c-4.7.4-n7hweek4e4l7z424tytkvm7zs7xkz2y5" CACHE PATH "")

set(SUPERLUDIST_DIR "${TPL_ROOT}/superlu-dist-8.1.2-t6kjmsntmut4iomecuxm352cje7sx3nc" CACHE PATH "")

set(ARPACK_DIR "${TPL_ROOT}/arpack-ng-3.9.1-zv5giatrest6jgysp75rbupcekcrce47" CACHE PATH "")

# ADIAK not built

# AMGX not built

# CALIPER not built

# ENZYME not built

set(PETSC_DIR "${TPL_ROOT}/petsc-3.21.6-fswv5hfi23yypboxi7aoj4djfqokwu6q" CACHE PATH "")

set(RAJA_DIR "${TPL_ROOT}/raja-2024.07.0-hbxa5g4eqneaq3e2pnyaouk2mnrk47zc" CACHE PATH "")

set(SLEPC_DIR "${TPL_ROOT}/slepc-3.21.2-3tsbvlf2xmyxypn2ztnzv3j4qtnm7sif" CACHE PATH "")

set(STRUMPACK_DIR "${TPL_ROOT}/strumpack-8.0.0-yiev5sryvprb4xiygtweai6s4cx5bway" CACHE PATH "")

set(SUNDIALS_DIR "${TPL_ROOT}/sundials-6.7.0-ipet5qmt2fqqa4dqvottg36xs2hiap7e" CACHE PATH "")

set(UMPIRE_DIR "${TPL_ROOT}/umpire-2024.07.0-yvluhnikhhswrlvuhfwbgcbur55jtxlu" CACHE PATH "")

set(TRIBOL_DIR "${TPL_ROOT}/tribol-0.1.0.21-g7ql3j6epwksrs4mtuuuwogor3f7ps3c" CACHE PATH "")

#------------------------------------------------------------------------------
# Devtools
#------------------------------------------------------------------------------

# Code checks disabled due to disabled devtools

set(SERAC_ENABLE_CODE_CHECKS OFF CACHE BOOL "")

set(ENABLE_CLANGFORMAT OFF CACHE BOOL "")

set(ENABLE_CLANGTIDY OFF CACHE BOOL "")

set(ENABLE_DOCS OFF CACHE BOOL "")


