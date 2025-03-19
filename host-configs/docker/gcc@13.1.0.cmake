#------------------------------------------------------------------------------
# !!!! This is a generated file, edit at own risk !!!!
#------------------------------------------------------------------------------
# CMake executable path: /usr/local/bin/cmake
#------------------------------------------------------------------------------

set(CMAKE_PREFIX_PATH "/home/serac/serac_tpls/gcc-13.1.0/gmake-4.4.1-z34nmxeqddoyrbasgnrpeqp4yjd54m2t;/home/serac/serac_tpls/gcc-13.1.0/tribol-0.1.0.17-atseg3evutrr6wf75covqisqejf5hqge;/home/serac/serac_tpls/gcc-13.1.0/axom-0.10.1.1-vxizmcyxowc4akxwrozjdz5isaidey45;/home/serac/serac_tpls/gcc-13.1.0/conduit-0.9.2-4ulnywfr4k3t7qrpgkplfga6eb7v7fn4;/home/serac/serac_tpls/gcc-13.1.0/mfem-4.7.0.2-s56kqhpmosrhmxlhgyzqvo7ulpimh5ki;/home/serac/serac_tpls/gcc-13.1.0/raja-2024.07.0-iab7mcnmmavtrzju27faac5qjdf4du6x;/home/serac/serac_tpls/gcc-13.1.0/umpire-2024.07.0-55omwhbcpmnujczgmpxzeiv2pxytcrxu;/home/serac/serac_tpls/gcc-13.1.0/netcdf-c-4.7.4-pmrvx4zy6j25xjt3p7wqa444xajny4ad;/home/serac/serac_tpls/gcc-13.1.0/slepc-3.21.2-tch6th6bmzttcsj7cw53tkzhe3pjxzet;/home/serac/serac_tpls/gcc-13.1.0/sundials-6.7.0-vuiz2i4cyei6byqc3j3ezln4n3jthoiq;/home/serac/serac_tpls/gcc-13.1.0/camp-2024.07.0-jclodv7vjbh7asdj2qfud3ppa6vya3ee;/home/serac/serac_tpls/gcc-13.1.0/fmt-11.0.2-rkhj5kkpjizk5o2fxmi2m4tld7klrsia;/home/serac/serac_tpls/gcc-13.1.0/hdf5-1.8.23-mnbt6digjbyn5ma2fejtj6gwavzmw2cr;/home/serac/serac_tpls/gcc-13.1.0/arpack-ng-3.9.0-hhqmfa2uvbeftgfoujacf3u2ib2rpznv;/home/serac/serac_tpls/gcc-13.1.0/petsc-3.21.5-2tyz4of75d2pfrnjpjkbjierrpiduoq2;/home/serac/serac_tpls/gcc-13.1.0/zlib-ng-2.2.3-edia5h2zt5zvruke2ol2ouogyvtj2v5k;/home/serac/serac_tpls/gcc-13.1.0/hypre-2.26.0-4r4ixwd2ljphct3nwlbzrpug6e74q4cn;/home/serac/serac_tpls/gcc-13.1.0/strumpack-8.0.0-tjmgbtszwga65ccn2wmccuxkfmgiwaqh;/home/serac/serac_tpls/gcc-13.1.0/superlu-dist-8.1.2-sraesmlkbsg6ofaqc5rbeaeniqwv7pwu;/home/serac/serac_tpls/gcc-13.1.0/netlib-scalapack-2.2.2-ynmkrr2uxfmcuudo36c2x6hgtp5dh7hc;/home/serac/serac_tpls/gcc-13.1.0/parmetis-4.0.3-bdl5cr5555hdzs6kfffym3q6xvx336iy;/home/serac/serac_tpls/gcc-13.1.0/metis-5.1.0-yg53qncoc5pxme6wqshmqwmhdc526mau;/home/serac/serac_tpls/gcc-13.1.0/gcc-runtime-13.1.0-nzfuos3dxrohrycbhmuzdbnihjgu65zo" CACHE STRING "")

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH "ON" CACHE STRING "")

set(CMAKE_BUILD_RPATH "/home/serac/serac_tpls/gcc-13.1.0/serac-develop-6kt4u57ytv3xgje5x54xq4sd22vaftqg/lib;/home/serac/serac_tpls/gcc-13.1.0/serac-develop-6kt4u57ytv3xgje5x54xq4sd22vaftqg/lib64;;" CACHE STRING "")

set(CMAKE_INSTALL_RPATH "/home/serac/serac_tpls/gcc-13.1.0/serac-develop-6kt4u57ytv3xgje5x54xq4sd22vaftqg/lib;/home/serac/serac_tpls/gcc-13.1.0/serac-develop-6kt4u57ytv3xgje5x54xq4sd22vaftqg/lib64;;" CACHE STRING "")

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "")

#------------------------------------------------------------------------------
# Compilers
#------------------------------------------------------------------------------
# Compiler Spec: gcc@=13.1.0
#------------------------------------------------------------------------------
if(DEFINED ENV{SPACK_CC})

  set(CMAKE_C_COMPILER "/home/serac/serac_tpls/spack/lib/spack/env/gcc/gcc" CACHE PATH "")

  set(CMAKE_CXX_COMPILER "/home/serac/serac_tpls/spack/lib/spack/env/gcc/g++" CACHE PATH "")

  set(CMAKE_Fortran_COMPILER "/home/serac/serac_tpls/spack/lib/spack/env/gcc/gfortran" CACHE PATH "")

else()

  set(CMAKE_C_COMPILER "/usr/bin/gcc" CACHE PATH "")

  set(CMAKE_CXX_COMPILER "/usr/bin/g++" CACHE PATH "")

  set(CMAKE_Fortran_COMPILER "/usr/bin/gfortran-13" CACHE PATH "")

endif()

set(CMAKE_C_FLAGS "-fPIC -pthread" CACHE STRING "")

set(CMAKE_CXX_FLAGS "-fPIC -pthread" CACHE STRING "")

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

set(TPL_ROOT "/home/serac/serac_tpls/gcc-13.1.0" CACHE PATH "")

set(AXOM_DIR "${TPL_ROOT}/axom-0.10.1.1-vxizmcyxowc4akxwrozjdz5isaidey45" CACHE PATH "")

set(CAMP_DIR "${TPL_ROOT}/camp-2024.07.0-jclodv7vjbh7asdj2qfud3ppa6vya3ee" CACHE PATH "")

set(CONDUIT_DIR "${TPL_ROOT}/conduit-0.9.2-4ulnywfr4k3t7qrpgkplfga6eb7v7fn4" CACHE PATH "")

set(LUA_DIR "/usr" CACHE PATH "")

set(MFEM_DIR "${TPL_ROOT}/mfem-4.7.0.2-s56kqhpmosrhmxlhgyzqvo7ulpimh5ki" CACHE PATH "")

set(HDF5_DIR "${TPL_ROOT}/hdf5-1.8.23-mnbt6digjbyn5ma2fejtj6gwavzmw2cr" CACHE PATH "")

set(HYPRE_DIR "${TPL_ROOT}/hypre-2.26.0-4r4ixwd2ljphct3nwlbzrpug6e74q4cn" CACHE PATH "")

set(METIS_DIR "${TPL_ROOT}/metis-5.1.0-yg53qncoc5pxme6wqshmqwmhdc526mau" CACHE PATH "")

set(PARMETIS_DIR "${TPL_ROOT}/parmetis-4.0.3-bdl5cr5555hdzs6kfffym3q6xvx336iy" CACHE PATH "")

set(NETCDF_DIR "${TPL_ROOT}/netcdf-c-4.7.4-pmrvx4zy6j25xjt3p7wqa444xajny4ad" CACHE PATH "")

set(SUPERLUDIST_DIR "${TPL_ROOT}/superlu-dist-8.1.2-sraesmlkbsg6ofaqc5rbeaeniqwv7pwu" CACHE PATH "")

set(ARPACK_DIR "${TPL_ROOT}/arpack-ng-3.9.0-hhqmfa2uvbeftgfoujacf3u2ib2rpznv" CACHE PATH "")

# ADIAK not built

# AMGX not built

# CALIPER not built

set(PETSC_DIR "${TPL_ROOT}/petsc-3.21.5-2tyz4of75d2pfrnjpjkbjierrpiduoq2" CACHE PATH "")

set(RAJA_DIR "${TPL_ROOT}/raja-2024.07.0-iab7mcnmmavtrzju27faac5qjdf4du6x" CACHE PATH "")

set(SLEPC_DIR "${TPL_ROOT}/slepc-3.21.2-tch6th6bmzttcsj7cw53tkzhe3pjxzet" CACHE PATH "")

set(STRUMPACK_DIR "${TPL_ROOT}/strumpack-8.0.0-tjmgbtszwga65ccn2wmccuxkfmgiwaqh" CACHE PATH "")

set(SUNDIALS_DIR "${TPL_ROOT}/sundials-6.7.0-vuiz2i4cyei6byqc3j3ezln4n3jthoiq" CACHE PATH "")

set(UMPIRE_DIR "${TPL_ROOT}/umpire-2024.07.0-55omwhbcpmnujczgmpxzeiv2pxytcrxu" CACHE PATH "")

set(TRIBOL_DIR "${TPL_ROOT}/tribol-0.1.0.17-atseg3evutrr6wf75covqisqejf5hqge" CACHE PATH "")

#------------------------------------------------------------------------------
# Devtools
#------------------------------------------------------------------------------

# Code checks disabled due to disabled devtools

set(SERAC_ENABLE_CODE_CHECKS OFF CACHE BOOL "")

set(ENABLE_CLANGFORMAT OFF CACHE BOOL "")

set(ENABLE_CLANGTIDY OFF CACHE BOOL "")

set(ENABLE_DOCS OFF CACHE BOOL "")


