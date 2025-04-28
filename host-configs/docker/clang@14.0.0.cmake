#------------------------------------------------------------------------------
# !!!! This is a generated file, edit at own risk !!!!
#------------------------------------------------------------------------------
# CMake executable path: /usr/local/bin/cmake
#------------------------------------------------------------------------------

set(CMAKE_PREFIX_PATH "/home/serac/serac_tpls/clang-14.0.0/gmake-4.4.1-qakmu75fp2cyfl36ks3il6e4iej4y7ro;/home/serac/serac_tpls/clang-14.0.0/tribol-0.1.0.18-rchhol6izy7v7mihk6bwxvmeaq5ksi2w;/home/serac/serac_tpls/clang-14.0.0/axom-0.10.1.1-wtyqlfjcggzker6pa43lkx2y23u54mdc;/home/serac/serac_tpls/clang-14.0.0/conduit-0.9.2-yr3qhirjpumf72ckcqxpeyaxa6mdajzl;/home/serac/serac_tpls/clang-14.0.0/mfem-4.8.0.1-y3cutfhkze7wp5ayv7kqzl3fxpatkdin;/home/serac/serac_tpls/clang-14.0.0/raja-2024.07.0-lxwuayzgszq7ihsyhlkb66pced24hlet;/home/serac/serac_tpls/clang-14.0.0/umpire-2024.07.0-avxbs6d42b7uevjgyapanigop72kvdbn;/home/serac/serac_tpls/clang-14.0.0/netcdf-c-4.7.4-dixc2ptl5fuzlco2dm64urxv6tibbqba;/home/serac/serac_tpls/clang-14.0.0/slepc-3.21.2-c3owxvswj76zlvsxi2okdkvtsengugor;/home/serac/serac_tpls/clang-14.0.0/sundials-6.7.0-fwmrsmtttedc7kq6ygr4xim4dqucvzjh;/home/serac/serac_tpls/clang-14.0.0/camp-2024.07.0-nk6veqa4npx4b5kohai6dewgfp5x2t34;/home/serac/serac_tpls/clang-14.0.0/fmt-11.0.2-yc4tb6krft2dzioamgh7eoxvpkpggjud;/home/serac/serac_tpls/clang-14.0.0/hdf5-1.8.23-4dwzsjii6ma2ybvfxdlykhtqcdz5bkq2;/home/serac/serac_tpls/clang-14.0.0/arpack-ng-3.9.0-k3j5a6ni5jcp6g5k64u6tphfdvzpcdsq;/home/serac/serac_tpls/clang-14.0.0/petsc-3.21.5-so4ijbwj57k5jjjbtehzkyu4fz4febao;/home/serac/serac_tpls/clang-14.0.0/zlib-ng-2.2.3-hspizze2o4s3hzjvzllxdkvsqrqitctb;/home/serac/serac_tpls/clang-14.0.0/hypre-2.26.0-inzacs7jfon36l5e4wbss6t7de445cbl;/home/serac/serac_tpls/clang-14.0.0/strumpack-8.0.0-222mse75a64mxp2fgzu4ey4ugzibq5eb;/home/serac/serac_tpls/clang-14.0.0/superlu-dist-8.1.2-f7jalk6opixiu34j33twwgmdoaxqn5wj;/home/serac/serac_tpls/clang-14.0.0/netlib-scalapack-2.2.2-hop67gvdtnxd2akonmgc2n4hhjthle2t;/home/serac/serac_tpls/clang-14.0.0/parmetis-4.0.3-urgxlhmqfbzdbdfcto6h6ikfdud3h4vj;/home/serac/serac_tpls/clang-14.0.0/metis-5.1.0-r6rm2szcrtuf6k3emyax67xewz4v5tec" CACHE STRING "")

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH "ON" CACHE STRING "")

set(CMAKE_BUILD_RPATH "/home/serac/serac_tpls/clang-14.0.0/serac-develop-djeyzcdpuhxn6nmztlezgxzi3d36ds5h/lib;/home/serac/serac_tpls/clang-14.0.0/serac-develop-djeyzcdpuhxn6nmztlezgxzi3d36ds5h/lib64;;" CACHE STRING "")

set(CMAKE_INSTALL_RPATH "/home/serac/serac_tpls/clang-14.0.0/serac-develop-djeyzcdpuhxn6nmztlezgxzi3d36ds5h/lib;/home/serac/serac_tpls/clang-14.0.0/serac-develop-djeyzcdpuhxn6nmztlezgxzi3d36ds5h/lib64;;" CACHE STRING "")

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "")

#------------------------------------------------------------------------------
# Compilers
#------------------------------------------------------------------------------
# Compiler Spec: clang@=14.0.0
#------------------------------------------------------------------------------
if(DEFINED ENV{SPACK_CC})

  set(CMAKE_C_COMPILER "/home/serac/serac_tpls/spack/lib/spack/env/clang/clang" CACHE PATH "")

  set(CMAKE_CXX_COMPILER "/home/serac/serac_tpls/spack/lib/spack/env/clang/clang++" CACHE PATH "")

  set(CMAKE_Fortran_COMPILER "/home/serac/serac_tpls/spack/lib/spack/env/clang/gfortran" CACHE PATH "")

else()

  set(CMAKE_C_COMPILER "/usr/bin/clang" CACHE PATH "")

  set(CMAKE_CXX_COMPILER "/usr/bin/clang++" CACHE PATH "")

  set(CMAKE_Fortran_COMPILER "/usr/bin/gfortran-11" CACHE PATH "")

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

set(TPL_ROOT "/home/serac/serac_tpls/clang-14.0.0" CACHE PATH "")

set(AXOM_DIR "${TPL_ROOT}/axom-0.10.1.1-wtyqlfjcggzker6pa43lkx2y23u54mdc" CACHE PATH "")

set(CAMP_DIR "${TPL_ROOT}/camp-2024.07.0-nk6veqa4npx4b5kohai6dewgfp5x2t34" CACHE PATH "")

set(CONDUIT_DIR "${TPL_ROOT}/conduit-0.9.2-yr3qhirjpumf72ckcqxpeyaxa6mdajzl" CACHE PATH "")

set(LUA_DIR "/usr" CACHE PATH "")

set(MFEM_DIR "${TPL_ROOT}/mfem-4.8.0.1-y3cutfhkze7wp5ayv7kqzl3fxpatkdin" CACHE PATH "")

set(HDF5_DIR "${TPL_ROOT}/hdf5-1.8.23-4dwzsjii6ma2ybvfxdlykhtqcdz5bkq2" CACHE PATH "")

set(HYPRE_DIR "${TPL_ROOT}/hypre-2.26.0-inzacs7jfon36l5e4wbss6t7de445cbl" CACHE PATH "")

set(METIS_DIR "${TPL_ROOT}/metis-5.1.0-r6rm2szcrtuf6k3emyax67xewz4v5tec" CACHE PATH "")

set(PARMETIS_DIR "${TPL_ROOT}/parmetis-4.0.3-urgxlhmqfbzdbdfcto6h6ikfdud3h4vj" CACHE PATH "")

set(NETCDF_DIR "${TPL_ROOT}/netcdf-c-4.7.4-dixc2ptl5fuzlco2dm64urxv6tibbqba" CACHE PATH "")

set(SUPERLUDIST_DIR "${TPL_ROOT}/superlu-dist-8.1.2-f7jalk6opixiu34j33twwgmdoaxqn5wj" CACHE PATH "")

set(ARPACK_DIR "${TPL_ROOT}/arpack-ng-3.9.0-k3j5a6ni5jcp6g5k64u6tphfdvzpcdsq" CACHE PATH "")

# ADIAK not built

# AMGX not built

# CALIPER not built

set(PETSC_DIR "${TPL_ROOT}/petsc-3.21.5-so4ijbwj57k5jjjbtehzkyu4fz4febao" CACHE PATH "")

set(RAJA_DIR "${TPL_ROOT}/raja-2024.07.0-lxwuayzgszq7ihsyhlkb66pced24hlet" CACHE PATH "")

set(SLEPC_DIR "${TPL_ROOT}/slepc-3.21.2-c3owxvswj76zlvsxi2okdkvtsengugor" CACHE PATH "")

set(STRUMPACK_DIR "${TPL_ROOT}/strumpack-8.0.0-222mse75a64mxp2fgzu4ey4ugzibq5eb" CACHE PATH "")

set(SUNDIALS_DIR "${TPL_ROOT}/sundials-6.7.0-fwmrsmtttedc7kq6ygr4xim4dqucvzjh" CACHE PATH "")

set(UMPIRE_DIR "${TPL_ROOT}/umpire-2024.07.0-avxbs6d42b7uevjgyapanigop72kvdbn" CACHE PATH "")

set(TRIBOL_DIR "${TPL_ROOT}/tribol-0.1.0.18-rchhol6izy7v7mihk6bwxvmeaq5ksi2w" CACHE PATH "")

#------------------------------------------------------------------------------
# Devtools
#------------------------------------------------------------------------------

# Code checks disabled due to disabled devtools

set(SERAC_ENABLE_CODE_CHECKS OFF CACHE BOOL "")

set(ENABLE_CLANGFORMAT OFF CACHE BOOL "")

set(ENABLE_CLANGTIDY OFF CACHE BOOL "")

set(ENABLE_DOCS OFF CACHE BOOL "")


