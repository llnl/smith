#------------------------------------------------------------------------------
# !!!! This is a generated file, edit at own risk !!!!
#------------------------------------------------------------------------------
# CMake executable path: /usr/local/bin/cmake
#------------------------------------------------------------------------------

set(CMAKE_PREFIX_PATH "/home/serac/serac_tpls/gcc-12.3.0/gmake-4.4.1-ymq3lgy6vrbjb2pvexpyowq2cvaqynbs;/home/serac/serac_tpls/gcc-12.3.0/tribol-0.1.0.18-4gtqnigqmmc3xeisisd4cedkowv3i757;/home/serac/serac_tpls/gcc-12.3.0/axom-0.10.1.1-alwj37jdrpf63oqnbtaa7c5x6rgyb6wo;/home/serac/serac_tpls/gcc-12.3.0/conduit-0.9.2-n6jclbdnok7mxwtyn4hhfbqqkj3lzdeh;/home/serac/serac_tpls/gcc-12.3.0/mfem-4.7.0.2-reldslqlx6bmfinm47kfammyinrejxws;/home/serac/serac_tpls/gcc-12.3.0/raja-2024.07.0-nb5o4qgx5qldsqgqyqdo7sfi65gq6lqj;/home/serac/serac_tpls/gcc-12.3.0/umpire-2024.07.0-fm36bfy7fuhtza2hzfzdiqo3hk6fvygg;/home/serac/serac_tpls/gcc-12.3.0/amgx-2.3.0.1-4oggyhhdubsmle5td6f2godhlxcmzbod;/home/serac/serac_tpls/gcc-12.3.0/netcdf-c-4.7.4-4r3m543fq3o3hu4wdmrsbac3hth3hf25;/home/serac/serac_tpls/gcc-12.3.0/slepc-3.21.2-so7beuhv2mlyazxkrmlcbwl4ybzcv2na;/home/serac/serac_tpls/gcc-12.3.0/sundials-6.7.0-mwjwbxhiqa3bzo6hziffzy6jhqyj4c7l;/home/serac/serac_tpls/gcc-12.3.0/camp-2024.07.0-bsp277v6kx2hamdfwdjtcbckc5o4ajbn;/home/serac/serac_tpls/gcc-12.3.0/fmt-11.0.2-emeh6vbopcuufqtuwc5oayagquv3c4ig;/home/serac/serac_tpls/gcc-12.3.0/hdf5-1.8.23-y755lvqpm4vliovsmdcltq64yqgwc4rl;/home/serac/serac_tpls/gcc-12.3.0/arpack-ng-3.9.0-mzcbenw56k6mai3k5exif324rl36z4xt;/home/serac/serac_tpls/gcc-12.3.0/petsc-3.21.5-m4xs2lgcqubdzhbtw2beiburtuiqfbkx;/home/serac/serac_tpls/gcc-12.3.0/zlib-ng-2.2.3-udwrqfamudlubkactayddag5tonfphel;/home/serac/serac_tpls/gcc-12.3.0/hypre-2.26.0-3rtivgqomkpkxthaemvon6bqsmo3ujc6;/home/serac/serac_tpls/gcc-12.3.0/strumpack-8.0.0-zygnx7q6nf6i3syu6yo5hnp65twndyfk;/home/serac/serac_tpls/gcc-12.3.0/superlu-dist-8.1.2-nwlz7fntndlx3677rpchtfcyxjnjqw6n;/home/serac/serac_tpls/gcc-12.3.0/netlib-scalapack-2.2.2-ebix6eytoauathgyxqmueqiwomgrj437;/home/serac/serac_tpls/gcc-12.3.0/parmetis-4.0.3-okcqkcpjtcur7tvwt6udrpllwkznifzh;/home/serac/serac_tpls/gcc-12.3.0/metis-5.1.0-2ysjtrvqvyr6zrrwwy2rjx7tj7xj2dsd;/home/serac/serac_tpls/gcc-12.3.0/gcc-runtime-12.3.0-ymetyiumm4ee32xinhir2zcuron3xf2y;/usr/local/cuda-12.3" CACHE STRING "")

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH "ON" CACHE STRING "")

set(CMAKE_BUILD_RPATH "/home/serac/serac_tpls/gcc-12.3.0/serac-develop-mtt5s4opc32fjcbufqy3iib5fxuho4w5/lib;/home/serac/serac_tpls/gcc-12.3.0/serac-develop-mtt5s4opc32fjcbufqy3iib5fxuho4w5/lib64;;" CACHE STRING "")

set(CMAKE_INSTALL_RPATH "/home/serac/serac_tpls/gcc-12.3.0/serac-develop-mtt5s4opc32fjcbufqy3iib5fxuho4w5/lib;/home/serac/serac_tpls/gcc-12.3.0/serac-develop-mtt5s4opc32fjcbufqy3iib5fxuho4w5/lib64;;" CACHE STRING "")

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "")

#------------------------------------------------------------------------------
# Compilers
#------------------------------------------------------------------------------
# Compiler Spec: gcc@=12.3.0
#------------------------------------------------------------------------------
if(DEFINED ENV{SPACK_CC})

  set(CMAKE_C_COMPILER "/home/serac/serac_tpls/spack/lib/spack/env/gcc/gcc" CACHE PATH "")

  set(CMAKE_CXX_COMPILER "/home/serac/serac_tpls/spack/lib/spack/env/gcc/g++" CACHE PATH "")

  set(CMAKE_Fortran_COMPILER "/home/serac/serac_tpls/spack/lib/spack/env/gcc/gfortran" CACHE PATH "")

else()

  set(CMAKE_C_COMPILER "/usr/bin/gcc" CACHE PATH "")

  set(CMAKE_CXX_COMPILER "/usr/bin/g++" CACHE PATH "")

  set(CMAKE_Fortran_COMPILER "/usr/bin/gfortran-12" CACHE PATH "")

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

#------------------------------------------------
# Cuda
#------------------------------------------------

set(CUDAToolkit_ROOT "/usr/local/cuda-12.3" CACHE PATH "")

set(CMAKE_CUDA_COMPILER "${CUDAToolkit_ROOT}/bin/nvcc" CACHE PATH "")

set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}" CACHE PATH "")

set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.3" CACHE PATH "")

set(CMAKE_CUDA_ARCHITECTURES "70" CACHE STRING "")

set(CMAKE_CUDA_FLAGS "" CACHE STRING "")

set(ENABLE_OPENMP ON CACHE BOOL "")

set(ENABLE_CUDA ON CACHE BOOL "")

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON CACHE BOOL "")

set(CMAKE_CUDA_FLAGS "-fPIC -pthread ${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr " CACHE STRING "" FORCE)

# nvcc does not like gtest's 'pthreads' flag

set(gtest_disable_pthreads ON CACHE BOOL "")

#------------------------------------------------------------------------------
# TPLs
#------------------------------------------------------------------------------

set(TPL_ROOT "/home/serac/serac_tpls/gcc-12.3.0" CACHE PATH "")

set(AXOM_DIR "${TPL_ROOT}/axom-0.10.1.1-alwj37jdrpf63oqnbtaa7c5x6rgyb6wo" CACHE PATH "")

set(CAMP_DIR "${TPL_ROOT}/camp-2024.07.0-bsp277v6kx2hamdfwdjtcbckc5o4ajbn" CACHE PATH "")

set(CONDUIT_DIR "${TPL_ROOT}/conduit-0.9.2-n6jclbdnok7mxwtyn4hhfbqqkj3lzdeh" CACHE PATH "")

set(LUA_DIR "/usr" CACHE PATH "")

set(MFEM_DIR "${TPL_ROOT}/mfem-4.7.0.2-reldslqlx6bmfinm47kfammyinrejxws" CACHE PATH "")

set(HDF5_DIR "${TPL_ROOT}/hdf5-1.8.23-y755lvqpm4vliovsmdcltq64yqgwc4rl" CACHE PATH "")

set(HYPRE_DIR "${TPL_ROOT}/hypre-2.26.0-3rtivgqomkpkxthaemvon6bqsmo3ujc6" CACHE PATH "")

set(METIS_DIR "${TPL_ROOT}/metis-5.1.0-2ysjtrvqvyr6zrrwwy2rjx7tj7xj2dsd" CACHE PATH "")

set(PARMETIS_DIR "${TPL_ROOT}/parmetis-4.0.3-okcqkcpjtcur7tvwt6udrpllwkznifzh" CACHE PATH "")

set(NETCDF_DIR "${TPL_ROOT}/netcdf-c-4.7.4-4r3m543fq3o3hu4wdmrsbac3hth3hf25" CACHE PATH "")

set(SUPERLUDIST_DIR "${TPL_ROOT}/superlu-dist-8.1.2-nwlz7fntndlx3677rpchtfcyxjnjqw6n" CACHE PATH "")

set(ARPACK_DIR "${TPL_ROOT}/arpack-ng-3.9.0-mzcbenw56k6mai3k5exif324rl36z4xt" CACHE PATH "")

# ADIAK not built

set(AMGX_DIR "${TPL_ROOT}/amgx-2.3.0.1-4oggyhhdubsmle5td6f2godhlxcmzbod" CACHE PATH "")

# CALIPER not built

set(PETSC_DIR "${TPL_ROOT}/petsc-3.21.5-m4xs2lgcqubdzhbtw2beiburtuiqfbkx" CACHE PATH "")

set(RAJA_DIR "${TPL_ROOT}/raja-2024.07.0-nb5o4qgx5qldsqgqyqdo7sfi65gq6lqj" CACHE PATH "")

set(SLEPC_DIR "${TPL_ROOT}/slepc-3.21.2-so7beuhv2mlyazxkrmlcbwl4ybzcv2na" CACHE PATH "")

set(STRUMPACK_DIR "${TPL_ROOT}/strumpack-8.0.0-zygnx7q6nf6i3syu6yo5hnp65twndyfk" CACHE PATH "")

set(SUNDIALS_DIR "${TPL_ROOT}/sundials-6.7.0-mwjwbxhiqa3bzo6hziffzy6jhqyj4c7l" CACHE PATH "")

set(UMPIRE_DIR "${TPL_ROOT}/umpire-2024.07.0-fm36bfy7fuhtza2hzfzdiqo3hk6fvygg" CACHE PATH "")

set(TRIBOL_DIR "${TPL_ROOT}/tribol-0.1.0.18-4gtqnigqmmc3xeisisd4cedkowv3i757" CACHE PATH "")

#------------------------------------------------------------------------------
# Devtools
#------------------------------------------------------------------------------

# Code checks disabled due to disabled devtools

set(SERAC_ENABLE_CODE_CHECKS OFF CACHE BOOL "")

set(ENABLE_CLANGFORMAT OFF CACHE BOOL "")

set(ENABLE_CLANGTIDY OFF CACHE BOOL "")

set(ENABLE_DOCS OFF CACHE BOOL "")


