#------------------------------------------------------------------------------
# !!!! This is a generated file, edit at own risk !!!!
#------------------------------------------------------------------------------
# CMake executable path: /usr/local/bin/cmake
#------------------------------------------------------------------------------

set(CMAKE_PREFIX_PATH "/home/serac/serac_tpls/llvm-19.1.1/tribol-0.1.0.21-wok4mlejt6ejilghgj5rcptuumpg23vh;/home/serac/serac_tpls/llvm-19.1.1/axom-0.11.0.1-gooopmn2vvph5ns6av64e2vgzrzr2vcr;/home/serac/serac_tpls/llvm-19.1.1/conduit-0.9.2-7etgueezbzdtd4pp5k5avxmijgh43tdv;/home/serac/serac_tpls/llvm-19.1.1/mfem-4.9.0.1-6zux6wimz2p3ypa3jk2n2csvud6adcac;/home/serac/serac_tpls/llvm-19.1.1/raja-2024.07.0-fzfgpyaew3paebt7gliwbtkxr54bmt55;/home/serac/serac_tpls/llvm-19.1.1/umpire-2024.07.0-juxmhtd6mga7pu5bo6ergxw5urjxhlgb;/home/serac/serac_tpls/llvm-19.1.1/enzyme-0.0.180-qm5oifx4qp3ovnucldgra2kg4k5ewpoz;/home/serac/serac_tpls/llvm-19.1.1/netcdf-c-4.7.4-v5l5xmqciosurfpasm6zhtzwmvc4vi24;/home/serac/serac_tpls/llvm-19.1.1/slepc-3.21.2-cszx3iu3lcforjpq76a44cfi4onrk3gy;/home/serac/serac_tpls/llvm-19.1.1/sundials-6.7.0-ywh7jd7sjhxuwzf65lm7jue27gismouv;/home/serac/serac_tpls/llvm-19.1.1/camp-2024.07.0-tcpdy6rxi7v4xbdojkwfufax2pf6x2ts;/home/serac/serac_tpls/llvm-19.1.1/fmt-11.0.2-vtj543jq2uuqliyilrg4qlp6ndxxiemd;/home/serac/serac_tpls/llvm-19.1.1/hdf5-1.8.23-rslupkamjuagcfeev6xnyc6qtjjyga3b;/home/serac/serac_tpls/llvm-19.1.1/arpack-ng-3.9.1-fohphl6dmponhr2glsjqz5bp6buuh4yb;/home/serac/serac_tpls/llvm-19.1.1/petsc-3.21.6-un5xab7a74gnaj47ycwwsq3b5blwgrsh;/home/serac/serac_tpls/llvm-19.1.1/hypre-2.26.0-znseoud32x336nnfqbmfadbpc5kd7miy;/home/serac/serac_tpls/llvm-19.1.1/strumpack-8.0.0-bmuxinztn6thgh3uijtqot7wvknifcws;/home/serac/serac_tpls/llvm-19.1.1/superlu-dist-8.1.2-rjyzr2lfgcauuhmyooymprmy4ucho4vl;/home/serac/serac_tpls/llvm-19.1.1/netlib-scalapack-2.2.2-7dd3c7f5hvp7uut32nhgh22we7625hfr;/home/serac/serac_tpls/llvm-19.1.1/parmetis-4.0.3-ojrtoaujp3cchiajoo7r3vak2k5m527n;/home/serac/serac_tpls/none-none/gcc-runtime-13.3.0-6f3a7cf5tiwwl5jhia65h56s4qand7rx;/home/serac/serac_tpls/llvm-19.1.1/metis-5.1.0-47i25u7hnn4xxny6b5z3pekjvj6kjlku;/home/serac/serac_tpls/none-none/compiler-wrapper-1.0-adr4m722sut3yibwgl2ehvjbtuzeozzm;/usr/lib/llvm-19" CACHE STRING "")

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH "ON" CACHE STRING "")

set(CMAKE_BUILD_RPATH "/home/serac/serac_tpls/llvm-19.1.1/serac-develop-e6bpcrrsxr43gk564vbp6vut5gjka6ja/lib;/home/serac/serac_tpls/llvm-19.1.1/serac-develop-e6bpcrrsxr43gk564vbp6vut5gjka6ja/lib64;;" CACHE STRING "")

set(CMAKE_INSTALL_RPATH "/home/serac/serac_tpls/llvm-19.1.1/serac-develop-e6bpcrrsxr43gk564vbp6vut5gjka6ja/lib;/home/serac/serac_tpls/llvm-19.1.1/serac-develop-e6bpcrrsxr43gk564vbp6vut5gjka6ja/lib64;;" CACHE STRING "")

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "")

#------------------------------------------------------------------------------
# Compilers
#------------------------------------------------------------------------------
# Compiler Spec: llvm@19.1.1/yq5rjyb4vplepvexmd5okbdyoezvrvsz
#------------------------------------------------------------------------------
if(DEFINED ENV{SPACK_CC})

  set(CMAKE_C_COMPILER "/home/serac/serac_tpls/none-none/compiler-wrapper-1.0-adr4m722sut3yibwgl2ehvjbtuzeozzm/libexec/spack/clang/clang" CACHE PATH "")

  set(CMAKE_CXX_COMPILER "/home/serac/serac_tpls/none-none/compiler-wrapper-1.0-adr4m722sut3yibwgl2ehvjbtuzeozzm/libexec/spack/clang/clang++" CACHE PATH "")

  set(CMAKE_Fortran_COMPILER "/home/serac/serac_tpls/none-none/compiler-wrapper-1.0-adr4m722sut3yibwgl2ehvjbtuzeozzm/libexec/spack/gcc/gfortran" CACHE PATH "")

else()

  set(CMAKE_C_COMPILER "/usr/lib/llvm-19/bin/clang" CACHE PATH "")

  set(CMAKE_CXX_COMPILER "/usr/lib/llvm-19/bin/clang++" CACHE PATH "")

  set(CMAKE_Fortran_COMPILER "/usr/bin/gfortran-13" CACHE PATH "")

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

set(TPL_ROOT "/home/serac/serac_tpls/llvm-19.1.1" CACHE PATH "")

set(AXOM_DIR "${TPL_ROOT}/axom-0.11.0.1-gooopmn2vvph5ns6av64e2vgzrzr2vcr" CACHE PATH "")

set(CAMP_DIR "${TPL_ROOT}/camp-2024.07.0-tcpdy6rxi7v4xbdojkwfufax2pf6x2ts" CACHE PATH "")

set(CONDUIT_DIR "${TPL_ROOT}/conduit-0.9.2-7etgueezbzdtd4pp5k5avxmijgh43tdv" CACHE PATH "")

set(LUA_DIR "/usr" CACHE PATH "")

set(MFEM_DIR "${TPL_ROOT}/mfem-4.9.0.1-6zux6wimz2p3ypa3jk2n2csvud6adcac" CACHE PATH "")

set(HDF5_DIR "${TPL_ROOT}/hdf5-1.8.23-rslupkamjuagcfeev6xnyc6qtjjyga3b" CACHE PATH "")

set(HYPRE_DIR "${TPL_ROOT}/hypre-2.26.0-znseoud32x336nnfqbmfadbpc5kd7miy" CACHE PATH "")

set(METIS_DIR "${TPL_ROOT}/metis-5.1.0-47i25u7hnn4xxny6b5z3pekjvj6kjlku" CACHE PATH "")

set(PARMETIS_DIR "${TPL_ROOT}/parmetis-4.0.3-ojrtoaujp3cchiajoo7r3vak2k5m527n" CACHE PATH "")

set(NETCDF_DIR "${TPL_ROOT}/netcdf-c-4.7.4-v5l5xmqciosurfpasm6zhtzwmvc4vi24" CACHE PATH "")

set(SUPERLUDIST_DIR "${TPL_ROOT}/superlu-dist-8.1.2-rjyzr2lfgcauuhmyooymprmy4ucho4vl" CACHE PATH "")

set(ARPACK_DIR "${TPL_ROOT}/arpack-ng-3.9.1-fohphl6dmponhr2glsjqz5bp6buuh4yb" CACHE PATH "")

# ADIAK not built

# AMGX not built

# CALIPER not built

set(ENZYME_DIR "${TPL_ROOT}/enzyme-0.0.180-qm5oifx4qp3ovnucldgra2kg4k5ewpoz" CACHE PATH "")

set(PETSC_DIR "${TPL_ROOT}/petsc-3.21.6-un5xab7a74gnaj47ycwwsq3b5blwgrsh" CACHE PATH "")

set(RAJA_DIR "${TPL_ROOT}/raja-2024.07.0-fzfgpyaew3paebt7gliwbtkxr54bmt55" CACHE PATH "")

set(SLEPC_DIR "${TPL_ROOT}/slepc-3.21.2-cszx3iu3lcforjpq76a44cfi4onrk3gy" CACHE PATH "")

set(STRUMPACK_DIR "${TPL_ROOT}/strumpack-8.0.0-bmuxinztn6thgh3uijtqot7wvknifcws" CACHE PATH "")

set(SUNDIALS_DIR "${TPL_ROOT}/sundials-6.7.0-ywh7jd7sjhxuwzf65lm7jue27gismouv" CACHE PATH "")

set(UMPIRE_DIR "${TPL_ROOT}/umpire-2024.07.0-juxmhtd6mga7pu5bo6ergxw5urjxhlgb" CACHE PATH "")

set(TRIBOL_DIR "${TPL_ROOT}/tribol-0.1.0.21-wok4mlejt6ejilghgj5rcptuumpg23vh" CACHE PATH "")

#------------------------------------------------------------------------------
# Devtools
#------------------------------------------------------------------------------

# Code checks disabled due to disabled devtools

set(SERAC_ENABLE_CODE_CHECKS OFF CACHE BOOL "")

set(ENABLE_CLANGFORMAT OFF CACHE BOOL "")

set(ENABLE_CLANGTIDY OFF CACHE BOOL "")

set(ENABLE_DOCS OFF CACHE BOOL "")


