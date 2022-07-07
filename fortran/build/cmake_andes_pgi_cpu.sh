#!/bin/bash

module load pgi parallel-netcdf cmake

export TEST_MPI_COMMAND="mpirun -n 1"
unset CUDAFLAGS
unset CXXFLAGS

./cmake_clean.sh

cmake -DCMAKE_Fortran_COMPILER=mpif90                                                 \
      -DFFLAGS="-O4 -tp=zen -DNO_INFORM -I${OLCF_PARALLEL_NETCDF_ROOT}/include"       \
      -DLDFLAGS="-L${OLCF_PARALLEL_NETCDF_ROOT}/lib -lpnetcdf"                        \
      -DOPENMP_FLAGS="-mp -Minfo=mp"                                                  \
      -DNX=200                                                                        \
      -DNZ=100                                                                        \
      -DDATA_SPEC="DATA_SPEC_GRAVITY_WAVES"                                           \
      -DSIM_TIME=1000                                                                 \
      ..
