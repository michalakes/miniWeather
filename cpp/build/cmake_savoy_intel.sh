#!/bin/bash

#source ${MODULESHOME}/init/bash
#module purge
#module load DefApps gcc/9.3.0 cuda parallel-netcdf cmake

#export TEST_MPI_COMMAND="jsrun -n 1 -c 1 -a 1 -g 1"
unset CUDAFLAGS
unset CXXFLAGS

export PARALLEL_NETCDF_ROOT=/home/michalak/pnetcdf-intel-mpi

./cmake_clean.sh

cmake -DCMAKE_CXX_COMPILER=mpiicpc                  \
      -DCMAKE_C_COMPILER=mpiicc                      \
      -DCMAKE_Fortran_COMPILER=mpiifort               \
      -DYAKL_ARCH="CUDA"                            \
      -DLDFLAGS="-L${PARALLEL_NETCDF_ROOT}/lib -lpnetcdf"  \
      -DNX=200                                      \
      -DNZ=100                                      \
      -DSIM_TIME=1000                               \
      -DOUT_FREQ=10                     \
      -DYAKL_CUDA_FLAGS="-DHAVE_MPI -O3 --use_fast_math -arch sm_70 -ccbin g++ -I${PARALLEL_NETCDF_ROOT}/include -I/opt/nvidia/hpc_sdk/Linux_x86_64/2021/math_libs/11.2/targets/x86_64-linux/include" \
      -DCMAKE_EXE_LINKER_FLAGS="-L/opt/nvidia/hpc_sdk/Linux_x86_64/2021/cuda/lib64 -L/opt/nvidia/hpc_sdk/Linux_x86_64/21.1/math_libs/11.2/targets/x86_64-linux/lib" \
      ..

# a little helper surgery to link with Fortran compiler instead of c++ 
# no idea how to get the wonderful wizard of cmake to do this
sed --in-place -e 's/mpiicpc/mpiifort/' -e 's/$/-nofor-main -cxxlib/' CMakeFiles/jmparallelfor.dir/link.txt

