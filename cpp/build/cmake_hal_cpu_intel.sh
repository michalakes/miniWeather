#!/bin/bash

#source ${MODULESHOME}/init/bash
#module purge
#module load DefApps gcc/9.3.0 cuda parallel-netcdf cmake

module load gcc-9.3.0/gcc-9.3.0
module load cmake-3.18.2/cmake-3.18.2
module load jm-nvhpc-byo-compiler-20.11
module load intel/intel-2020.0.166
module load gcc-9.3.0/gcc-9.3.0

module list

echo $LD_LIBRARY_PATH #| sed 's/://g'


#export TEST_MPI_COMMAND="jsrun -n 1 -c 1 -a 1 -g 1"
unset CUDAFLAGS
unset CXXFLAGS

export PARALLEL_NETCDF_ROOT=/software/depot/pnetcdf-1.11.2

./cmake_clean.sh

export NEPLIBS="-L/users/michalakes/neptune_atmos/lib -lneptune -ldsapi -lioapi \
-L/users/michalakes/p4est-2.8/local/lib -lp4est -lsc \
-L/software/depot/hdf5-1.10.2-parallel-intel-2018/lib -lhdf5_fortran -lhdf5 -ldl -lz \
-L/users/michalakes/.local/software/yaml-0.2.5/lib -lyaml \
/users/michalakes/neptune_atmos/external/timemgr/libesmf_time.a -r8  -traceback -cxxlib -lmkl_intel_lp64 -lmkl_sequential -lmkl_core"

cmake -DCMAKE_CXX_COMPILER=mpiicpc                  \
      -DCMAKE_C_COMPILER=mpiicc                      \
      -DCMAKE_Fortran_COMPILER=mpiifort               \
      -DYAKL_ARCH=OPENMP \
      -DYAKL_OPENMP_FLAGS="-qopenmp" \
      -DLDFLAGS="-g -L${PARALLEL_NETCDF_ROOT}/lib -lpnetcdf $NEPLIBS "  \
      -DNX=200                                      \
      -DNZ=100                                      \
      -DSIM_TIME=1000                               \
      -DOUT_FREQ=10                     \
      -DFFLAGS="-g -I/users/michalakes/neptune_atmos/objdir" \
      -DCXXFLAGS="-g -I /software7/depot/intel-2020/compilers_and_libraries_2020.0.166/linux/mpi/intel64/include -DHAVE_MPI -O3 -I${PARALLEL_NETCDF_ROOT}/include -I/software7/depot/hpc_sdk/Linux_x86_64/20.11/math_libs/include" \
      ..

# a little helper surgery to link with Fortran compiler instead of c++ 
# no idea how to get the wonderful wizard of cmake to do this
sed --in-place -e 's/mpiicpc/mpiifort/' -e 's/$/-nofor-main -cxxlib/' CMakeFiles/nepwrap.dir/link.txt

