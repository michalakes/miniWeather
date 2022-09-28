cmake_minimum_required(VERSION 3.8)
project(miniWeather CXX)

if ("${YAKL_ARCH}" STREQUAL "CUDA")
  enable_language(CUDA)
endif()

enable_testing()


############################################################
## Set Parameters
############################################################
if ("${NX}" STREQUAL "")
  SET(NX 100)
endif()
if ("${NZ}" STREQUAL "")
  SET(NZ 50)
endif()
if ("${SIM_TIME}" STREQUAL "")
  SET(SIM_TIME 1000)
endif()
if ("${OUT_FREQ}" STREQUAL "")
  SET(OUT_FREQ 10)
endif()
if ("${DATA_SPEC}" STREQUAL "")
  SET(DATA_SPEC DATA_SPEC_THERMAL)
endif()
SET(EXE_DEFS "-D_NX=${NX} -D_NZ=${NZ} -D_SIM_TIME=${SIM_TIME} -D_OUT_FREQ=${OUT_FREQ} -D_DATA_SPEC=${DATA_SPEC}")

if ( ("${YAKL_CXX_FLAGS}"    MATCHES ".*SINGLE_PREC.*") OR 
     ("${YAKL_CUDA_FLAGS}"   MATCHES ".*SINGLE_PREC.*") OR 
     ("${YAKL_HIP_FLAGS}"    MATCHES ".*SINGLE_PREC.*") OR 
     ("${YAKL_OPENMP_FLAGS}" MATCHES ".*SINGLE_PREC.*") OR 
     ("${YAKL_SYCL_FLAGS}"   MATCHES ".*SINGLE_PREC.*") )
  message(STATUS "Using single precision")
else()
  message(STATUS "Using double precision")
endif()

############################################################
## Append CXXFLAGS
############################################################
SET(CMAKE_CXX_FLAGS "${CXXFLAGS}")

###
SET(CMAKE_Fortran_FLAGS "${FFLAGS}")
SET(CMAKE_Fortran_COMPILE_OBJECT "<CMAKE_Fortran_COMPILER> -o <OBJECT> <DEFINES> <FLAGS> -c <SOURCE>")
SET(CMAKE_C_COMPILE_OBJECT "<CMAKE_C_COMPILER> -o <OBJECT> <DEFINES> <FLAGS> -c <SOURCE>")


############################################################
## Add YAKL 
############################################################
add_subdirectory(YAKL)
if (${CMAKE_VERSION} VERSION_GREATER "3.18.0")
  set_property(TARGET yakl PROPERTY CUDA_ARCHITECTURES OFF)
  set_property(TARGET yakl PROPERTY CUDA_STANDARD 14)
endif()
include_directories(YAKL)

############################################################
## JM version
############################################################
add_executable(jmparallelfor jm_mpi_parallelfor.cpp avec_timer.c reader.F90 fmain.F90 )
set_target_properties(jmparallelfor PROPERTIES COMPILE_FLAGS "${EXE_DEFS}")

target_link_libraries(jmparallelfor      "${LDFLAGS}")

############################################################
## NEPWRAP
############################################################
add_executable(nepwrap nepwrap.F90 )
set_target_properties(nepwrap PROPERTIES COMPILE_FLAGS "${EXE_DEFS}")
include_directories(/users/michalakes/neptune_atmos/objdir )
target_link_libraries(nepwrap      "${LDFLAGS}")


include(YAKL/yakl_utils.cmake)
yakl_process_target(nepwrap)

if ("${YAKL_ARCH}" STREQUAL "CUDA")
  set_target_properties(nepwrap PROPERTIES LINKER_LANGUAGE CXX)
  if (${CMAKE_VERSION} VERSION_GREATER "3.18.0")
    set_target_properties(nepwrap PROPERTIES CUDA_ARCHITECTURES OFF CUDA_STANDARD 14)
  endif()
endif()



