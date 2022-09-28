Program neptune_fcst
  use mpi_f08
  use mod_neptune_model
  implicit none
  type (neptune_model_t) :: model
  integer ierr, itimestep, provided, ntime

! Executable
  call MPI_INIT(ierr)

  call model%init(1,MPI_COMM_WORLD)

  call model%step

! Finalize
  call model%destroy
  call mpi_finalize(ierr)
  stop "neptune_fcst done"

end program neptune_fcst

