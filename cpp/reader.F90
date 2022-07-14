#ifdef SINGLE
# define RTYPE real
#else
# define RTYPE real*8
#endif
subroutine read_scalars(  &
    iunit       &
   ,ndof        &
   ,mrows       &
   ,klower      &
   ,kupper      &
  )
  implicit none
  integer, intent(in)  :: iunit
  integer, intent(out) :: ndof
  integer, intent(out) :: mrows
  integer, intent(out) :: klower
  integer, intent(out) :: kupper
  ! local
  integer              :: idummy
  rewind(iunit)
  read(iunit) idummy  ! ib
  read(iunit) idummy  ! es
  read(iunit) idummy  ! ee
  read(iunit) ndof
  read(iunit) mrows
  read(iunit) klower
  read(iunit) kupper
end subroutine read_scalars
subroutine read_arrays( &
    iunit       &
   ,lcblk       &
   ,ncblk       &
   ,ndof        &
   ,mrows       &
   ,prec_build  &
   ,ipiv        &
   ,afac        &
   ,bblk        &
  )
  implicit none
  integer, intent(in)  :: iunit,lcblk,ncblk,ndof,mrows
  integer, intent(out) :: prec_build(lcblk,ncblk)
  integer, intent(out) ::       ipiv(lcblk,ndof,ncblk)
  real,    intent(out) ::       afac(lcblk,mrows,ndof,ncblk)
  real,    intent(out) ::       bblk(lcblk,ndof,ncblk)
  ! local
  integer              :: idummy, ib
  rewind(iunit)
  read(iunit) idummy  ! ib
  read(iunit) idummy  ! es
  read(iunit) idummy  ! ee
  read(iunit) idummy  ! ndof
  read(iunit) idummy  ! mrows
  read(iunit) idummy  ! klower
  read(iunit) idummy  ! kupper
  read(iunit) prec_build(:,1)
  read(iunit) ipiv(:,:,1)
  read(iunit) afac(:,:,:,1)
  read(iunit) bblk(:,:,1)
  do ib = 2, ncblk
    prec_build(:,ib) = prec_build(:,1)
    ipiv(:,:,ib)     = ipiv(:,:,1)
    afac(:,:,:,ib)   = afac(:,:,:,1)
    bblk(:,:,ib)     = bblk(:,:,1)
  enddo
end subroutine read_arrays

