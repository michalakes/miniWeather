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
  RTYPE,   intent(out) ::       afac(lcblk,mrows,ndof,ncblk)
  RTYPE,   intent(out) ::       bblk(lcblk,ndof,ncblk)
  ! local
  integer              :: idummy, ib
integer i,j,k
  rewind(iunit)
  read(iunit) idummy  ! ib
  read(iunit) idummy  ! es
  read(iunit) idummy  ! ee
  read(iunit) idummy  ! ndof
  read(iunit) idummy  ! mrows
  read(iunit) idummy  ! klower
  read(iunit) idummy  ! kupper
  if ( iunit .ne. 99 ) read(iunit) prec_build(:,1)
  read(iunit) ipiv(:,:,1)
  read(iunit) afac(:,:,:,1)
  read(iunit) bblk(:,:,1)

  do ib = 2, ncblk
    if ( iunit .ne. 99 ) prec_build(:,ib) = prec_build(:,1)
    ipiv(:,:,ib)     = ipiv(:,:,1)
    afac(:,:,:,ib)   = afac(:,:,:,1)
    bblk(:,:,ib)     = bblk(:,:,1)
  enddo
end subroutine read_arrays


subroutine diff ( a_in , b_in, n )

  integer :: ifdiffs, n, dgts, ii
  RTYPE   :: a_in(*), b_in(*)
  RTYPE   :: a, b
  double precision :: sumE, sum1, sum2, diff1, diff2, serr, perr, rmse, rms1, rms2, tmp1, tmp2


  IFDIFFS=0
  sumE = 0.0
  sum1 = 0.0
  sum2 = 0.0
  diff1 = 0.0
  diff2 = 0.0

  do ii = 1, n 
     a = a_in(ii)
     b = b_in(ii)
write(298,*)ii,a
write(299,*)ii,b

     ! borrowed from  Thomas Oppe's comp program
     sumE = sumE + ( a - b ) * ( a - b )
     sum1 = sum1 + a * a
     sum2 = sum2 + b * b
     diff1 = max ( diff1 , abs ( a - b ) )
     diff2 = max ( diff2 , abs ( b ) )
     IF (a .ne. b) then
       IFDIFFS = IFDIFFS + 1
     ENDIF
     goto 60
20   exit
30   exit
60   continue
  enddo

  rmsE = sqrt ( sumE / dble( n ) )
  rms1 = sqrt ( sum1 / dble( n ) )
  rms2 = sqrt ( sum2 / dble( n ) )
  serr = 0.0
  IF ( sum2 .GT. 0.0d0 ) THEN
    serr = sqrt ( sumE / sum2 )
  ELSE
    IF ( sumE .GT. 0.0d0 ) serr = 1.0
  ENDIF
  perr = 0.0
  IF ( diff2 .GT. 0.0d0 ) THEN
    perr = diff1/diff2
  ELSE
    IF ( diff1 .GT. 0.0d0 ) perr = 1.0
  ENDIF

  IF ( rms1 - rms2 .EQ. 0.0d0 ) THEN
    dgts = 15
  ELSE
    IF ( rms2 .NE. 0 ) THEN
      tmp1 = 1.0d0/( ( abs( rms1 - rms2 ) ) / rms2 )
      IF ( tmp1 .NE. 0 ) THEN
        dgts = log10(tmp1)
      ENDIF
    ENDIF
  ENDIF

  IF (IFDIFFS .NE. 0 ) THEN
    print 76
    PRINT 77, IFDIFFS, rms1, rms2, dgts, rmsE, perr
 76 FORMAT (5x,2x,'Ndifs',4x,'RMS (1)',12x,'RMS (2)',5x,'DIGITS',4x,'RMSE',5x,'pntwise max')
 77 FORMAT ( 1x,I9,2x,e18.10,1x,e18.10,1x,i3,1x,e12.4,1x,e12.4 )
  ENDIF
  return
  
end subroutine diff
