#define LCBLK   32
#define CVEC    LCBLK
#define NCBLK   120

program fmain

! void my_yakl( int *first, int *lcblk, int *ncblk, int *ndof, int *mrows,
!               int *ipiv_host, real *afac_host, real *bblk_host                    ) {

  integer, pointer, contiguous :: precond_host(:,:)
  integer, pointer, contiguous :: ipiv_host(:,:,:)
  real*8,  pointer, contiguous :: afac_host(:,:,:,:)
  real*8,  pointer, contiguous :: bblk_host(:,:,:)
  integer                      :: ndof, nrows, kl, ku

  call read_scalars( 98, ndof, mrows, kl, ku )
  allocate(precond_host(LCBLK,NCBLK))
  allocate(ipiv_host(LCBLK,ndof,NCBLK))
  allocate(afac_host(LCBLK,mrows,ndof,NCBLK))
  allocate(bblk_host(LCBLK,ndof,NCBLK))
  call read_arrays(98,LCBLK,NCBLK,ndof,mrows,precond_host,ipiv_host,afac_host,bblk_host)

  call my_yakl( 1, LCBLK, NCBLK, ndof, mrows, precond_host, ipiv_host, afac_host, bblk_host )

end program fmain
