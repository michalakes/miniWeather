#define LCBLK   32
#define CVEC    LCBLK
#define NCBLK_G 120
#define EPS  (1.0e-1)

#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include "const.h"
#include "pnetcdf.h"
#include <ctime>
#include <chrono>
#include <cstring>

using namespace yakl::fortran ;

extern "C"{
void read_scalars_(int*, int*, int*, int*, int*) ;
void read_arrays_(int*, int*, int*, int*, int*, int*, int*, real*, real* ) ;
void diff_(real*, real*, int* ) ;
int avec_microclock_() ;
void vdgbsv_(int*, int* , int*, int* , int* , int* , int* , real*, real*  ) ;
#ifndef YAKL_ARCH_CUDA
int omp_get_thread_num(); int omp_get_num_threads(); int omp_get_max_threads() ;
#endif
}

// We're going to define all arrays on the host because this doesn't use parallel_for
typedef yakl::Array<int   ,1,yakl::memDevice,yakl::styleFortran> int1d;
typedef yakl::Array<int   ,2,yakl::memDevice,yakl::styleFortran> int2d;
typedef yakl::Array<int   ,3,yakl::memDevice,yakl::styleFortran> int3d;
typedef yakl::Array<int   ,4,yakl::memDevice,yakl::styleFortran> int4d;
typedef yakl::Array<real  ,1,yakl::memDevice,yakl::styleFortran> real1d;
typedef yakl::Array<real  ,2,yakl::memDevice,yakl::styleFortran> real2d;
typedef yakl::Array<real  ,3,yakl::memDevice,yakl::styleFortran> real3d;
typedef yakl::Array<real  ,4,yakl::memDevice,yakl::styleFortran> real4d;
typedef yakl::Array<double,1,yakl::memDevice,yakl::styleFortran> doub1d;
typedef yakl::Array<double,2,yakl::memDevice,yakl::styleFortran> doub2d;
typedef yakl::Array<double,3,yakl::memDevice,yakl::styleFortran> doub3d;
typedef yakl::Array<double,4,yakl::memDevice,yakl::styleFortran> doub4d;

typedef yakl::Array<int    const,1,yakl::memDevice,yakl::styleFortran> intConst1d;
typedef yakl::Array<int    const,2,yakl::memDevice,yakl::styleFortran> intConst2d;
typedef yakl::Array<int    const,3,yakl::memDevice,yakl::styleFortran> intConst3d;
typedef yakl::Array<real   const,1,yakl::memDevice,yakl::styleFortran> realConst1d;
typedef yakl::Array<real   const,2,yakl::memDevice,yakl::styleFortran> realConst2d;
typedef yakl::Array<real   const,3,yakl::memDevice,yakl::styleFortran> realConst3d;
typedef yakl::Array<double const,1,yakl::memDevice,yakl::styleFortran> doubConst1d;
typedef yakl::Array<double const,2,yakl::memDevice,yakl::styleFortran> doubConst2d;
typedef yakl::Array<double const,3,yakl::memDevice,yakl::styleFortran> doubConst3d;

// Some arrays still need to be on the host, so we will explicitly create Host Array typedefs
typedef yakl::Array<int   ,1,yakl::memHost,yakl::styleFortran> int1dHost;
typedef yakl::Array<int   ,2,yakl::memHost,yakl::styleFortran> int2dHost;
typedef yakl::Array<int   ,3,yakl::memHost,yakl::styleFortran> int3dHost;
typedef yakl::Array<int   ,4,yakl::memHost,yakl::styleFortran> int4dHost;
typedef yakl::Array<real  ,1,yakl::memHost,yakl::styleFortran> real1dHost;
typedef yakl::Array<real  ,2,yakl::memHost,yakl::styleFortran> real2dHost;
typedef yakl::Array<real  ,3,yakl::memHost,yakl::styleFortran> real3dHost;
typedef yakl::Array<real  ,4,yakl::memHost,yakl::styleFortran> real4dHost;
typedef yakl::Array<double,1,yakl::memHost,yakl::styleFortran> doub1dHost;
typedef yakl::Array<double,2,yakl::memHost,yakl::styleFortran> doub2dHost;
typedef yakl::Array<double,3,yakl::memHost,yakl::styleFortran> doub3dHost;

///////////////////////////////////////////////////////////////////////////////////////
// Variables that are initialized but remain static over the coure of the simulation
///////////////////////////////////////////////////////////////////////////////////////
struct Fixed_data {
  int lcblk, ncblk, ndof, mrows;  
  int kl, ku ;
  int i_beg, k_beg;           //beginning index in the x- and z-directions for this MPI task
  int nranks, myrank;         //Number of MPI ranks and my rank id
  int left_rank, right_rank;  //MPI Rank IDs that exist to my left and right in the global domain
  int mainproc;               //Am I the main process (rank == 0)?
  int2d precond_build;        //   Dimensions: (lcblk, ncblk)
  int2d ju;                   //   Dimensions: (lcblk, ncblk)
  int2d jp;                   //   Dimensions: (lcblk, ncblk)
};

///////////////////////////////////////////////////////////////////////////////////////
// Variables that are dynamics over the course of the simulation
///////////////////////////////////////////////////////////////////////////////////////

//Declaring the functions defined after "main"
void init                 ( int*, int*, real*, real*, int3d &ipiv, real4d &afac, real3d &bblk, Fixed_data &fixed_data );
void verify               ( int3d &ipiv, real4d &afac, real3d &bblk, Fixed_data &fixed_data );
YAKL_INLINE void vdgbtf( int ib, int n, int kl, int ku,
                          real4d const &ab,
                          int ldab,
                          int3d const &ipiv,
                          int2d const &ju,
                          int2d const &jp,
                          Fixed_data const &fixed_data,
                          yakl::InnerHandler & inner_handler ) ;
YAKL_INLINE void vdgbtrs( int ib,
                          int n,                      // fixed_data.ndof
                          int kl, int ku, int nrhs,   // fixed_data.kl, fixed_data.ku, 1
                          real4d const &ab,           // afac
                          int ladb,                   // fixed_data.mrows
                          int3d const &ipiv,          // ipiv
                          real3d const &b,            // bblk
                          int ldb,                    // fixed_data.ndof
                          real2d const &tempv,        // tempv
                          yakl::InnerHandler & inner_handler ) ;


///////////////////////////////////////////////////////////////////////////////////////
// THE MAIN PROGRAM STARTS HERE
///////////////////////////////////////////////////////////////////////////////////////
//int main(int argc, char **argv) {
void vdgbsv_( int *first, int *lcblk, int *ncblk, int *ndof, int *mrows, 
               int *precond_build_host, int *ipiv_host, real *afac_host, real *bblk_host                    ) {
  int s, e ;
  //MPI_Init(&argc,&argv);
  if ( *first ) yakl::init();
  {
    Fixed_data fixed_data;
    int3d  ipiv;   // lcblk, ndof, ncblk
    real4d afac;   // lcblk, mrows, ndof, ncblk
    real3d bblk;   // lcblk, ndof, ncblk

    // Init allocates the state and hydrostatic arrays hy_*
    init( precond_build_host, ipiv_host, afac_host, bblk_host, ipiv, afac, bblk, fixed_data );

    int2d  ju    = int2d("ju",LCBLK,fixed_data.ncblk) ;
    int2d  jp    = int2d("jp",LCBLK,fixed_data.ncblk) ;
    real2d tempv = real2d( "tempv", LCBLK, fixed_data.ncblk ) ;

    s = avec_microclock_() ;
#ifndef YAKL_ARCH_CUDA
fprintf(stderr,"max thread %d\n",omp_get_max_threads()) ;
fprintf(stderr,"num thread %d\n",omp_get_num_threads()) ;
#endif
    yakl::fence() ;
    parallel_outer( "vdgbtf", 
                             fixed_data.ncblk, 
                             YAKL_LAMBDA ( int ib, yakl::InnerHandler inner_handler )
    {
//yakl::fence_inner(inner_handler) ;
//single_inner( [&] () {
//  printf("ib %d\n",ib); 
//}, inner_handler) ;
//yakl::fence_inner(inner_handler) ;
//fprintf(stderr,"thread %d\n",omp_get_thread_num()) ;
      vdgbtf( ib, 
               fixed_data.ndof,
               fixed_data.kl, fixed_data.ku, 
               afac, 
               fixed_data.mrows,
               ipiv,
               ju, jp,
               fixed_data,
               inner_handler ) ;
      vdgbtrs( ib,
               fixed_data.ndof,                 // n
               fixed_data.kl, fixed_data.ku, 1, // kl, ku, nrhs
               afac,                            // ab
               fixed_data.mrows,                // ldab
               ipiv,                            // ipiv
               bblk,                            // bblk
               fixed_data.ndof,                 // ldb
               tempv,                           // temp storage
               inner_handler           ) ;
    }, yakl::LaunchConfig<LCBLK>() ) ;
    yakl::fence() ;

    e = avec_microclock_() ;

    verify( ipiv, afac, bblk, fixed_data );

  }

fprintf(stderr,"kernel time %d\n",e-s) ;

  yakl::finalize();
//  MPI_Finalize();
}

YAKL_INLINE void vdgbtrs( int ib,
                          int n,                     // fixed_data.ndof
                          int kl, int ku, int nrhs,  // fixed_data.kl,fixed_data.ku,1
                          real4d const &ab,          // afac
                          int ladb,                  // fixed_data.mrows
                          int3d const &ipiv,         // ipiv
                          real3d const &b,           // bblk
                          int ldb,                   // fixed_data.ndof
                          real2d const &tempv,       // tempv
                          yakl::InnerHandler & inner_handler )
{
  if ( n == 0 || nrhs == 0 ) return ;
  int kd = ku + kl + 1 ;
  int lnoti = kl > 0 ;
  if ( lnoti ) {
    for ( int j = 1 ; j <= n-1 ; j++ ) {
      int lm = (kl < n-j)?kl:n-j ;
      for ( int ii = 0 ; ii < nrhs ; ii++ ) {
        parallel_inner( yakl::fortran::Bounds<2>(kl,LCBLK),[&](int i, int ie){
          int l = ipiv(ie,j,ib) ;
          if ( l != j ) {
            real tempb = b(ie,l+ii*ldb,ib) ;
            b(ie,l+ii*ldb,ib) = b(ie,j+ii*ldb,ib) ;
            b(ie,j+ii*ldb,ib) = tempb ;
          }
        }, inner_handler ) ;
      }
//    CALL vdger( es, ee                      &  ! elem start/end
//               , lm, nrhs                   &  ! m, n
//               , -one                       &  ! alpha
//               , ab(M2DEX(1,kd+1,j)), 1     &  ! x, incx
//               , b(M2DEX(1,j,1))    , ldb   &  ! y, incy
//               , b(M2DEX(1,j+1,1))  , ldb   )  ! a, lda
      // inline of dger
      parallel_inner( yakl::fortran::Bounds<1>(LCBLK),[&] (int ie) {
        int jy = 0 ;
        for ( int jj = 0 ; jj < nrhs ; jj++ ) {
          for ( int ii = 0 ; ii < lm ; ii++ ) {
            //    a 
            b(ie,j+1+ii+jj*ldb, ib) -=  // minus because alpha = -1
            //              x                  y
               ab(ie,kd+1+ii,j,ib) * b(ie,j+jy,ib) ;
          }
          jy = jy + ldb ;
        }
      }, inner_handler );
    }
    for ( int i = 1 ; i <= nrhs ; i++ ) {
// CALL vdtbsv( es,ee,'Upper', 'No transpose', 'Non-unit', n, kl+ku, &
//              ab, ldab, b(M2DEX(1,1,i)), 1, tempv )
#define X(I) b(ie,I,ib)
#define A(I,J) ab(ie,I+(J-1)*ldb,1,ib)
      int kx1 = 1 ;
      int k = kl+ku ;
      int kplus1 = k+1 ;
      for (int j = n ; j >= 1 ; j-- ) {
        int l = kplus1 - j ;
        parallel_inner( yakl::fortran::Bounds<1>(LCBLK),[&] (int ie) {
          X(j) /= A(kplus1,j) ;
          real temp = X(j) ;
          int iend = (1>(j-k))?1:(j-k) ; // max(1,j-k)
          for (int i = j-1 ; i >= iend ; i-- ) {
            X(i) -= temp * A(l+i,j) ;
          }
        }, inner_handler );
      }
    }
  }
#undef X
#undef A
}


YAKL_INLINE void vdgbtf( int ib, 
                         int n,                // ndof
                         int kl, int ku,
                         real4d const &ab,
                         int ldab,             // mrows
                         int3d const &ipiv,
                         int2d const &ju,
                         int2d const &jp,
                         Fixed_data const &fixed_data,
                         yakl::InnerHandler & inner_handler )
{
  int kv = ku + kl ;
  int km ;
  if ( n == 0 ) return ; // quick return if possible

// Gaussian elimination with partial pivoting
// Set fill-in elements in columns KU+2 to KV to zero.
//        DO 20 J = KU + 2, MIN( KV, N )
//  !$acc loop collapse(2)
//           DO 10 I = KV - J + 2, KL
//              do ie = es, ee
//  !               if ( .not. precon_build(ie) ) cycle
//                 AB( M2DEX(ie, I, J) ) = ZERO
//              end do
//     10    CONTINUE
//     20 CONTINUE
  for ( int j = (ku+2)-1 ; j <= ((kv<n)?kv:n)-1 ; j++ ) { 
    parallel_inner( yakl::fortran::Bounds<2>({(kv-(j+1)+2)-1,(kl)-1},LCBLK),
                    [&] (int i, int ie )  // TODO check, are these still in C order?
      {
        ab(ie,i,j,ib) = 0. ;
      }, inner_handler
    ) ;
  }
// JU is the index of the last column affected by the current stage
// of the factorization.
//        JU = 1
//
  parallel_inner( yakl::fortran::Bounds<1>(LCBLK),[&](int ie) {ju(ie,ib)=1;}, inner_handler );
//         DO 40 J = 1, N
// !          Set fill-in elements in column J+KV to zero.
//           IF( J+KV.LE.N ) THEN
//              DO 30 I = 1, KL
//                do ie = es, ee
//                  AB( M2DEX(ie, I, J+KV) ) = ZERO
//                end do
//    30        CONTINUE
//           END IF
// !         Find pivot and test for singularity. KM is the number of
// !         subdiagonal elements in the current column.
//           KM = MIN( KL, N-J )
  for ( int j=1 ; j <= n ; j++ )
  {
    if ( j+kv <= n ) {
      parallel_inner( yakl::fortran::Bounds<2>(kl,LCBLK),[&](int i, int ie){
        ab(ie,i,j+kv,ib) = 0. ;
      }, inner_handler ) ;
    }
    km = (kl<n-j)?kl:n-j ;
//           CALL VIDAMAX1( es, ee, KM+1, AB( M2DEX(1, KV+1, J)), JP )
yakl::fence_inner(inner_handler) ;
#define dx(A,B) ab(A,(kv+1)+(B)-1,j,ib)
#define absdx(I) ((dx(ie,I)>0.)?(dx(ie,I)):(-dx(ie,I)))
    parallel_inner( yakl::fortran::Bounds<1>(LCBLK),[&](int ie){
      real dmax = absdx(1) ;
      for ( int ii = 2 ; ii <= km+1 ; ii++ ) {
        if ( absdx(ii)>dmax ) dmax = absdx(ii) ;
      }
      int pivot = (dmax < EPS)?1:0;
      jp(ie,ib)=1 ;
      for ( int ii = 2 ; ii <= km+1 ; ii++ ) {
        if ( absdx(ii)>dmax && pivot ) {
          dmax = absdx(ii) ;
          jp(ie,ib) = ii ;
        }
      }
    }, inner_handler );
#undef dx
#undef absdx
yakl::fence_inner(inner_handler) ;

//           IPIV(M1DEXv(J)) = JP(M0DEXv) + J - 1
//           do ie = es, ee
//             JU(M0DEX(ie)) = MAX( JU(M0DEX(ie)), MIN( J+KU+JP(M0DEX(ie))-1, N ) )
//           enddo
    parallel_inner( yakl::fortran::Bounds<1>(LCBLK),[&] (int ie) {
      ipiv(ie,j,ib) = jp(ie,ib)+j-1 ;
      int j1 = ((j+ku+jp(ie,ib)-1)<n)?(j+ku+jp(ie,ib)-1):n ;
      ju(ie,ib)=(ju(ie,ib)>j1)?ju(ie,ib):j1 ;
    }, inner_handler );

    int minj = 9999999 ;
    int maxj = -9999999 ;
    for ( int ie = 1 ; ie <= LCBLK ; ie++ ) {
      if ( ju(ie,ib) < minj ) minj = ju(ie,ib) ;
      if ( ju(ie,ib) > maxj ) maxj = ju(ie,ib) ;
    }
#if 1
    parallel_inner( yakl::fortran::Bounds<1>(LCBLK),[&] (int ie) {
      if ( jp(ie,ib) != 1 ) {
//        swap( ju(ie)-j+1, ab(ie,kv+jp(ie),j), (ldab-1)*LCBLK,
//                          ab(ie,  kv+1   ,j), (ldab-1)*LCBLK )
        int ix = 0 ;
        for ( int i=1 ; i <= ju(ie,ib)-j+1 ; i++ ) {
          real temp = ab(ie,kv+jp(ie,ib)+ix,j,ib) ;
          ab(ie,kv+jp(ie,ib)+ix,j,ib) = ab(ie,kv+1+ix,j,ib) ;
          ab(ie,kv+1+ix,j,ib) = temp ;
          ix += (ldab-1)*LCBLK ;
        }
      }
    }, inner_handler );
#endif
    if ( km > 0 ) {
//        mydscal( km,1.0/ab(ie,kv+1,j), ab(ie,kv+2,j), 1)
      parallel_inner( yakl::fortran::Bounds<1>(LCBLK),[&] (int ie) {
        for ( int i = 0 ; i < km ; i++ ) {
          ab(ie,kv+2+i,j,ib) *= 1.0/ab(ie,kv+1,j,ib) ;
        }
      }, inner_handler );
    }
yakl::fence_inner(inner_handler) ;
#if 1
    if ( km > 0 && minj > j ) {
// vdger( es, ee                                &  ! elem start/end
//      , km, minj-j                            &  ! m, n
//      , -one                                  &  ! alpha
//      , AB( M2DEX(1, KV+2, J  ) ) , 1         &  ! x, incx
//      , AB( M2DEX(1, KV  , J+1) ) , LDAB-1    &  ! y, incy
//      , AB( M2DEX(1, KV+1, J+1) ) , LDAB-1    )  ! a, lda
//    real4d afac;   // lcblk, mrows (ldab), ndof (n), ncblk
      // inline of dger
      parallel_inner( yakl::fortran::Bounds<1>(LCBLK),[&] (int ie) {
        int jy = 0 ;
        for ( int jj = 0 ; jj < minj-j ; jj++ ) {
          for ( int ii = 0 ; ii < km ; ii++ ) {
            //    a 
            ab(ie, kv+1  +ii +jj*(ldab-1) ,j+1, ib) -=  // minus because alpha = -1
            //              x                  y
                   ab(ie,kv+2+ii,j,ib) * ab(ie,kv+jy,j+1,ib) ;
          }
          jy = jy + ldab-1 ;
        }
      }, inner_handler );
#if 0
      parallel_inner( yakl::fortran::Bounds<1>(LCBLK),[&] (int ie) {
        for ( int jj = minj - j + 1 ; jj <= maxj - j ; jj++ ) {
          for ( int ii = 0 ; ii < km ; ii++ ) {
            if ( jj <= ju(ie,ib)-j ) {
              ab(ie,kv+1+ii-jj,j+1+jj-1,ib)-=ab(ie,kv+2+ii-1,j,ib)*ab(ie,kv-jj+1,j+1+jj-1,ib);
            }
          }
        }
      }, inner_handler );
#endif
    }
yakl::fence_inner(inner_handler) ;
#endif
  } // j loop
}


void init( int *precond_build_host, int *ipiv_host, real *afac_host, real *bblk_host,
           int3d &ipiv, real4d &afac, real3d &bblk, Fixed_data &fixed_data ) {
  auto &nranks           = fixed_data.nranks          ;
  auto &ndof             = fixed_data.ndof            ;
  auto &mrows            = fixed_data.mrows           ;
  auto &lcblk            = fixed_data.lcblk           ;
  auto &ncblk            = fixed_data.ncblk           ;
  auto &myrank           = fixed_data.myrank          ;
  auto &mainproc         = fixed_data.mainproc        ;
  auto &i_beg            = fixed_data.i_beg           ;
  int  ierr;
  int  memcheck ;

//  ierr = MPI_Comm_size(MPI_COMM_WORLD,&nranks);
//  ierr = MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

  ndof  = 145 ;
  mrows = 70 ;
  lcblk = LCBLK ;
  ncblk = NCBLK_G / 1 ; //nranks ;

  fprintf(stderr, "ndof %d \n"  , ndof );
  fprintf(stderr, "mrows %d \n" , mrows );
  fprintf(stderr, "lcblk %d \n" , lcblk );
  fprintf(stderr, "ncblk %d \n" , ncblk );
  memcheck =  0 ;
  memcheck += sizeof(int) *lcblk*ndof*ncblk       ; //ipiv
  memcheck += sizeof(real)*lcblk*ndof*ncblk       ; //bblk
  memcheck += sizeof(real)*lcblk*mrows*ndof*ncblk ; //afac
  memcheck += sizeof(real)*lcblk*ncblk            ; //ju
  memcheck += sizeof(real)*lcblk*ncblk            ; //jp
  fprintf(stderr, "memcheck  %d \n" , memcheck ) ;
#ifdef YAKL_ARCH_CUDA
  if ( memcheck > 1073741824 ) {
    fprintf(stderr, "memcheck > 1GB %d \n" , memcheck ) ;
    exit(-1) ;
  }
#endif

  ipiv          =  int3d( "ipiv" , lcblk, ndof, ncblk);
  afac          = real4d( "afac" , lcblk, mrows, ndof, ncblk) ;
  bblk          = real3d( "bblk" , lcblk, ndof, ncblk);
  fixed_data.precond_build =  int2d( "precond_build"   , lcblk, ncblk) ;

  int2dHost  precHost =  int2dHost( "precHost" , lcblk, ncblk);
  int3dHost  ipivHost =  int3dHost( "ipivHost" , lcblk, ndof, ncblk);
  real4dHost afacHost = real4dHost( "afacHost" , lcblk, mrows, ndof, ncblk) ;
  real3dHost bblkHost = real3dHost( "bblkHost" , lcblk, ndof, ncblk);

  int iunit = 98 ;
  read_scalars_( &iunit,
                 &fixed_data.ndof,
                 &fixed_data.mrows,
                 &fixed_data.kl,
                 &fixed_data.ku ) ;

  fprintf(stderr,"ndof %d \n",fixed_data.ndof) ;
  fprintf(stderr,"mrows %d \n",fixed_data.mrows) ;
  fprintf(stderr,"kl %d \n",fixed_data.kl) ;
  fprintf(stderr,"ku %d \n",fixed_data.ku) ;


  int  *p_precond_build = precHost.data(); 
  int  *p_ipiv = ipivHost.data() ;
  real *p_afac = afacHost.data() ;
  real *p_bblk = bblkHost.data() ;

  std::memcpy( p_precond_build, precond_build_host, lcblk*ncblk*sizeof(int) ) ;
  std::memcpy( p_ipiv,          ipiv_host,          lcblk*ndof*ncblk*sizeof(int) ) ;
  std::memcpy( p_afac,          afac_host,          lcblk*mrows*ndof*ncblk*sizeof(real) ) ;
  std::memcpy( p_bblk,          bblk_host,          lcblk*ndof*ncblk*sizeof(real) ) ;

//  read_arrays_( &iunit, &lcblk, &ncblk
//               ,&fixed_data.ndof
//               ,&fixed_data.mrows
//               ,p_precond_build
//               ,p_ipiv
//               ,p_afac
//               ,p_bblk ) ;
                
  precHost.deep_copy_to(fixed_data.precond_build) ;
  ipivHost.deep_copy_to(ipiv) ;
  afacHost.deep_copy_to(afac) ;
  bblkHost.deep_copy_to(bblk) ;

}

void verify( int3d &ipiv, real4d &afac, real3d &bblk, Fixed_data &fixed_data ) {
  auto &nranks           = fixed_data.nranks          ;
  auto &ndof             = fixed_data.ndof            ;
  auto &mrows            = fixed_data.mrows           ;
  auto &lcblk            = fixed_data.lcblk           ;
  auto &ncblk            = fixed_data.ncblk           ;
  auto &myrank           = fixed_data.myrank          ;
  auto &mainproc         = fixed_data.mainproc        ;
  auto &i_beg            = fixed_data.i_beg           ;
  int  ierr;
  int  memcheck ;

// computed
  int2dHost  precHost  =  int2dHost( "precHost1" , lcblk, ncblk);
  int3dHost  ipivHost  =  int3dHost( "ipivHost1" , lcblk, ndof, ncblk);
  real4dHost afacHost  = real4dHost( "afacHost1" , lcblk, mrows, ndof, ncblk) ;
  real3dHost bblkHost  = real3dHost( "bblkHost1" , lcblk, ndof, ncblk);

  afac.deep_copy_to(afacHost) ;
  bblk.deep_copy_to(bblkHost) ;

// read in
  int2dHost  precHost2 =  int2dHost( "precHost2" , lcblk, ncblk);
  int3dHost  ipivHost2 =  int3dHost( "ipivHost2" , lcblk, ndof, ncblk);
  real4dHost afacHost2 = real4dHost( "afacHost2" , lcblk, mrows, ndof, ncblk) ;
  real3dHost bblkHost2 = real3dHost( "bblkHost2" , lcblk, ndof, ncblk);

  int iunit = 99 ;
  int *p_precond_build = precHost2.data(); 
  int *p_ipiv = ipivHost2.data() ;
  real *p_afac = afacHost2.data() ;
  real *p_bblk = bblkHost2.data() ;

  read_arrays_( &iunit, &lcblk, &ncblk
               ,&fixed_data.ndof
               ,&fixed_data.mrows
               ,p_precond_build
               ,p_ipiv
               ,p_afac
               ,p_bblk ) ;

  int n = lcblk*mrows*ndof ;
printf("compare afac n=%d \n",n) ;

  diff_( afacHost.data(), afacHost2.data(), &n ) ;

}
