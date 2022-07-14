#define LCBLK   32
#define CVEC    LCBLK
#define NCBLK_G 100

#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include "const.h"
#include "pnetcdf.h"
#include <ctime>
#include <chrono>

// We're going to define all arrays on the host because this doesn't use parallel_for
typedef yakl::Array<int   ,1,yakl::memDevice> int1d;
typedef yakl::Array<int   ,2,yakl::memDevice> int2d;
typedef yakl::Array<int   ,3,yakl::memDevice> int3d;
typedef yakl::Array<int   ,4,yakl::memDevice> int4d;
typedef yakl::Array<real  ,1,yakl::memDevice> real1d;
typedef yakl::Array<real  ,2,yakl::memDevice> real2d;
typedef yakl::Array<real  ,3,yakl::memDevice> real3d;
typedef yakl::Array<real  ,4,yakl::memDevice> real4d;
typedef yakl::Array<double,1,yakl::memDevice> doub1d;
typedef yakl::Array<double,2,yakl::memDevice> doub2d;
typedef yakl::Array<double,3,yakl::memDevice> doub3d;
typedef yakl::Array<double,4,yakl::memDevice> doub4d;

typedef yakl::Array<real   const,1,yakl::memDevice> realConst1d;
typedef yakl::Array<real   const,2,yakl::memDevice> realConst2d;
typedef yakl::Array<real   const,3,yakl::memDevice> realConst3d;
typedef yakl::Array<double const,1,yakl::memDevice> doubConst1d;
typedef yakl::Array<double const,2,yakl::memDevice> doubConst2d;
typedef yakl::Array<double const,3,yakl::memDevice> doubConst3d;

// Some arrays still need to be on the host, so we will explicitly create Host Array typedefs
typedef yakl::Array<int   ,1,yakl::memHost> int1dHost;
typedef yakl::Array<int   ,2,yakl::memHost> int2dHost;
typedef yakl::Array<int   ,3,yakl::memHost> int3dHost;
typedef yakl::Array<int   ,4,yakl::memHost> int4dHost;
typedef yakl::Array<real  ,1,yakl::memHost> real1dHost;
typedef yakl::Array<real  ,2,yakl::memHost> real2dHost;
typedef yakl::Array<real  ,3,yakl::memHost> real3dHost;
typedef yakl::Array<real  ,4,yakl::memHost> real4dHost;
typedef yakl::Array<double,1,yakl::memHost> doub1dHost;
typedef yakl::Array<double,2,yakl::memHost> doub2dHost;
typedef yakl::Array<double,3,yakl::memHost> doub3dHost;

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
  realConst2d precond_build;       //   Dimensions: (lcblk, ncblk)
  realConst2d ju;                  //   Dimensions: (lcblk, ncblk)
  realConst2d jp;                  //   Dimensions: (lcblk, ncblk)
};

///////////////////////////////////////////////////////////////////////////////////////
// Variables that are dynamics over the course of the simulation
///////////////////////////////////////////////////////////////////////////////////////
#if 0
void finalize             ( );
YAKL_INLINE void injection            ( real x , real z , real &r , real &u , real &w , real &t , real &hr , real &ht );
YAKL_INLINE void density_current      ( real x , real z , real &r , real &u , real &w , real &t , real &hr , real &ht );
YAKL_INLINE void gravity_waves        ( real x , real z , real &r , real &u , real &w , real &t , real &hr , real &ht );
YAKL_INLINE void thermal              ( real x , real z , real &r , real &u , real &w , real &t , real &hr , real &ht );
YAKL_INLINE void collision            ( real x , real z , real &r , real &u , real &w , real &t , real &hr , real &ht );
YAKL_INLINE void hydro_const_theta    ( real z                    , real &r , real &t );
YAKL_INLINE void hydro_const_bvfreq   ( real z , real bv_freq0    , real &r , real &t );
YAKL_INLINE real sample_ellipse_cosine( real x , real z , real amp , real x0 , real z0 , real xrad , real zrad );
void output               ( realConst3d state , real etime , int &num_out , Fixed_data const &fixed_data );
void ncwrap               ( int ierr , int line );
void perform_timestep     ( real3d const &state , real dt , int &direction_switch , Fixed_data const &fixed_data );
void semi_discrete_step   ( realConst3d state_init , real3d const &state_forcing , real3d const &state_out , real dt , int dir , Fixed_data const &fixed_data );
void compute_tendencies_x ( realConst3d state , real3d const &tend , real dt , Fixed_data const &fixed_data );
void compute_tendencies_z ( realConst3d state , real3d const &tend , real dt , Fixed_data const &fixed_data );
void set_halo_values_x    ( real3d const &state  , Fixed_data const &fixed_data );
void set_halo_values_z    ( real3d const &state  , Fixed_data const &fixed_data );
void reductions           ( realConst3d state , double &mass , double &te , Fixed_data const &fixed_data );
#endif

//Declaring the functions defined after "main"
void init                 ( int3d &ipiv, real4d &afac, real3d &bblk, real &dt , Fixed_data &fixed_data );
YAKL_INLINE void vdgbtf2( int ib, int n, int kl, int ku,
                          real4d const &ab,
                          int ldab,
                          int3d const &ipiv,
                          Fixed_data const &fixed_data,
                          yakl::InnerHandler & inner_handler ) ;


///////////////////////////////////////////////////////////////////////////////////////
// THE MAIN PROGRAM STARTS HERE
///////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  MPI_Init(&argc,&argv);
  yakl::init();
  {
    Fixed_data fixed_data;
    int3d  ipiv;   // lcblk, ndof, ncblk
    real4d afac;   // lcblk, mrows, ndof, ncblk
    real3d bblk;   // lcblk, ndof, ncblk
    real2d tempv;  // lcblk, ncblk
    real dt;                    //Model time step (seconds)

    // Init allocates the state and hydrostatic arrays hy_*
    init( ipiv, afac, bblk, dt, fixed_data );

    parallel_outer( "vdgbtf2", 
                          Bounds<1>( fixed_data.ncblk ) , 
                          YAKL_LAMBDA ( int ib, yakl::InnerHandler inner_handler )
    {
      vdgbtf2( ib, 
               fixed_data.ndof,
               fixed_data.kl, fixed_data.ku, 
               afac, 
               fixed_data.mrows,
               ipiv,
               fixed_data,
               inner_handler ) ;
    }, yakl::LaunchConfig<LCBLK>() ) ;

  }
  yakl::finalize();
  MPI_Finalize();
}

YAKL_INLINE void vdgbtf2( int ib, int n, int kl, int ku,
                          real4d const &ab,
                          int ldab,
                          int3d const &ipiv,
                          Fixed_data const &fixed_data,
                          yakl::InnerHandler & inner_handler )
{
  int kv = ku + kl ;
  if ( n == 0 ) return ; // quick return if possible
  for ( int j = ku+2 ; j <= (kv<n)?kv:n ; j++ ) { 
    parallel_inner( Bounds<2>({kv-j+2,kl},LCBLK),
                    [&] (int i, int ie )
      {
        ab(ib,j,i,ie) = 0. ;
      }, inner_handler
    ) ;
  }
}

extern "C"{
void read_scalars_(int*, int*, int*, int*, int*) ;
}

void init( int3d &ipiv, real4d &afac, real3d &bblk, real &dt , Fixed_data &fixed_data ) {
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

  ierr = MPI_Comm_size(MPI_COMM_WORLD,&nranks);
  ierr = MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

  ndof  = 145 ;
  mrows = 70 ;
  lcblk = LCBLK ;
  ncblk = NCBLK_G / nranks ;

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

  ipiv          = int3d( "ipiv"  , ncblk, ndof, lcblk);
  afac          = real4d( "afac" , ncblk, ndof, mrows, lcblk) ;
  bblk          = real3d( "bblk" , ncblk, ndof, lcblk);
  fixed_data.ju = real2d( "ju"   , ncblk, lcblk) ;
  fixed_data.jp = real2d( "jp"   , ncblk, lcblk) ;

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

}

