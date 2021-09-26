// COSC3500, Semester 2, 2021
// Assignment 2
// Implementation file for the eigensolver functions.

#include "eigensolver.h"
#include <cstdint>
#include <cassert>
#include <iostream>

#if defined(FORTRAN_DOUBLE_UNDERSCORE)
#define F77NAME(x) x##__
#else
#define F77NAME(x) x##_
#endif

// ARPACK wrappers
namespace ARPACK
{

// correspondence between fortran types and C++ types
using integer = std::int32_t;
using real    = double;
using logical = std::int32_t;

struct iparam_t
{
   integer ishift;
   integer mxiter;
   integer nconv;
   integer mode;
   integer np;
   integer numop;
   integer numopb;
   integer numreo;

   iparam_t() : ishift(1), mxiter(0), nconv(0), mode(1), np(0), numop(0), numopb(0), numreo(0) {}

   void get_from_raw(int const* raw);
   void put_to_raw(int* raw);

   static int const size = 11;
};

void iparam_t::get_from_raw(int const* raw)
{
   ishift = raw[0];
   mxiter = raw[2];
   nconv  = raw[4];
   mode   = raw[6];
   np     = raw[7];
   numop  = raw[8];
   numopb = raw[9];
   numreo = raw[10];
}

void iparam_t::put_to_raw(int* raw)
{
   raw[0] = ishift;
   raw[2] = mxiter;
   raw[3] = 1;        // NB must be set to 1 by ARPACK spec
   raw[4] = nconv;
   raw[6] = mode;
   raw[7] = np;
   raw[8] = numop;
   raw[9] = numopb;
   raw[10] = numreo;
}

// pointers into the workd and workl arrays for double precision variants.
// These are converted between 0-based arrays
// in dn_ipntr_t, and 1-based arrays in the fortran array.
struct dn_ipntr_t
{
   integer x;
   integer y;
   integer bx;
   integer next_free;
   integer t;
   integer ritz_values;
   integer ritz_estimates;
   integer ritz_values_original;
   integer ritz_error_bounds;
   integer t_eigenvectors;
   integer np_shifts;

   dn_ipntr_t() : x(0), y(0), bx(0), next_free(0), t(0), ritz_values(0),
               ritz_estimates(0), ritz_values_original(0), ritz_error_bounds(0),
               t_eigenvectors(0), np_shifts(0) {}

   void get_from_raw(int const* raw);
   void put_to_raw(int* raw);

   static int const size = 11;
};

void dn_ipntr_t::get_from_raw(int const* raw)
{
   x                    = raw[0]-1;
   y                    = raw[1]-1;
   bx                   = raw[2]-1;
   next_free            = raw[3]-1;
   t                    = raw[4]-1;
   ritz_values          = raw[5]-1;
   ritz_estimates       = raw[6]-1;
   ritz_values_original = raw[7]-1;
   ritz_error_bounds    = raw[8]-1;
   t_eigenvectors       = raw[9]-1;
   np_shifts            = raw[10]-1;
}

void dn_ipntr_t::put_to_raw(int* raw)
{
   raw[0] = x+1;
   raw[1] = y+1;
   raw[2] = bx+1;
   raw[3] = next_free+1;
   raw[4] = t+1;
   raw[5] = ritz_values+1;
   raw[6] = ritz_estimates+1;
   raw[7] = ritz_values_original+1;
   raw[8] = ritz_error_bounds+1;
   raw[9] = t_eigenvectors+1;
   raw[10] = np_shifts+1;
}

namespace FORTRAN
{

extern "C"
{
void F77NAME(dsaupd)(integer* ido, char const* bmat, integer const* n, char const* which,
                     integer const* nev, double const* tol, double* resid,
                     integer const* ncv, double* V, integer const* ldv,
                     integer* iparam, integer* ipntr, double* workd,
                     double* workl, integer const* lworkl, integer* info);

void F77NAME(dseupd)(logical *rvec, char *HowMny, logical *select,
                     double *d, double *Z, integer *ldz,
                     double *sigma, char *bmat, integer *n,
                     char const* which, integer const* nev, double const* tol,
                     double *resid, integer const* ncv, double *V,
                     integer *ldv, integer *iparam, integer *ipntr,
                     double* workd, double* workl,
                     integer const* lworkl, integer *info);
} // extern "C"
} // namespace FORTRAN

void dsaupd(integer* ido, char bmat, integer n, char const* which,
            integer nev, double tol, double* resid,
            integer ncv, double* V, integer ldv,
            iparam_t* iparam, dn_ipntr_t* ipntr, double* workd,
            double* workl, integer lworkl, integer* info)
{
   integer raw_iparam[iparam_t::size];
   integer raw_ipntr[dn_ipntr_t::size];

   iparam->put_to_raw(raw_iparam);
   ipntr->put_to_raw(raw_ipntr);

   FORTRAN::F77NAME(dsaupd)(ido, &bmat, &n, which,
                  &nev, &tol, resid,
                  &ncv, V, &ldv,
                  raw_iparam, raw_ipntr, workd,
                  workl, &lworkl, info);

   iparam->get_from_raw(raw_iparam);
   ipntr->get_from_raw(raw_ipntr);
}

void dseupd(bool rvec, char HowMny, logical* select,
            double* d, double* z, integer ldz,
            double sigma, char bmat, integer n,
            char const* which, integer nev, double tol,
            double* resid, integer ncv, double* V,
            integer ldv, iparam_t* iparam, dn_ipntr_t* ipntr,
            double *workd, double *workl,
            integer lworkl, integer* info)
{
   logical rvec_raw = rvec;
   integer raw_iparam[iparam_t::size];
   integer raw_ipntr[dn_ipntr_t::size];

   iparam->put_to_raw(raw_iparam);
   ipntr->put_to_raw(raw_ipntr);

   FORTRAN::F77NAME(dseupd)(&rvec_raw, &HowMny, select,
                           d, z, &ldz,
                           &sigma, &bmat, &n,
                           which, &nev, &tol,
                           resid, &ncv, V,
                           &ldv, raw_iparam, raw_ipntr,
                           workd, workl,
                           &lworkl, info);

   iparam->get_from_raw(raw_iparam);
   ipntr->get_from_raw(raw_ipntr);
}

} // namespace ARPACK

EigensolverInfo
eigenvalues_arpack(int n, int nev, double tol)
{
   auto StartEigensolver = std::chrono::high_resolution_clock::now();
   // arpack parameters
   int ido = 0;                   // first call
   char bmat = 'I';               // standard eigenvalue problem
   char which[3] = "LM";          // largest magnitude
   nev = std::min(nev, n-2);      // ARPACK can't calulate more than n-2 eigenvalues
   std::vector<double> resid(n);  // residual
   int ncv = std::min(2*nev, n);  // length of the arnoldi sequence
   std::vector<double> v(n*ncv);  // workshpace for the Krylov vectors
   int const ldv = n;
   ARPACK::iparam_t iparam;
   iparam.ishift = 1;      // exact shifts
   iparam.mxiter = 10000;  // maximum number of arnoldi iterations (restarts?)
   iparam.mode = 1;        // standard eigenvalue problem
   ARPACK::dn_ipntr_t ipntr;
   std::vector<double> workd(3*n);
   int const lworkl = ncv * (ncv * 8);
   std::vector<double> workl(lworkl);
   int info = 0;  // no initial residual

   int NumMultiplies = 0;
   auto MultiplyTime = std::chrono::microseconds::zero();

   ARPACK::dsaupd(&ido, bmat, n, which, nev, tol, &resid[0], ncv,
                  &v[0], ldv, &iparam, &ipntr, &workd[0],
                  &workl[0], lworkl, &info);
   assert(info >= 0);

   while (ido != 99)
   {
      if (ido == -1 || ido == 1)
      {
         // evaluate Y = Matrix * X
         auto StartMultiply = std::chrono::high_resolution_clock::now();
         MatrixVectorMultiply(&workd[ipntr.y], &workd[ipntr.x]);
         auto EndMultiply = std::chrono::high_resolution_clock::now();
         ++NumMultiplies;
         MultiplyTime += std::chrono::duration_cast<std::chrono::microseconds>(EndMultiply-StartMultiply);
      }
      else
      {
         std::cerr << "unexpected reverse communication operation:" << ido << '\n';
         abort();
      }

      ARPACK::dsaupd(&ido, bmat, n, which, nev, tol, &resid[0], ncv,
                     &v[0], ldv, &iparam, &ipntr, &workd[0],
                     &workl[0], lworkl, &info);
      assert(info >= 0);
   }

   // get the eigenvalues
   bool rvec = false; // no eigenvectors
   char howmny = 'A'; // all computed ritz vectors
   std::vector<ARPACK::logical> select(ncv);
   std::vector<double> d(nev);
   std::vector<double> z(1); // output array - not used since we are not calculating eigenvectors
   int ldz = n;
   double sigma;   // not referenced
   ARPACK::dseupd(rvec, howmny, &select[0], &d[0], &z[0], ldz, sigma,
                  bmat, n, which, nev, tol, &resid[0], ncv, &v[0], ldv,
                  &iparam, &ipntr, &workd[0],
                  &workl[0], lworkl, &info);
   assert(info >= 0);

   auto EndEigensolver = std::chrono::high_resolution_clock::now();
   EigensolverInfo Result;
   Result.Eigenvalues = d;
   Result.NumMultiplies = NumMultiplies;
   Result.TimeInEigensolver = std::chrono::duration_cast<std::chrono::microseconds>(EndEigensolver-StartEigensolver);
   Result.TimeInMultiply = MultiplyTime;

   return Result;
}
