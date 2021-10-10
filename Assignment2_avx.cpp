// COSC3500, Semester 2, 2021
// Assignment 2
// Main file - serial version

#include "eigensolver.h"
#include "randutil.h"
#include <string>
#include <iostream>
#include <iomanip>
#include <string.h>
#include <immintrin.h> 
#include <fmaintrin.h>

#define ALIGN 64
// global variables to store the matrix
typedef double v4df __attribute__((vector_size(4 * sizeof(double))));
double* M = nullptr;

v4df load_4_doubles_intel(const double *p) { return _mm256_loadu_pd(p); }

v4df avx_constant(const double *p) { return _mm256_setr_pd( p[0], p[1], p[2], p[3] ); }

int N = 0;
__m256d* quad_matrix;
int quad_double_mat_size;
int quad_double_vec_size;

// implementation of the matrix-vector multiply function
void MatrixVectorMultiply(double* Y, const double* X)
{
   __m256d localX;
   for (int i = 0; i < quad_double_vec_size; i++)
   {
       __m256d localY = _mm256_setzero_pd();
       for (int j = 0; j < quad_double_vec_size; j++)  // doubles are 64bit, so doing  4 at a tiem with __m256d type
       {
           localX =_mm256_loadu_pd (&X[j*4]);
           
           localY = _mm256_fmadd_pd (quad_matrix[i*quad_double_vec_size+j], localX, localY);
       }
       _mm256_storeu_pd (&Y[i*4], localY);
   }

}

        // _mm256_storeu_pd(X[i*quad_double_vec_size*4+j*4], 
//__m256d _mm256_loadu_pd
         //Y[i] += M[i*N+j] * X[j];

int main(int argc, char** argv)
{
   // get the current time, for benchmarking
   auto StartTime = std::chrono::high_resolution_clock::now();

   // get the input size from the command line
   if (argc < 2)
   {
      std::cerr << "expected: matrix size <N>\n";
      return 1;
   }
   N = std::stoi(argv[1]);

   quad_double_vec_size = N / 4;
   if (N % 4 != 0)
      quad_double_vec_size++;

   quad_double_mat_size =  quad_double_vec_size *  quad_double_vec_size;
      
   // Allocate memory for the matrix
   //M = static_cast<double*>(malloc(N*N*sizeof(double)));

   quad_matrix = static_cast<__m256d*> (aligned_alloc(ALIGN, sizeof(__m256d) *  quad_double_mat_size));
   M =  (double *) quad_matrix;
   memset(M, 0,  sizeof(double)* 4 * quad_double_mat_size);
  // quad_matrix = static_cast<__m256d*> (M);

   // seed the random number generator to a known state
   randutil::seed(4);  // The standard random number.  https://xkcd.com/221/

   // Initialize the matrix.  This is a matrix from a Gaussian Orthogonal Ensemble.
   // The matrix is symmetric.
   // The diagonal entries are gaussian distributed with variance 2.
   // The off-diagonal entries are gaussian distributed with variance 1.
   for (int i = 0; i < N; ++i)
   {
      M[i*N+i] = std::sqrt(2.0) * randutil::randn();
      for (int j = i+1; j < N; ++j)
      {
         M[i*N + j] = M[j*N + i] = randutil::randn();
      }
   }
   auto FinishInitialization = std::chrono::high_resolution_clock::now();

   // Call the eigensolver
   EigensolverInfo Info = eigenvalues_arpack(N, 100);

   auto FinishTime = std::chrono::high_resolution_clock::now();

   auto InitializationTime = std::chrono::duration_cast<std::chrono::microseconds>(FinishInitialization - StartTime);
   auto TotalTime = std::chrono::duration_cast<std::chrono::microseconds>(FinishTime - StartTime);

   std::cout << "Obtained " << Info.Eigenvalues.size() << " eigenvalues.\n";
   std::cout << "The largest eigenvalue is: " << std::setw(16) << std::setprecision(12) << Info.Eigenvalues.back() << '\n';
   std::cout << "Total time:                             " << std::setw(12) << TotalTime.count() << " us\n";
   std::cout << "Time spent in initialization:           " << std::setw(12) << InitializationTime.count() << " us\n";
   std::cout << "Time spent in eigensolver:              " << std::setw(12) << Info.TimeInEigensolver.count() << " us\n";
   std::cout << "   Of which the multiply function used: " << std::setw(12) << Info.TimeInMultiply.count() << " us\n";
   std::cout << "   And the eigensolver library used:    " << std::setw(12) << (Info.TimeInEigensolver - Info.TimeInMultiply).count() << " us\n";
   std::cout << "Total serial (initialization + solver): " << std::setw(12) << (TotalTime - Info.TimeInMultiply).count() << " us\n";
   std::cout << "Number of matrix-vector multiplies:     " << std::setw(12) << Info.NumMultiplies << '\n';
   std::cout << "Time per matrix-vector multiplication:  " << std::setw(12) << (Info.TimeInMultiply / Info.NumMultiplies).count() << " us\n";

   // free memory
   free(M);
}
