// COSC3500, Semester 2, 2021
// Assignment 2
// Main file - serial version

#include "eigensolver.h"
#include "randutil.h"
#include <string>
#include <iostream>
#include <iomanip>
#include <omp.h>

// global variables to store the matrix
using namespace std;

double* M = nullptr;
int N = 0;
int NUM_THREADS=0;
string msg="";


   void update_cyclic_params ( int thread_num, int num_of_threads, int & from, int & to, int orig_from, int orig_to)
   {
       int thread_len = (orig_to - orig_from) / num_of_threads;
       from = thread_num * thread_len + orig_from;
       to = (thread_num +1) * thread_len  + orig_from;
       if (thread_num == num_of_threads - 1)
       {
           to += (orig_to - orig_from) - (num_of_threads * thread_len);
           msg += "Reqested:" + to_string(NUM_THREADS) + " Got " + to_string(num_of_threads)+ "\n";
      //     cout << "Num of threads : " << num_of_threads << endl;
       }
   }

// implementation of the matrix-vector multiply function
void MatrixVectorMultiply(double* Y, const double* X)
{
   omp_set_num_threads(NUM_THREADS);
   #pragma omp parallel
   {
      int id = omp_get_thread_num();
      int nthrds = omp_get_num_threads();
      int from, to;
      update_cyclic_params( id, nthrds, from, to, 0, N);
      for (int i = from; i < to; ++i)
      {
         Y[i] = 0;
         for (int j = 0; j < N; ++j)
         {
            Y[i] += M[i*N+j] * X[j];
         }
      }
   }
}

void MatrixVectorMultiply3(double* Y, const double* X)
{
   omp_set_num_threads(NUM_THREADS);
   {
      int id = omp_get_thread_num();
      int nthrds = omp_get_num_threads();
      int from, to;
      update_cyclic_params( id, nthrds, from, to, 0, N);
   #pragma omp parallel
   {
      #pragma omp for
      for (int i = from; i < to; ++i)
      {
         Y[i] = 0;
         for (int j = 0; j < N; ++j)
         {
            Y[i] += M[i*N+j] * X[j];
         }
      }
    }
   }
}


// implementation of the matrix-vector multiply function
void MatrixVectorMultiply_poor(double* Y, const double* X)
{
   omp_set_num_threads(NUM_THREADS);
   omp_set_schedule( omp_sched_dynamic, 80 );
   for (int i = 0; i < N; ++i)
   {
      double tmp = 0;
      #pragma omp parallel for reduction(+:tmp)
      for (int j = 0; j < N; ++j)
      {
         tmp += M[i*N+j] * X[j];
      //if (omp_get_thread_num()==0) cout << "No fo threads=" << omp_get_num_threads()<<endl;
      }
      Y[i] = tmp;
   }
}

// implementation of the matrix-vector multiply function
void MatrixVectorMultiply_orig(double* Y, const double* X)
{
   for (int i = 0; i < N; ++i)
   {
      Y[i] = 0;
      for (int j = 0; j < N; ++j)
      {
         Y[i] += M[i*N+j] * X[j];
      }
   }
}



int main(int argc, char** argv)
{
   // get the current time, for benchmarking
   auto StartTime = std::chrono::high_resolution_clock::now();
   bool dbg=false;
   // get the input size from the command line
   if (argc < 3)
   {
      std::cerr << "expected: matrix size <N> <Threads>\n";
      return 1;
   } 
   N = std::stoi(argv[1]);
   NUM_THREADS = std::stoi(argv[2]);
   if (argc > 3)
     dbg=true;

   // Allocate memory for the matrix
   M = static_cast<double*>(malloc(N*N*sizeof(double)));

   // seed the random number generator to a known state
   randutil::seed(4);  // The standard random number.  https://xkcd.com/221/

   // Initialize the matrix.  This is a matrix from a Gaussian Orthogonal Ensemble.
   // The matrix is symmetric.
   // The diagonal entries are gaussian distributed with variance 2.
   // The off-diagonal entries are gaussian distributed with variance 1.
   for (int i = 0 ; i < N; ++i)
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
   if (dbg)
      cout << msg << endl;
}
