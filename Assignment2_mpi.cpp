// COSC3500, Semester 2, 2021
// Assignment 2
// Main file - serial version

#include "eigensolver.h"
#include "randutil.h"
#include <string>
#include <iostream>
#include <iomanip>
#include <mpi.h>
#include <unistd.h>

// global variables to store the matrix

using namespace std;

double* M = nullptr;
int N = 0;
int world_size;
int my_rank;
int root_process=0;
 int matrix_chunk;
void partial_matrix_multiply(const double * Xin, double * inM, double * retbuf)
{
   void * dummy_null_void = NULL;
    int dummy_int = 0;
    double *myXbuff, *outgoing, *incoming;
    incoming = (double *)malloc(matrix_chunk * sizeof(double));
    int retcnt = 2*matrix_chunk; /// N) +1; // first and last are partial entries middle ones are return vector entries
    int cnt=0;
   double *datain;
cout << my_rank << " hitting partial_matrix_multiply"<< endl << flush;
    outgoing = (double *)malloc(retcnt * sizeof(double));
    myXbuff = (double *)malloc(N*sizeof(double));
   if (my_rank==0)
       datain=(double *)Xin;
   else
       datain=myXbuff;
cout << my_rank << " hitting MPI_Bcast"<< endl << flush;
   MPI_Bcast((void *)datain, N, MPI_DOUBLE, my_rank, MPI_COMM_WORLD);
cout << my_rank << " hitting MPI_Scatter inM="<<inM << " incoming=" << incoming << " matrix_chunk=" << matrix_chunk<< endl << flush;
   MPI_Scatter(inM, matrix_chunk, MPI_DOUBLE, incoming, matrix_chunk, MPI_DOUBLE, my_rank, MPI_COMM_WORLD);
cout << my_rank << " finishing MPI_Scatter"<< endl << flush;
/*
    if (my_rank != 0)
    {
       sleep(10);
cout << my_rank << " Doing MPI_Bcast rx"<< endl << flush;
       MPI_Bcast(myXbuff, N, MPI_DOUBLE, my_rank, MPI_COMM_WORLD);
cout << my_rank << " Doing MPI_Scatter rx"<< endl << flush;
       MPI_Scatter(dummy_null_void, dummy_int, MPI_DOUBLE, incoming, matrix_chunk, MPI_DOUBLE, my_rank, MPI_COMM_WORLD);
    }
    else
    {
       incoming = rootin;
cout << my_rank << " root setting point to its scattered data"<< endl << flush;
    }
*/
    int from = my_rank * matrix_chunk;
    int to = (my_rank + 1) * matrix_chunk;
    outgoing[cnt] = 0.0;

cout << my_rank << " from="<< from << " to=" << to<< " N=" << N <<endl << flush;
    for (int k = from; k < to; k++)
    {
        int col = k % N;
cout << my_rank << " col="<< col<< " k=" << k<< " N=" << N <<endl << flush;
        int row = k / N;
        if (col == 0 && k > 0)
           outgoing[++cnt]=0;
        outgoing[cnt] += myXbuff[col] * incoming[row*N+col];
   } 
cout << "cnt="<< cnt<<endl;
cout << my_rank << " Doing MPI_Gather tx"<< endl << flush;
   MPI_Gather(&outgoing, retcnt, MPI_DOUBLE, retbuf, matrix_chunk*2, MPI_DOUBLE, my_rank, MPI_COMM_WORLD);
}
// implementation of the matrix-vector multiply function
void MatrixVectorMultiply(double* Y, const double* X)
{
   void * dummy_null_void = NULL;
  
cout << my_rank << " hitting MatrixVectorMultiply"<< endl << flush;
   int dummy_int = 0;
cout << my_rank << " Doing MPI_Bcast tx"<< endl << flush;
   MPI_Bcast((void *)X, N, MPI_DOUBLE, root_process, MPI_COMM_WORLD);
cout << my_rank << " Done tx"<< endl << flush;
   int vec_block = matrix_chunk*2; //(matrix_chunk / N +1 )*N;
cout << my_rank << " vec_block "<<vec_block<<  endl << flush;
   double * ret_buffer = (double *)malloc(vec_block*sizeof(double));
   double * Xcopy =  (double *)malloc(matrix_chunk*sizeof(double)); 
   int left_over = (N*N) % world_size;
   int cnt=0;
      Y[cnt]=0;
//cout << my_rank << " Doing MPI_Scatter tx"<< endl << flush;
      //MPI_Scatter(M, matrix_chunk, MPI_DOUBLE, Xcopy, matrix_chunk, MPI_DOUBLE, root_process, MPI_COMM_WORLD);
      partial_matrix_multiply(X, M, ret_buffer);
//cout << my_rank << " Doing MPI_Gather rx"<< endl << flush;
      //MPI_Gather(dummy_null_void, dummy_int, MPI_DOUBLE, ret_buffer, N, MPI_DOUBLE, root_process, MPI_COMM_WORLD);
      double partial=0.0;
      for (int i=0; i< vec_block*world_size; i++)
      {
          if ((i+1) % vec_block == 0)
             partial = ret_buffer[i];
          else if (i % vec_block == 0)
              Y[cnt++] = ret_buffer[i] + partial;
          else
              Y[cnt++] = ret_buffer[i];
      }
      cout << "cnt = " << cnt << endl << flush;
      //Y[i] += M[i*N+j] * X[j];
}
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

   MPI_Init(&argc, &argv);

   MPI_Comm_size(MPI_COMM_WORLD, &world_size);

   matrix_chunk = (N*N) / world_size;

   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

   if (my_rank == root_process)
     cout << "matrix_chunk = " << matrix_chunk << " where N=" << N << endl << flush;
   if (my_rank != 0)
   {
     while (true)
       partial_matrix_multiply(NULL,NULL,NULL);
   }

   // Allocate memory for the matrix
   M = static_cast<double*>(malloc(N*N*sizeof(double)));

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
