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

//  module load gnu/7.2.0 gnutools mpi/openmpi3_eth

// global variables to store the matrix
//
using namespace std;

double* M = nullptr;
double* Y_chunk = nullptr;
double* M_chunk = nullptr;
int N = 0;
int world_size;
int my_rank;
int root_process=0;
int matrix_chunk;
int vector_chunk;
int matrix_remainder;
int vector_remainder;
bool thread_multiply_active = true;

void partial_matrix_multiply(double * X, double * M, double * Y_chunk, double * out_Y)
{
    int cnt=0;
    MPI_Status recv_status;

    // send & receive the X vector
    MPI_Bcast((void *)X, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int from = my_rank * matrix_chunk;
    int to = (my_rank + 1) * matrix_chunk-1;
    int real_start_row = from / N;
    int real_start_col = from % N;
    int real_end_row = to / N;
    int real_end_col = to % N;
    int whole_row_start, whole_row_end;
    double prev_partial=0.0;
    double first_partial = 0.0;
    double final_partial = 0.0;
 cout << "rank: " << my_rank << " doing " << from << " to " << to << " cell eles " << endl << flush;
    if (real_start_col != 0)
    {
        whole_row_start = (real_start_row + 1)*N;
        for (int q=from; q<whole_row_start; q++)
        {
           int pcol = q % N;
           first_partial +=  X[pcol] * M[q-from]; //M[real_start_row*N+pcol];
        }
        // Get partial sum from 'previous' process
 cout << "rank: " << my_rank << " doing Recv for src "<< my_rank-1 << " on tag " << from-1 << endl << flush;
        MPI_Recv( &prev_partial, 1, MPI_DOUBLE, my_rank-1, from-1, MPI_COMM_WORLD, &recv_status);
        Y_chunk[0] = prev_partial;
   cout << "rank="<<my_rank<<" Y_chunk["<<real_start_row<<"]=0 (carryover partial)"<< endl<<flush;

    }
    else
    {
        // chunk starts on first column so no partial sum to recv
        whole_row_start = real_start_row * N;
        cnt=-1; // account for col==0
   cout << "rank="<<my_rank<<" Y_chunk["<<real_start_row<<"]=0 (initialised accumulator)"<< endl<<flush;
        
        Y_chunk[0]=0.0;
    }

    if (real_end_col == N-1)
    {
        // chunck ends on final column so no partial sum to send
        whole_row_end = (real_end_row+1) * N;
    }
    else
    {
        // calculate partial sum for sending to 'next' process
        whole_row_end = (real_end_row) * N;
        for (int k = whole_row_end; k < to; k++)
        {
            int col = k % N;
            int row = k / N;
            final_partial += X[col] * M[matrix_chunk-to+k];
        } 
        if (my_rank != world_size-1)
        {
            MPI_Send( &final_partial, 1, MPI_DOUBLE, my_rank+1, to, MPI_COMM_WORLD);
 cout << "rank: " << my_rank << " doing Recv for src "<< my_rank-1 << " on tag " << from-1 << endl << flush;
        }
    }
      cout << "whole_row_start="<<whole_row_start<<" whole_row_end="<<whole_row_end<<endl << flush; 
    for (int k = whole_row_start; k < whole_row_end; k++)
    {
        int col = k % N;
        int row = k / N;
        if (col == 0)
        {
           Y_chunk[++cnt]=0;
        }
   cout << "rank="<<my_rank<<" col=" << col<< " row=" << row << " Y_chunk["<<cnt<<"]="<< X[col]<<" * " << M[row*N+col] << endl<<flush;
        Y_chunk[cnt] += X[col] * M[matrix_chunk-real_end_col-whole_row_end+k];//* M[row*N+col];
    } 

cout <<"rank="<<my_rank<< "chunk=" << vector_chunk <<endl << flush;
    //MPI_Barrier(MPI_COMM_WORLD);
for (int y=0;y<vector_chunk;y++)
   cout << "rank="<<my_rank<<" Y_chunk["<<y<<"]="<< Y_chunk[y] << endl<<flush;
cout << "Passed barrier"<<endl<<flush;
    MPI_Gather(Y_chunk, vector_chunk-1, MPI_DOUBLE, out_Y, vector_chunk-1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

cout <<"rank="<<my_rank<< " gather finished" << vector_chunk <<endl << flush;
    int last_start=vector_chunk * world_size;
    if (my_rank == 0 && last_start < N)
    {
          out_Y[last_start % N] = final_partial; 
          for (int i=last_start; i < N; i++)
          {
              int col = i % N;
              int row = i / N;
              if (col == 0)
                out_Y[col] = 0.0;
              out_Y[col] += X[col] * M[row*N+col];
          }
    }
}

// implementation of the matrix-vector multiply function
void MatrixVectorMultiply(double* Y, const double* X)
{

    partial_matrix_multiply((double*)X, M_chunk, Y_chunk, Y);

}

int main(int argc, char** argv)
{
    // get the current time, for benchmarking
    auto StartTime = std::chrono::high_resolution_clock::now();
    double * X;

    // get the input size from the command line
    if (argc < 2)
    {
        std::cerr << "expected: matrix size <N>\n";
        return 1;
    }
    N = std::stoi(argv[1]);
  std::cout << "Supplied: matrix size "<< N <<"\n";

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    matrix_chunk = (N*N) / world_size;
    vector_chunk = N / world_size;

    matrix_remainder  = (N*N) % world_size;
    vector_remainder  = N % world_size;

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);


    Y_chunk = static_cast<double*>(malloc(vector_chunk*sizeof(double)));
    M_chunk = static_cast<double*>(malloc(matrix_chunk*sizeof(double)));

    X = static_cast<double*>(malloc(N*sizeof(double)));

//    if (my_rank == root_process)
       cout << "matrix_chunk = " << M_chunk << " where N=" << N << endl << flush;

    if (my_rank != 0)
    {
       // All not root processes wait until they get the copy of M
       // Only need it once, as it doesnt change
       MPI_Scatter(M_chunk, matrix_chunk, MPI_DOUBLE, M_chunk, matrix_chunk, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  for (int i=my_rank*matrix_chunk;i<(my_rank+1)*matrix_chunk;i++) 
cout <<"Rank="<<my_rank<< "M_chunk["<<i/N<<","<<i%N<<"]="<<M_chunk[i-my_rank*matrix_chunk]<<endl<<flush;

      // MPI_Bcast((void *)M, N*N, MPI_DOUBLE, my_rank, MPI_COMM_WORLD);
       while (thread_multiply_active)
          partial_matrix_multiply(X, M_chunk, Y_chunk, NULL);
       exit (0);
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
cout << my_rank << " finiisdhed init"<< endl << flush;
    //MPI_Bcast((void *)M, N*N, MPI_DOUBLE, root_process, MPI_COMM_WORLD);
    MPI_Scatter(M, matrix_chunk, MPI_DOUBLE, M_chunk, matrix_chunk, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  for (int i=my_rank*matrix_chunk;i<(my_rank+1)*matrix_chunk;i++) 
cout <<"Rank="<<my_rank<< "M_chunk["<<i/N<<","<<i%N<<"]="<<M_chunk[i-my_rank*matrix_chunk]<<endl<<flush;
cout << my_rank << " finiisdhed Bcast"<< endl << flush;
    auto FinishInitialization = std::chrono::high_resolution_clock::now();

    // Call the eigensolver
    EigensolverInfo Info = eigenvalues_arpack(N, 100);

    auto FinishTime = std::chrono::high_resolution_clock::now();

    auto InitializationTime = std::chrono::duration_cast<std::chrono::microseconds>(FinishInitialization - StartTime);
    auto TotalTime = std::chrono::duration_cast<std::chrono::microseconds>(FinishTime - StartTime);

    thread_multiply_active = false;

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
    MPI_Finalize();
}
