// COSC3500, Semester 2, 2021
// Assignment 2
// Main file - CUDA version
// Built under Visual Studio 2019 (Community Ed) & Nvida Tool Kit 11.4
// And using ARPACK windows prebuilt binaries from http://wo80.bplaced.net/math/packages.html namely ARPACK 3.7.0 and its dependency SUPERLU 5.2.1
// Once all installed/unzipped
// Click START menu on Windows10, select "x64 Native Tools Command Prompt for Visual Studio 2019"
// nvcc --compile eigensolver.cpp -o eigensolver.o
// nvcc --compile randutil.cpp -o randutil.o
// nvcc - O2 --gpu-architecture=sm_35 -Wno-deprecated-gpu-targets Assignment2_cuda.cu eigensolver.o randutil.o -l ..\arpack\arpack-3.7.0-shared\arpack-3.7.0-shared\x64\libarpack -o Assignment2_cuda.exe
#include "eigensolver.h"
#include "randutil.h"
#include <string>
#include <iostream>
#include <iomanip>

//  For goliath (getafix - not required)
//  module load cuda/10.1 gcc

// global variables to store the matrix

double* M = nullptr;
int N = 0;
double* mDevice; // input matrix to multply to 
double* xDevice; // this input vector, which gives
double* yDevice; // this resultant output vector for returning to cpu from gpu

int Threads = 256;
int Blocks;


void checkError(cudaError_t e)
{
    if (e != cudaSuccess)
    {
        std::cerr << "CUDA error: " << int(e) << " : " << cudaGetErrorString(e) << '\n';
        abort();
    }
}
/*
__global__
void add(int n, double* x, double const* y)
{
    // blockDim is the number of threads in a block
    // gridDim is the number of blocks in the grid
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
    {
        x[i] = x[i] + y[i];
    }

    // ALT version in v3
    // This version will fail if the number of blocks is insufficient to cover the whole array
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n)
    {
        for (int i = 0; i < N; i++)
            sum += vec[i] * mat[(i * M) + tid];
        out[tid] = sum;
    }
}
*/

__global__
void CUDA_MatrixVectorMultiply(int n, double* M, double* Y, const double* X)
{


    // blockDim is the number of threads in a block
    // gridDim is the number of blocks in the grid
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    //int stride = blockDim.x * gridDim.x;
    if (index < n)
    {
        Y[index] = 0;
        if (index < n)
        {
            for (int j = 0; j < n; ++j)
            {
                Y[index] += M[index * n + j] * X[j];
            }
        }
    }
}

/*
    void kernel(float* vec, float* mat, float* out, const int N, const int M) {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        float sum = 0;
        if (tid < M) {
            for (int i = 0; i < N; i++)
                sum += vec[i] * mat[(i * M) + tid];
            out[tid] = sum;
        }
    }
    */
 


// implementation of the matrix-vector multiply function
void MatrixVectorMultiply(double* Y, const double* X)
{

    // copy memory from host to device
 
    //checkError(cudaMemcpy(xDevice, X, N * sizeof(double), cudaMemcpyHostToDevice));

    CUDA_MatrixVectorMultiply<<<Blocks, Threads >>> (N, mDevice, xDevice, yDevice);
    checkError(cudaDeviceSynchronize());
    checkError(cudaMemcpy(Y, yDevice, N * sizeof(double), cudaMemcpyDeviceToHost));


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
   
   Blocks = (N + Threads - 1) / Threads;

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
   // allocate memory on the device

   checkError(cudaMalloc(&mDevice, N * N * sizeof(double)));
   checkError(cudaMalloc(&xDevice, N * sizeof(double)));
   checkError(cudaMalloc(&yDevice, N * sizeof(double)));

   checkError(cudaMemcpy(mDevice, M, N * N * sizeof(double), cudaMemcpyHostToDevice));

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
