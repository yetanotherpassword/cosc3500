#include <string>
#include <iostream>
#include <iomanip>
#include <chrono>
#include "matrix.h"
//  module load cuda/10.1 gcc #goliath
//  module load cuda #getafix

using namespace std;
// global variables to store the matrix

double* mDevice; // input matrix to multply to 
double* xDevice; // this input vector, which gives
double* yDevice; // this resultant output vector for returning to cpu from gpu

int threadsPerBlock;
int blocksPerGrid;


void checkError(cudaError_t e)
{
    if (e != cudaSuccess)
    {
        std::cerr << "CUDA error: " << int(e) << " : " << cudaGetErrorString(e) << '\n';
        abort();
    }
}

__global__
void CUDA_MatrixVectorMultiply(int nr, int nc, double* M, double* Y, const double* X)
{

    // blockDim is the number of threads in a block
    // gridDim is the number of blocks in the grid
    int xindex = blockIdx.x * blockDim.x + threadIdx.x;
    int yindex = blockIdx.y * blockDim.y + threadIdx.y;
    int xstride = blockDim.x * gridDim.x;
    int ystride = blockDim.y * gridDim.y;
    for (int i = xindex; i <= nc; i+= xstride)
    {
        Y[i] = 0;
        for (int j = yindex; j <= nr; j+= ystride)
        {
             Y[i] += M[i * nr + j] * X[j];
        }
    }
}



// implementation of the matrix-vector multiply function
void MatrixVectorMultiply(double* Y, double* X, double * M, int m_rows, int m_cols)
{
   int N=m_rows*m_cols;
   double q[20];
    // copy memory from host to device
   threadsPerBlock =16;
/*
   if (m_cols<m_rows)
   {
      N=m_cols;
      colrow='c';
   }
   else
   {
      N=m_rows;
      colrow='r';
   }
*/
   blocksPerGrid = (N + threadsPerBlock- 1) / threadsPerBlock;
   cout << "Using " << threadsPerBlock << " threadsPerBlock and blocksPerGrid = " << blocksPerGrid <<endl;
 
    //checkError(cudaMemcpy(xDevice, X, N * sizeof(double), cudaMemcpyHostToDevice));

    checkError(cudaMemcpy(xDevice, X, m_rows * sizeof(double), cudaMemcpyHostToDevice));
    checkError(cudaMemcpy(mDevice, M, m_cols* m_rows * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_MatrixVectorMultiply<<<blocksPerGrid, threadsPerBlock>>> (m_rows, m_cols, mDevice, yDevice, xDevice);
    checkError(cudaDeviceSynchronize());
    checkError(cudaMemcpy(Y, yDevice, m_cols * sizeof(double), cudaMemcpyDeviceToHost));
    //for (int i=0;i<N;i++)
    //cout <<"Y["<<i<<"]="<<Y[i]<<endl;


}

#if 0
class Matrix {
   public:
      double * e;   // Length of a box
      int rows;
      int cols;
   Matrix(int r, int c);
   multiplyto(Matrix m);
   ~Matrix();
};
Matrix::multiplyto(Matrix & m, Matrix & o)
{
  if (m.rows == cols)
  {
    // then can do it
    // check if output matrix is right size
      if ((rows == o.rows) && (m.cols == o.cols))
      {
           for (int j=0;j<o.cols;j++)
           {
             // j is same as 'm.cols'
             for (int p=0;p<o.rows;p++)
             {
                  // p is same as 'rows'
                  o.e[j+p*o.cols] = 0;
                  for (int c=0;c<cols;c++)
                  {
                      o.e[j+p*o.cols] += e[p*cols+c]*m.e[c*m.rows+j]
                  }
              }
           }
      }
  }
}
Matrix::Matrix(r,c)
{
    rows=r;
    cols=c;
    e= new double [(r+1)*(c+1)];
}
Matrix::~Matrix()
{
    delete[] e;
}
#endif
int main(int argc, char** argv)
{
   // get the current time, for benchmarking
   auto StartTime = std::chrono::high_resolution_clock::now();

  Matrix V(1,4); 
  Matrix M(4,3); 
  Matrix O(1,3); 
   // Allocate memory for the matrix
//   M = static_cast<double*>(malloc(N*N*sizeof(double)));

   // seed the random number generator to a known state
//   randutil::seed(4);  // The standard random number.  https://xkcd.com/221/

   // Initialize the matrix.  This is a matrix from a Gaussian Orthogonal Ensemble.
   // The matrix is symmetric.
   // The diagonal entries are gaussian distributed with variance 2.
   // The off-diagonal entries are gaussian distributed with variance 1.
   for (int i = 0; i < V.rows; ++i)
   {
      for (int j = 0; j < V.cols; ++j)
      {
         V.e[i*V.cols + j] =  i+j;//( (double) rand())/(double) RAND_MAX;;
      }
   }
   for (int i = 0; i < M.rows; ++i)
   {
      for (int j = 0; j < M.cols; ++j)
      {
         M.e[i*M.cols + j] =  i+j;//( (double) rand())/(double) RAND_MAX;;
      }
   }
   // allocate memory on the device
#if 0
// Host code
int width = 64, height = 64;
float* devPtr;
size_t pitch;
cudaMallocPitch(&devPtr, &pitch,
                N * sizeof(double), N);
MyKernel<<<100, 512>>>(devPtr, pitch, width, height);

// Device code
__global__ void MyKernel(float* devPtr,
                         size_t pitch, int width, int height)
{
    for (int r = 0; r < height; ++r) {
        float* row = (float*)((char*)devPtr + r * pitch);
        for (int c = 0; c < width; ++c) {
            float element = row[c];
        }
    }
}
#endif

   for (int i=0;i<V.cols;i++)
      O.e[i]=0;
   checkError(cudaMalloc(&mDevice, M.rows * M.cols * sizeof(double)));
   checkError(cudaMalloc(&xDevice, M.rows * sizeof(double)));
   checkError(cudaMalloc(&yDevice, M.cols * sizeof(double)));

   checkError(cudaMemcpy(mDevice, M.e, M.rows * M.cols * sizeof(double), cudaMemcpyHostToDevice));

   auto FinishInitialization = std::chrono::high_resolution_clock::now();

   // Call the eigensolver

   auto FinishTime = std::chrono::high_resolution_clock::now();

   auto InitializationTime = std::chrono::duration_cast<std::chrono::microseconds>(FinishInitialization - StartTime);
   auto TotalTime = std::chrono::duration_cast<std::chrono::microseconds>(FinishTime - StartTime);

   MatrixVectorMultiply( O.e,  V.e,  M.e, M.rows, M.cols);


   cout << endl << "V=" << endl;
   for (int i=0;i<V.cols;i++)
     cout << V.e[i] << "  ";
   cout << endl;

   cout << endl << "M=" << endl;
   for (int i=0;i<M.rows;i++)
   {
     for (int j=0;j<M.cols;j++)
       cout << M.e[i*M.cols+j] << "  ";
     cout << endl;
   }
   cout << endl << "O.cols="<< O.cols << endl;

   M.multiplyto(V,O);


   for (int i=0;i<O.cols;i++)
     cout << O.e[i] << "  ";
   cout << endl;
   std::cout << "Total time:                             " << std::setw(12) << TotalTime.count() << " us\n";
   std::cout << "Time spent in initialization:           " << std::setw(12) << InitializationTime.count() << " us\n";

   std::cout << "Time per matrix-vector multiplication:, " << std::endl;

   //std::cout << std::setw(12) << TotalTime.count() << ",";
   //std::cout << std::setw(12) << InitializationTime.count() << ",";

   // free memory
   checkError(cudaFree(mDevice));
   checkError(cudaFree(xDevice));
   checkError(cudaFree(yDevice));
}
