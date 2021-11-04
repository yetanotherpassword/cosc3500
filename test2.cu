#include <iostream>
#include <armadillo>
using namespace std;
using namespace arma;
//nvcc --gpu-architecture=sm_35 -Wno-deprecated-gpu-targets -std=c++11 -g -Iarmadillo-10.6.2/include/   test2.cu  -l cublas_static -l lapack_static -L/usr/lib64  -o test2

void checkError(cudaError_t e)
{
    if (e != cudaSuccess)
    {
        std::cerr << "CUDA error: " << int(e) << " : " << cudaGetErrorString(e) << '\n';
        abort();
    }
}


__global__ void gen_matvec(double *A, double*x, double*y, const int m, const int n)
{
  unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
  if ( xIndex < n ){
    double c = 0.0f;
    for(int i=0; i<m; i++)
      c = c + x[i] * A[xIndex + n * i];
    y[xIndex] = c;
  }
}

float matVecNaive (double * out, double * in, double * A, const int m, const int n) {

  // set up threading and blocking variables
  double * dev_A;
  double * dev_in;
  double * dev_out;
  cudaDeviceProp dp;
  cudaGetDeviceProperties(&dp,0);
  unsigned int max_threads_per_block = dp.maxThreadsPerBlock;
cout << "max_threads_per_block=" << max_threads_per_block << endl;
  int threads_perblockm = min(m, max_threads_per_block);
cout << "threads_perblockm=" << threads_perblockm << endl;
  dim3 threadsPerBlockm(threads_perblockm);
  int num_blocksm = (int)ceil((double)m/(double)threads_perblockm);
cout << "num_blocksm=" << num_blocksm << endl;
  dim3 numBlocksm(num_blocksm);

  // set up timing
  cudaEvent_t start, stop;
  float time;
  checkError(cudaEventCreate(&start));
  checkError(cudaEventCreate(&stop));
  checkError(cudaEventRecord(start,0));

 checkError(cudaMalloc( &dev_A, m*n*sizeof(double)));
 checkError(cudaMalloc( &dev_in, m*sizeof(double)));
 checkError(cudaMalloc( &dev_out, n*sizeof(double)));
 checkError(cudaMemcpy(dev_A, A,  m*n*sizeof(double), cudaMemcpyHostToDevice));
 checkError(cudaMemcpy(dev_in, in,  m*sizeof(double), cudaMemcpyHostToDevice));


  // execute kernel
  gen_matvec <<< numBlocksm, threadsPerBlockm >>>((double*)dev_A, (double*)dev_in, (double*)dev_out, m, n);
  checkError(cudaThreadSynchronize());
 checkError(cudaMemcpy(out, dev_out,  n*sizeof(double), cudaMemcpyDeviceToHost));
  checkError(cudaEventRecord(stop,0));
  checkError(cudaEventSynchronize(stop));
  checkError(cudaEventElapsedTime(&time, start, stop));
  checkError(cudaEventDestroy(start));
  checkError(cudaEventDestroy(stop));

cout << "out="<< out[0] << " "<< out[1] << endl;
  return time;
}


 void MultArmVM(double * V, double * M, double * R, int m_nr, int m_nc)
 {
  double sum;
  for (int c=0; c < m_nc; c++)
  {
    sum=0;
    for (int r = 0; r < m_nr; r++)  // m_nr == v_nc
       sum += M[c*m_nr+r] * V[r];
    R[c] = sum;
  }
 }
int main()
{
   double vbuf[20]= { 1, 2, 3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0 };

   double Abuf[40]={1, 2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,0,1, 2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,0};
   double rbuf[2];
   rowvec v(vbuf, 20, false, true);
   rowvec r(rbuf, 2, false, true);
    v = { 1, 2, 3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0 };

mat A(Abuf, 2, 20,  false, true);
  A  = { {1, 2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,0},
         {1, 2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,0}};
/*             { {1,2},
         
               {3, 4},
               {3, 4},
               {3, 4},
               { 5, 6},
               {3, 4},
               { 5, 6},
               {3, 4},
               { 5, 6},
               {3, 4},
               { 5, 6},
               {3, 4},
               { 5, 6},
               {3, 4},
               { 5, 6},
               { 5, 6},
               { 5, 6},
               {3, 4},
               { 5, 6},
               {7,8} }; */

cout << "A*v=";
cout << v*A.t() << endl;
//A=A.t();
//MultArmVM(vbuf, Abuf, rbuf, 20, 2);

 matVecNaive (rbuf, vbuf, Abuf, 20, 2) ;
cout << "rbuf=" << rbuf[0] << " " << rbuf[1] << endl;
cout << "r=" << r << endl;
}
