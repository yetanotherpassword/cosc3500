#include <iostream>
#include <stdio.h>
#include <math.h>
//##include <conio.h>
#include <armadillo>
#define TILE_DIM 16                     // Tile dimension
using namespace std;
using namespace arma;
__global__ void MatMulNoShared(double* A, double* B, double* C, int ARows, int ACols, int BRows, int BCols, int CRows, int CCols) {

    double CValue = 0;

    int Row = blockIdx.y*TILE_DIM + threadIdx.y;
    int Col = blockIdx.x*TILE_DIM + threadIdx.x;

    for (int k = 0; k < (TILE_DIM + ACols - 1)/TILE_DIM; k++) {

        for (int n = 0; n < TILE_DIM; ++n) 
            if ((k*TILE_DIM + n < ACols && Row < ARows) && (k*TILE_DIM + n < BRows && Col < BCols))
                CValue += A[Row*ACols + k*TILE_DIM + n] * B[(k*TILE_DIM + n)*BCols + Col];

    }

    if (Row < CRows && Col < CCols) C[((blockIdx.y * blockDim.y + threadIdx.y)*CCols)+(blockIdx.x*blockDim.x)+threadIdx.x]=CValue;
}

void PreMatMul(rowvec & a, mat & b, rowvec & c)
{
    int DIMZ = c.n_cols;
    int DIMX = c.n_rows;
    int DIMY = a.n_cols;
    if ((DIMX != a.n_rows) || (DIMY != b.n_rows) || (DIMZ != b.n_cols))
    {
       cout << "Incorrect dimensions passed to PreMatMul" << endl;
       exit(1);
    }

    int CCols = DIMZ, CRows=DIMX, ACols=DIMY, ARows=DIMX, BCols=DIMZ, BRows=DIMY;

    dim3 dimBlock(TILE_DIM, TILE_DIM, 1);
    dim3 dimGrid;

    dimGrid.x = (CCols + dimBlock.x - 1)/dimBlock.x;
    dimGrid.y = (CRows + dimBlock.y - 1)/dimBlock.y;
cout << " dimGrid.x = ("<< CCols << " + " << dimBlock.x << " - 1)/" << dimBlock.x<<endl;
cout << " dimGrid.y = ("<< CRows << " + " << dimBlock.y << " - 1)/" << dimBlock.y<<endl;
    double *deviceA, *deviceB, *deviceC;
    //hostC = 
    double* hostC    = (double*)malloc(DIMX*DIMZ*sizeof(double));

    cudaMalloc((void **)&deviceA, DIMX*DIMY*sizeof(double));
    cudaMalloc((void **)&deviceB, DIMY*DIMZ*sizeof(double));
    cudaMalloc((void **)&deviceC, DIMX*DIMZ*sizeof(double));

    cudaMemcpy(deviceA, a.memptr(), DIMX*DIMY*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, b.memptr(), DIMY*DIMZ*sizeof(double), cudaMemcpyHostToDevice);

    MatMulNoShared<<<dimGrid , dimBlock>>>(deviceA , deviceB , deviceC , ARows , ACols, BRows ,BCols , CRows , CCols);
 cudaDeviceSynchronize();

    cudaMemcpy(hostC, deviceC, DIMX*DIMZ*sizeof(double), cudaMemcpyDeviceToHost);
    for (int j=0;j<31;j++)
      cout << hostC[j] << " " ;
    cout << endl;

    memcpy(c.memptr(), hostC, 32*sizeof(double));

}

int main()
{
double q[500];
    rowvec a( 785);
    mat w(785,31);
    rowvec n(q, 31, false, true);



    for (int i=0;i<784;i++)
       a(i)=33.0;
    for (int i=0;i<784;i++)
       for (int j=0;j<31;j++)
          w(i,j)=77.0;


    PreMatMul(a,w, n);
    for (int j=0;j<31;j++)
      cout << n << endl;
}
