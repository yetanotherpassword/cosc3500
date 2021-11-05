#include <stdio.h>
#include <math.h>
#include <iostream>
#define TILE_DIM 16                     // Tile dimension
#define DIMX 1                            
#define DIMY 785
#define DIMZ 31

__global__ void MatMulNoShared(float* A, float* B, float* C, int ARows, int ACols, int BRows, int BCols, int CRows, int CCols) {

    float CValue = 0;

    int Row = blockIdx.y*TILE_DIM + threadIdx.y;
    int Col = blockIdx.x*TILE_DIM + threadIdx.x;

    for (int k = 0; k < (TILE_DIM + ACols - 1)/TILE_DIM; k++) {

        for (int n = 0; n < TILE_DIM; ++n) 
            if ((k*TILE_DIM + n < ACols && Row < ARows) && (k*TILE_DIM + n < BRows && Col < BCols))
                CValue += A[Row*ACols + k*TILE_DIM + n] * B[(k*TILE_DIM + n)*BCols + Col];

    }

    if (Row < CRows && Col < CCols) C[((blockIdx.y * blockDim.y + threadIdx.y)*CCols)+(blockIdx.x*blockDim.x)+threadIdx.x]=CValue;
}

int main() {

    int CCols = DIMZ, CRows=DIMX, ACols=DIMY, ARows=DIMX, BCols=DIMZ, BRows=DIMY;

    dim3 dimBlock(TILE_DIM, TILE_DIM, 1);
    dim3 dimGrid;

    dimGrid.x = (CCols + dimBlock.x - 1)/dimBlock.x;
    dimGrid.y = (CRows + dimBlock.y - 1)/dimBlock.y;

    float *deviceA, *deviceB, *deviceC;

    float* hostA    = (float*)malloc(DIMX*DIMY*sizeof(float));
    float* hostB    = (float*)malloc(DIMY*DIMZ*sizeof(float));
    float* hostC    = (float*)malloc(DIMX*DIMZ*sizeof(float));
    float* hostCp   = (float*)malloc(DIMX*DIMZ*sizeof(float));

    for (int x = 0; x<DIMY; x++)
    {
            hostA[x] = x+1;
        for (int y = 0; y<DIMZ; y++) {
            hostB[x*DIMZ+y] = x+y+1;
        }
   }

    cudaMalloc((void **)&deviceA, DIMX*DIMY*sizeof(float));
    cudaMalloc((void **)&deviceB, DIMY*DIMZ*sizeof(float));
    cudaMalloc((void **)&deviceC, DIMX*DIMZ*sizeof(float));

    cudaMemcpy(deviceA, hostA, DIMX*DIMY*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, DIMY*DIMZ*sizeof(float), cudaMemcpyHostToDevice);

    MatMulNoShared<<<dimGrid , dimBlock>>>(deviceA , deviceB , deviceC , ARows , ACols, BRows ,BCols , CRows , CCols);
    cudaMemcpy(hostC, deviceC, DIMX*DIMZ*sizeof(float), cudaMemcpyDeviceToHost);

std::cout << "A=";
for (int i=0;i<ARows;i++)
{
   for (int j=0;j<ACols;j++)
   {
      std::cout << hostA[i*ACols+j] << " ";
   }
 std::cout << std::endl;
}

std::cout << "B=";
for (int i=0;i<BRows;i++)
{
   for (int j=0;j<BCols;j++)
   {
      std::cout << hostB[i*BCols+j] << " ";
   }
 std::cout << std::endl;
}
std::cout << "C=";
for (int i=0;i<CRows;i++)
{
   for (int j=0;j<CCols;j++)
   {
      std::cout << hostC[i*CCols+j] << " ";
   }
 std::cout << std::endl;
}
    return 0;
}
