#include <iomanip>
#ifdef USE_CUBLAS
#include <cstdlib>
#include <ctime>
#include <cublas_v2.h>
#include <curand.h>
#endif
#include <cmath>
#include <chrono>
#ifdef WANT_TO_LOAD_WEIGHTS
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/split.hpp>
#endif
#include <vector>
#include <limits>
#include <sstream>
#include <fstream>
#include <iostream>
#include <string>
#include <map>
#include <iterator>
// Application Parameters
#define DEFTHREADS 256
#define INPUT_LINES 784
#define OUTPUT_LINES 10
#define MATRIX_SIDE 28
#define MAX_PIXEL_VAL 255.0f
#define IMAGE_OFFSET 16
#define DEFAULT_HIDDEN 300
#define DEFAULT_HIDDEN1 500
#define DEFAULT_HIDDEN2 300
#define ETA_DEFAULT 0.5f
#define EPSILON 1E-04
#define TRAININGSAMPLES 6
#define TESTINGSAMPLES 1
#define EPOCHS 1

// How often to print samples, 1=All, 2=every second one, etc
// Undefine or define to very large number to remove output
#define SAMPLEFREQ 1
//#undef SAMPLEFREQ
#ifdef USE_CUBLAS
cublasHandle_t handle;
void gpu_blas_mmul(const double *A, const double *B, double *C, const int m, const int k, const int n) {
	int lda=m,ldb=k,ldc=m;
	const double alf = 1;
	const double bet = 0;
	const double *alpha = &alf;
	const double *beta = &bet;


	// Do the actual multiplication
	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

}
#endif




void checkError(cudaError_t e)
{
    if (e != cudaSuccess)
    {
        std::cerr << "CUDA error: " << int(e) << " : " << cudaGetErrorString(e) <<
            '\n';
        abort();
    }
}


/*
 * ALLAN CAMPTON
 * COSC3500 Milestone 2 Parallel Version
 *
 * To perform a full build and run from scratch, do the following
 *
     unzip Project_AC.zip
     cd ~/cosc3500/
     unzip mnist.zip
     make
     sbatch ./goslurm.sh ann_mnist_digits_cuda    #Run parallel version (with default settings)
     sbatch ./goslurm.sh ann_mnist_digits_serial  #Run serial version for comparison

 */


int thrds = DEFTHREADS;

using namespace std;

float mintime = std::numeric_limits<float>::max();
float maxtime = std::numeric_limits<float>::min();

std::chrono::nanoseconds Process_MaxTime = std::chrono::nanoseconds::min();
std::chrono::nanoseconds Process_MinTime = std::chrono::nanoseconds::max();
class time_measurement 
{  
   public:
     std::chrono::nanoseconds Call_MaxTime;
     std::chrono::nanoseconds Call_MinTime;
     std::chrono::nanoseconds TotalCallTime;
     std::chrono::_V2::system_clock::time_point StartCallTime;
     std::chrono::_V2::system_clock::time_point EndCallTime;
     std::chrono::nanoseconds Tot_Time;
     int Tot_Cnt;
     int depth;
     bool in_measurement;
     string name;
     time_measurement(string n="")
     {
       depth=0;
       in_measurement=false;
       Call_MaxTime = std::chrono::nanoseconds::min();
       Call_MinTime = std::chrono::nanoseconds::max();
       Tot_Cnt =0;
       Tot_Time = std::chrono::nanoseconds::zero();
       name=n;
     };
     void start_measurement()
     {
        if (in_measurement)
        {
          depth++;
        }
        else
        {
          if (depth==0)
          {
            depth=1;
            in_measurement = true;
            StartCallTime = std::chrono::high_resolution_clock::now();
          }
          else
          {
             cout << "Error: in call sequence of Start Measurement" << endl;
          }
        }
     };
     void stop_measurement()
     {
          if (depth <= 0)
          {
             cout << "Error: in call sequence of Stop Measurement" << endl;
          }
          else
          {
             depth--;
             if (depth == 0)
             {
                EndCallTime = std::chrono::high_resolution_clock::now();
                TotalCallTime = std::chrono::duration_cast<std::chrono::nanoseconds> (EndCallTime - StartCallTime);

                if (TotalCallTime > Call_MaxTime)
                    Call_MaxTime = TotalCallTime;

                if (TotalCallTime < Call_MinTime)
                    Call_MinTime = TotalCallTime;
         
                Tot_Time += TotalCallTime;
                Tot_Cnt++;
             }
          }
     };
     int64_t min_time()
     {
         return Call_MinTime.count();
     };
     int64_t max_time()
     {
         return Call_MaxTime.count();
     };

     int number_of_calls()
     {
         return Tot_Cnt;
     };

     int64_t accumulated_time()
     {
         return Tot_Time.count();
     };

     string avg_time()
     {
         string s;
         if (Tot_Cnt == 0)
         {
             s="Never Called";
             return s;
         }
         else
         {
             int64_t t=Tot_Time.count() / (int64_t) Tot_Cnt;
             s=to_string(t);
         } 
         return s;
     };

     int64_t last_time()
     {
         return TotalCallTime.count();
     };
     string output_all_times(const string b)
     {
         stringstream s;
         string s2=avg_time();
         if (s2=="Never Called")
         {
            s << "For " << name << ":" << endl;
            s << "   No measurements as wasnt called" << endl << flush;
         }
         else
         {
            string bt2;
            if (b=="Parallel")
                bt2="|Parallel|";
            else
                bt2="..Serial.."; 
            string blank="                              ";
            string filler=blank.substr(name.length());
            s << "For " << name << bt2<<  filler << endl;
            s << "   Min Time    : " << min_time() << " ns" << endl;
            s << "   Max Time    : " << max_time() << " ns" << endl;
            s << "   All Time    : " << accumulated_time() << " ns" << endl;
            s << "   No of Calls : " << number_of_calls() << "        " << endl;
            s << "   Avg Time    : " << s2 << " ns" << endl;
            s << "   Last Time   : " << last_time() << " ns" << endl << flush;
         }
         return s.str();
     };

};
char ss[60000000];

#ifndef SERIAL_ONLY
double* LayerWeightsDevice;
double* ActuationDevice;
double* NetinDevice;
double* deviceA, * deviceB, * deviceC;
cudaEvent_t start, stop;
int tile_dimension = 8;
class newmat;
void PreMatMul(newmat& a, newmat& b, newmat& c, int norm);

#endif

typedef std::map< std::string, time_measurement* > maptype;
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


#define TILE_DIM 16                     // Tile dimension

#ifndef SERIAL_ONLY 
__global__ void MatMultMat(double* A, double* B, double* C, int ARows, int ACols, int BRows, int BCols, int CRows, int CCols) {

    double CValue = 0;

    int Row = blockIdx.y * TILE_DIM + threadIdx.y;
    int Col = blockIdx.x * TILE_DIM + threadIdx.x;

    for (int k = 0; k < (TILE_DIM + ACols - 1) / TILE_DIM; k++) {

        for (int n = 0; n < TILE_DIM; ++n)
            if ((k * TILE_DIM + n < ACols && Row < ARows) && (k * TILE_DIM + n < BRows && Col < BCols))
                CValue += A[Row * ACols + k * TILE_DIM + n] * B[(k * TILE_DIM + n) * BCols + Col];

    }

    if (Row < CRows && Col < CCols) C[((blockIdx.y * blockDim.y + threadIdx.y) * CCols) + (blockIdx.x * blockDim.x) + threadIdx.x] = CValue;
}


__global__ void TransposeMat(double *idata, double *odata, int height, int width)
{
	__shared__ double block[TILE_DIM][TILE_DIM+1];
	
	// read the matrix tile into shared memory
	unsigned int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
	unsigned int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
	if((xIndex < width) && (yIndex < height))
	{
		unsigned int index_in = yIndex * width + xIndex;
		block[threadIdx.y][threadIdx.x] = idata[index_in];
	}

	__syncthreads();

	// write the transposed matrix tile to global memory
	xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
	yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
	if((xIndex < height) && (yIndex < width))
	{
		unsigned int index_out = yIndex * height + xIndex;
		odata[index_out] = block[threadIdx.x][threadIdx.y];
	}
}

__global__ void MatMultMatEleWise(double* A, double* B, double* C)
{
    int i = threadIdx.x;
    C[i] = A[i] * B[i];
}


__global__ void MatSubMat(double* A, double* B, double* C, int n_rows, int n_cols)
{
    int Row = blockIdx.y * TILE_DIM + threadIdx.y;
    int Col = blockIdx.x * TILE_DIM + threadIdx.x;
    if((Col < n_cols) && (Row < n_rows))
    {
          int i=Row*n_cols+Col;
          C[i] = A[i] - B[i];
    }
}

__global__ void MatAddMat(double* A, double* B, double* C)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

__global__ void MatAddScalar(double scalar, double* C)
{
    int i = threadIdx.x;
    C[i] = C[i] + scalar;
}

__global__ void MatDivScalar(double scalar, double* C)
{
    int i = threadIdx.x;
    C[i] = C[i] / scalar;
}


__global__ void MatMultScalar(double scalar, double* C)
{
    int i = threadIdx.x;
    C[i] = C[i] * scalar;
}


#else
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// implementation of the matrix-vector multiply function
void SerialMatrixVectorMultiply(double* Y, double* X, int r1, double* M, int m_nr, int m_nc, int norm)
{
    // Need to ensure Y vector passed has been zeroised


    for (int i = 0; i < r1; ++i)
    {
        for (int j = 0; j < m_nc; ++j)
        {
            Y[i * m_nc + j] = 0;
            for (int k = 0; k < m_nr; ++k) // c1==r2
            {
                Y[i * m_nc + j] += X[i * m_nr + k] * M[k * m_nc + j];
                //                cout << "Y["<<i*m_nc+j<<"] += "<<X[i*m_nr+k] * M[k*m_nc+j] << endl;
            }
            Y[i * m_nc + j] =  Y[i * m_nc + j] / (double) norm;
        }
    }

}
#endif

class newmat {
public:
    double* ptr;
    int n_rows;
    int n_cols;
    static time_measurement* timeptr;
    static map< string, time_measurement* > *timers;
      
#ifdef SERIAL_ONLY
    const string build_type = "Serial";
#else
    const string build_type = "Parallel";
#endif
         void create_new_time_meas(string s)
         {
               timeptr = new time_measurement(s);
               (*timers)[s] = timeptr;
         }
    void init_timers()
    {
       if (timers==NULL)
          timers= new  map< string, time_measurement* >;
       if (timeptr==NULL)
       {
               create_new_time_meas("set_transpose");
               create_new_time_meas("add_mat");
               create_new_time_meas("add_scalar");
               create_new_time_meas("div_scalar");
               create_new_time_meas("mult_scalar");
               create_new_time_meas("set_mult1_add2_scalars");
               create_new_time_meas("set_mult1_add2_mat");
               create_new_time_meas("set_matmult");
               create_new_time_meas("piecewisemult");
               create_new_time_meas("set_diff2_piecewisemult3");
               create_new_time_meas("free_ele");
               create_new_time_meas("zeroize");
       }
    }
    newmat(int r, int c, double* p)
    {
        init_timers();
        n_rows = r;
        n_cols = c;
        ptr = new double[n_rows * n_cols];
#ifdef SERIAL_ONLY
        for (int i = 0; i < n_rows; i++)
            for (int j = 0; j < n_cols; j++)
                ptr[i * n_cols + j] = p[i * n_cols + j];
#else
        memcpy(ptr, p, n_rows * n_cols * sizeof(double));
#endif
    };
    newmat()
    {
        init_timers();
        n_rows = 0;
        n_cols = 0;
        ptr = NULL;
    };
    newmat(int r, int c)
    {
        init_timers();
        n_rows = r;
        n_cols = c;
        ptr = new double[r * c];
    };
    void set_transpose(newmat tmp)
    {
        (*timers)["set_transpose"]->start_measurement();
        if (n_rows == tmp.n_cols && n_cols == tmp.n_rows)
        {
#ifdef SERIAL_ONLY
            for (int i = 0; i < tmp.n_rows; i++)
                for (int j = 0; j < tmp.n_cols; j++)
                    ptr[j * tmp.n_rows + i] = tmp.ptr[i * tmp.n_cols + j];
#else
            int onedLen = n_rows * n_cols;
            cudaMemcpy(deviceA, tmp.ptr, onedLen * sizeof(double), cudaMemcpyHostToDevice);
            dim3 grid(tmp.n_cols*tmp.n_rows/ TILE_DIM, tmp.n_cols*tmp.n_rows/ TILE_DIM, 1);
            dim3 threads(TILE_DIM, TILE_DIM, 1);

            TransposeMat <<< grid, threads >>> (deviceA, deviceC, tmp.n_rows, tmp.n_cols);


            checkError(cudaDeviceSynchronize());

            cudaMemcpy(ptr, deviceC, onedLen * sizeof(double), cudaMemcpyDeviceToHost);
#endif
        }
        else
        {
            cout << "Cant transpose Matrix[" << tmp.n_rows << "," << tmp.n_cols << "] into given Matrix[" << n_rows << "," << n_cols << "]" << endl << flush;
            exit(1);
        }
        (*timers)["set_transpose"]->stop_measurement();
    };

    void add_mat(newmat m1)
    {
        (*timers)["add_mat"]->start_measurement();
        if (m1.n_rows != n_rows || m1.n_cols != n_cols)
        {
            cout << "Cant addmat m1[" << m1.n_rows << "," << m1.n_cols << "] with *this[" << n_rows << "," << n_cols << "]" << endl << flush;
            exit(1);
        }
#ifdef SERIAL_ONLY
        for (int i = 0; i < n_rows; i++)
            for (int j = 0; j < n_cols; j++)
                ptr[i * n_cols + j] += m1.ptr[i * n_cols + j];
#else
        int onedLen = n_rows * n_cols;
        cudaMemcpy(deviceA, ptr, onedLen * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(deviceB, m1.ptr, onedLen * sizeof(double), cudaMemcpyHostToDevice);

        MatAddMat <<< onedLen, 1 >>> (deviceA, deviceB, deviceC);

        checkError(cudaDeviceSynchronize());

        cudaMemcpy(ptr, deviceC, onedLen * sizeof(double), cudaMemcpyDeviceToHost);
#endif
        (*timers)["add_mat"]->stop_measurement();
    };

    void add_scalar(double d)
    {
        (*timers)["add_scalar"]->start_measurement();

#ifdef SERIAL_ONLY
 for (int i = 0; i < n_rows; i++)
     for (int j = 0; j < n_cols; j++)
         ptr[i * n_cols + j] += d;
#else
        int onedLen = n_rows * n_cols;
        cudaMemcpy(deviceC, ptr, onedLen * sizeof(double), cudaMemcpyHostToDevice);

        MatAddScalar <<< 1, onedLen >>> (d, deviceC);

        checkError(cudaDeviceSynchronize());

        cudaMemcpy(ptr, deviceC, onedLen * sizeof(double), cudaMemcpyDeviceToHost);
#endif
        (*timers)["add_scalar"]->stop_measurement();
    };

    void div_scalar(double d)
    {
        (*timers)["div_scalar"]->start_measurement();

#ifdef SERIAL_ONLY
        for (int i = 0; i < n_rows; i++)
          for (int j = 0; j < n_cols; j++)
             ptr[i * n_cols + j] /= d; 
#else
        int onedLen = n_rows * n_cols;
        cudaMemcpy(deviceC, ptr, onedLen * sizeof(double), cudaMemcpyHostToDevice);
        
        MatDivScalar <<< 1, onedLen >>> (d, deviceC);

        checkError(cudaDeviceSynchronize());

        cudaMemcpy(ptr, deviceC, onedLen * sizeof(double), cudaMemcpyDeviceToHost);
#endif

        (*timers)["div_scalar"]->stop_measurement();
    };

    void mult_scalar(double d)
    {
        (*timers)["mult_scalar"]->start_measurement();

#ifdef SERIAL_ONLY
       for (int i = 0; i < n_rows; i++)
          for (int j = 0; j < n_cols; j++)
             ptr[i * n_cols + j] *= d; 
#else
        int onedLen = n_rows * n_cols;
        cudaMemcpy(deviceC, ptr, onedLen * sizeof(double), cudaMemcpyHostToDevice);
        
        MatMultScalar <<< 1, onedLen >>> (d, deviceC);

        checkError(cudaDeviceSynchronize());

        cudaMemcpy(ptr, deviceC, onedLen * sizeof(double), cudaMemcpyDeviceToHost);
#endif
        (*timers)["mult_scalar"]->stop_measurement();

    };

    void set_mult1_add2_scalars(newmat y1, double d1, double d2)
    {
        (*timers)["set_mult1_add2_scalars"]->start_measurement();

        if (n_rows != y1.n_rows || n_cols != y1.n_cols)
        {
            cout << "Cant store y1[" << y1.n_rows << "," << y1.n_cols << " in *this[" << n_rows << "," << n_cols << "]" << endl;
            exit(1);
        }

#ifdef SERIAL_ONLY        
        for (int i = 0; i < n_rows; i++)
            for (int j = 0; j < n_cols; j++)
                ptr[i * n_cols + j] = y1.ptr[i * n_cols + j];
#else
        memcpy(ptr, y1.ptr, n_cols * n_rows * sizeof(double));
#endif

        mult_scalar(d1);
        add_scalar(d2);

        (*timers)["set_mult1_add2_scalars"]->stop_measurement();
    };
    void set_mult1_add2_mat(newmat y1, double d1, newmat y2)
    {
        (*timers)["set_mult1_add2_mat"]->start_measurement();
        if (n_rows != y1.n_rows || n_rows != y2.n_rows || n_cols != y1.n_cols || n_cols != y2.n_cols)
        {
            cout << "Cant add y1[" << y1.n_rows << "," << y1.n_cols << "] to y2[" << y2.n_rows << "," << y2.n_cols << "] and store in *this[" << n_rows << "," << n_cols << "]" << endl;
            exit(1);
        }
#ifdef SERIAL_ONLY
        for (int i = 0; i < n_rows; i++)
            for (int j = 0; j < n_cols; j++)
                ptr[i * n_cols + j] = y1.ptr[i * n_cols + j];
#else
         memcpy(ptr, y1.ptr, y1.n_cols * y1.n_rows * sizeof(double));
#endif
        mult_scalar(d1);
        add_mat(y2);

        (*timers)["set_mult1_add2_mat"]->stop_measurement();
    };
    void set_matmult(newmat p1, newmat p2, int norm=1)
    {
        (*timers)["set_matmult"]->start_measurement();
        if (p1.n_cols == p2.n_rows)
        {
            if (n_rows != p1.n_rows || n_cols != p2.n_cols)
            {
                cout << "Resultant matrix wont hold result, fixing by realloc in set_matmult" << endl;
                free_ele();
                n_rows = p1.n_rows;
                n_cols = p2.n_cols;
                ptr = new double[n_rows * n_cols];
            }
   
#ifdef SERIAL_ONLY
            SerialMatrixVectorMultiply(ptr, p1.ptr, p1.n_rows, p2.ptr, p2.n_rows, p2.n_cols, norm);
/*
            for (int i = 0; i < p1.n_rows; ++i)
                for (int j = 0; j < p2.n_cols; ++j)
                {
                    ptr[i * p2.n_cols + j] = 0;
                    for (int k = 0; k < p1.n_cols; ++k) // c1==r2
                    {
                        ptr[i * p2.n_cols + j] += p1.ptr[i * p1.n_cols + k] * p2.ptr[k * p2.n_cols + j];
                    }
                    ptr[i * p2.n_cols + j] = ptr[i * p2.n_cols + j] / (double) norm;
                }
 */              
#else
             PreMatMul(p1, p2, *this, norm);
#endif
        }
        else
        {
            cout << "Cant multiply p1[" << p1.n_rows << "," << p1.n_cols << "] by p2[" << p2.n_rows << "," << p2.n_cols << "]" << endl;
            exit(1);
        }

        (*timers)["set_matmult"]->stop_measurement();
    };

    void piecewisemult(newmat p1)
    {
        (*timers)["piecewisemult"]->start_measurement();

        if (p1.n_rows != n_rows || p1.n_cols != n_cols)
        {
            cout << "Cant piecewisemultiply p1[" << p1.n_rows << "," << p1.n_cols << "] with *this[" << n_rows << "," << n_cols << "]" << endl;
            exit(1);
        }
#ifdef SERIAL_ONLY
        for (int i = 0; i < p1.n_rows; i++)
        {
            for (int j = 0; j < n_cols; j++)
                ptr[i * n_cols + j] = p1.ptr[i * n_cols + j] * ptr[i * n_cols + j];
        }
#else
        int onedLen = n_rows * n_cols;
        cudaMemcpy(deviceA, ptr, onedLen * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(deviceB, p1.ptr, onedLen * sizeof(double), cudaMemcpyHostToDevice);

        MatMultMatEleWise <<< 1, onedLen >>> (deviceA, deviceB, deviceC);

        checkError(cudaDeviceSynchronize());

        cudaMemcpy(ptr, deviceC, onedLen * sizeof(double), cudaMemcpyDeviceToHost);
        
#endif

        (*timers)["piecewisemult"]->stop_measurement();
    };

    void set_diff2_piecewisemult3(newmat p1, newmat p2, newmat p3)
    {
        (*timers)["set_diff2_piecewisemult3"]->start_measurement();

        if (n_rows != p1.n_rows || n_rows != p2.n_rows || n_cols != p1.n_cols || n_cols != p2.n_cols)
        {
            cout << "Cant diff p2[" << p2.n_rows << "," << p2.n_cols << "] from p1[" << p1.n_rows << "," << p1.n_cols << "] and store in *this[" << n_rows << "," << n_cols << "]" << endl;
            exit(1);
        }
#ifdef SERIAL_ONLY
        for (int i = 0; i < n_rows; i++)
        {
            for (int j = 0; j < n_cols; j++)
                ptr[i * p1.n_cols + j] = p1.ptr[i * p1.n_cols + j] - p2.ptr[i * p1.n_cols + j];
        }
#else
        int onedLen = n_rows * n_cols;
        cudaMemcpy(deviceA, p1.ptr, onedLen * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(deviceB, p2.ptr, onedLen * sizeof(double), cudaMemcpyHostToDevice);
    dim3 dimBlock(TILE_DIM, TILE_DIM, 1);
    dim3 dimGrid;

    dimGrid.x = (n_cols + dimBlock.x - 1) / dimBlock.x;
    dimGrid.y = (n_rows + dimBlock.y - 1) / dimBlock.y;

        MatSubMat <<< dimGrid, dimBlock >>> (deviceA, deviceB, deviceC, n_rows, n_cols);

        checkError(cudaDeviceSynchronize());

        cudaMemcpy(ptr, deviceC, onedLen * sizeof(double), cudaMemcpyDeviceToHost);

#endif
        piecewisemult(p3);

        (*timers)["set_diff2_piecewisemult3"]->stop_measurement();
    };

    char* prtstr()
    {
        // string s="";
       //  char ss[100000];
        ss[0] = '\0';
        for (int i = 0; i < n_rows; i++)
        {
            for (int j = 0; j < n_cols; j++)
            {
                int len = strlen(ss);
                sprintf(&ss[len], "   %20.10g", ptr[i * n_cols + j]);
                //	   s = s + "   " + to_string(ptr[i*n_cols+j]);
            }
            //  s+= '\n';
            sprintf(&ss[strlen(ss)], "\n");
        }
        return ss;
    };
    void free_ele()
    {
        (*timers)["free_ele"]->start_measurement();
        if (ptr != NULL)
            delete[] ptr;

        (*timers)["free_ele"]->stop_measurement();
    };
    void zeroize()
    {
        (*timers)["zeroize"]->start_measurement();
#ifdef SERIAL_ONLY
        for (int i = 0; i < n_rows; i++)
        {
            for (int j = 0; j < n_cols; j++)
                ptr[i * n_cols + j] = 0.0;
        }
#else
        memset(ptr, 0, n_cols * n_rows * sizeof(double));
#endif
        (*timers)["zeroize"]->stop_measurement();
    };

    double* memptr()
    {
        return ptr;
    };

    int index_max_row(int r, int start, int stop)
    {
        int idx = 0;
        double max = std::numeric_limits<double>::min();
        if (((r < n_rows) && (r >= 0)) && (start >= 0) && (start < n_cols) && (stop >= 0) && (stop < n_cols) && (start <= stop))
            for (int i = r; i <= r; i++)
                for (int j = start; j <= stop; j++)
                    if (ptr[i * n_cols + j] > max)
                    {
                        idx = i * n_cols + j;
                        max = ptr[i * n_cols + j];
                    }
        return idx;
    };
    void output_all_procs(ostream & s)
    {
        int toplen=60;
        for (maptype::iterator t_it = timers->begin(); t_it != timers->end(); t_it++)
        {
            string outp = (*timers)[t_it->first]->output_all_times(build_type);
            int fill=toplen-outp.length();
            s << outp;
            for (int i=0;i<fill;i++)
               s << " ";
            s << endl;
            
        }
    };
};
maptype* newmat::timers;

time_measurement * newmat::timeptr=NULL;

#ifndef SERIAL_ONLY
void PreMatMul(newmat& a, newmat& b, newmat& c, int norm)
{
    int DIMZ = c.n_cols;
    int DIMX = c.n_rows;
    int DIMY = a.n_cols;
    if ((DIMX != a.n_rows) || (DIMY != b.n_rows) || (DIMZ != b.n_cols))
    {
        cout << "Incorrect dimensions passed to PreMatMul" << endl;
        cout << "c(" << c.n_rows << "," << c.n_cols << ") is to be set to "
            << "a(" << a.n_rows << "," << a.n_cols << ") * "
            << "b(" << b.n_rows << "," << b.n_cols << ")  "
            << endl;
        exit(1);
    }

    int CCols = DIMZ, CRows = DIMX, ACols = DIMY, ARows = DIMX, BCols = DIMZ, BRows = DIMY;

    dim3 dimBlock(TILE_DIM, TILE_DIM, 1);
    dim3 dimGrid;

    dimGrid.x = (CCols + dimBlock.x - 1) / dimBlock.x;
    dimGrid.y = (CRows + dimBlock.y - 1) / dimBlock.y;
    //cout << " dimGrid.x = ("<< CCols << " + " << dimBlock.x << " - 1)/" << dimBlock.x<<endl;
    //cout << " dimGrid.y = ("<< CRows << " + " << dimBlock.y << " - 1)/" << dimBlock.y<<endl;
        //hostC = 
    double* hostC = (double*)malloc(DIMX * DIMZ * sizeof(double));

    cudaMemcpy(deviceA, a.memptr(), DIMX * DIMY * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, b.memptr(), DIMY * DIMZ * sizeof(double), cudaMemcpyHostToDevice);


#ifndef USE_CUBLAS
    MatMultMat << <dimGrid, dimBlock >> > (deviceA, deviceB, deviceC, ARows, ACols, BRows, BCols, CRows, CCols);
#else
    gpu_blas_mmul(deviceA, deviceB, deviceC, ARows, ACols, BRows);
#endif
    checkError(cudaDeviceSynchronize());




    if (norm != 1)
    {
        MatDivScalar <<< 1, DIMX * DIMZ>>> ((double)norm, deviceC);

        checkError(cudaDeviceSynchronize());

    }

    cudaMemcpy(c.memptr(), deviceC, DIMX * DIMZ * sizeof(double), cudaMemcpyDeviceToHost);

}
#endif
std::time_t result = std::time(nullptr);
string fid = to_string(result);
unsigned int NumberOfLayers;
unsigned int OutputLayer;
unsigned int* nodes;
double eta;	// Learning factor

vector<newmat> netin;
vector<newmat> actuation;
vector<newmat> deltafn;
vector<newmat> deltafn_t;
vector<newmat> ftick;
vector<newmat> layer_weights;
vector<newmat> layer_weights_t;
vector<newmat> weight_updates;
vector<newmat> new_layer_weights;
newmat tgt(1, OUTPUT_LINES + 1);


ios init(NULL);
stringstream confusion_matrix;
stringstream time_output;
newmat err_summary(1, OUTPUT_LINES);


#ifdef WANT_TO_LOAD_WEIGHTS
// Used for loading weights from file (if ever required)
double l2[10][50000];
int nd[100];
int nd2[100];
int lays;
int t = 0;
int x = 0;
#endif


void sigmoid3(newmat& net, newmat& out)
{
    int c = net.n_cols - 1;
    for (int i = 0; i <= c; i++)
        out.ptr[i] = 1 / (1 + exp(-net.ptr[i]));
    out.ptr[c] = 1.0;	// add bias signal value
      //return out;
}

/////////////////////////////////////////////
//
// PRINT ROUTINES
//
void print_an_image_vals(unsigned char* c, int i)
{
    cout << "This is a : " << i << endl << flush;
    for (int i = 0; i < INPUT_LINES; i++)
    {
        if (i % MATRIX_SIDE == 0)
            cout << endl << flush;
        cout << hex << std::setfill('0') << std::setw(2) << (unsigned int)c[i] <<
            dec << " ";
    }
    cout << endl << flush;
}

void print_an_image(unsigned char* c, int i)
{
    cout << "This is a : " << i << endl << flush;
    for (int i = 0; i < INPUT_LINES; i++)
    {
        if (i % MATRIX_SIDE == 0)
            cout << endl << flush;
        if (c[i] == 0)
            cout << "  ";
        else if (c[i] < 128)
            cout << "xx";
        else
            cout << "XX";
    }
    cout << endl << flush;
}

void print_images(unsigned char* c, int size)
{
    for (int i = IMAGE_OFFSET; i < size; i++)
    {
        if (((i - IMAGE_OFFSET) % MATRIX_SIDE) == 0)
            cout << endl << flush;
        if (((i - IMAGE_OFFSET) % INPUT_LINES) == 0)
            cout << endl << "Image : " << dec <<
            ((i - IMAGE_OFFSET) / INPUT_LINES) + 1 << endl << flush;
        cout << hex << std::setfill('0') << std::setw(2) << (unsigned int)c[i] <<
            " ";
    }
}

//
//
/////////////////////////////////////////////

unsigned char* load_file(string filename, string labels, unsigned char** labs)
{
    unsigned char* memblock;
    ifstream inFile;
    streampos size;

    cout << "Using file '" << filename << "'" << endl << flush;
    //
    // Load MNIST DIGIT IMAGES
    //
    inFile.open(filename, ios::in | ios::binary | ios::ate);
    if (!inFile)
    {
        cout << "Unable to open file '" << filename << "'" << endl << flush;
        exit(1);	// terminate with error
    }
    else
    {
        cout << "OK opened '" << filename << "' Successfully" << endl << flush;
    }

    if (inFile.is_open())
    {
        size = inFile.tellg();
        memblock = new unsigned char[size];
        inFile.seekg(0, ios::beg);
        inFile.read((char*)memblock, size);
        inFile.close();

        cout << "the entire file content is in memory, all " << size <<
            " bytes of it" << endl << flush;
    }
    inFile.close();
    //
    // Load MNIST DIGIT LABELS
    //
    inFile.open(labels, ios::in | ios::binary | ios::ate);
    if (!inFile)
    {
        cout << "Unable to open file '" << labels << "'" << endl << flush;
        exit(1);	// terminate with error
    }
    else
    {
        cout << "OK opened '" << labels << "' Successfully" << endl << flush;
    }

    if (inFile.is_open())
    {
        size = inFile.tellg();
        *labs = new unsigned char[size];
        inFile.seekg(0, ios::beg);
        inFile.read((char*)*labs, size);
        inFile.close();

        cout << "the entire file content is in memory, all " << size <<
            " bytes of it" << endl << flush;
    }
    inFile.close();
    return memblock;
}

void load_an_image(int seq, unsigned char*& mptr, newmat& img, newmat& t,
    unsigned char*& lp, int e)
{
    int start = (INPUT_LINES * seq) + IMAGE_OFFSET;
    double greyval = MAX_PIXEL_VAL;

    for (int i = 0; i < INPUT_LINES; i++)
    {
        img.ptr[i] = ((double)mptr[start + i]) / greyval;
    }

    img.ptr[nodes[0]] = 1;      // set bias signal, so can multiply with[node weights |
       // bias weights] augmented matrix

    int img_is_digit = (int)lp[8 + seq];
#ifdef SAMPLEFREQ
    if ((seq + 1) % SAMPLEFREQ == 0)
    {
        cout << "For sample :" << seq + 1 << " About to do " << e << " Epochs on" << endl << flush;
        print_an_image(&mptr[start], img_is_digit);
    }
#endif
    t.zeroize();  // create the target vector (plus one for 'bias' bit)
    if (img_is_digit > 9)
    {
        cout << "Error: img_is_digit=" << img_is_digit << "seq=" << seq << endl;
        exit(1);
    }
    t.ptr[img_is_digit] = 1;    // set the target 'bit'
}


////////////////////////
//
// DEBUG ROUTINES
// For use with gdb
/*
void output(mat t)
{
     cout << t << endl;
}

// For use with gdb
void output(rowvec t)
{
     cout << t << endl;
}
*/
void output(newmat t, string g="")
{
    cout << g << endl;
    cout << t.prtstr();
}

double accu0(newmat m1)
{
    double tmp = 0;
    for (int i = 0; i < m1.n_rows; i++)
        for (int j = 0; j < m1.n_cols; j++)
            tmp += m1.ptr[i * m1.n_cols + j];
    return tmp;
}

#if 0
newmat diff0(newmat p1, newmat p2)
{
    newmat tmp(p1.n_rows, p1.n_cols);
    for (int i = 0; i < p1.n_rows; i++)
    {
        for (int j = 0; j < p1.n_cols; j++)
            tmp.ptr[i * p1.n_cols + j] = p1.ptr[i * p1.n_cols + j] - p2.ptr[i * p1.n_cols + j];
    }
    return tmp;
}
newmat matmult0(newmat p1, newmat p2)
{
    if (p1.n_cols == p2.n_rows)
    {
        newmat tmp(p1.n_rows, p2.n_cols);

        /*
            for(i = 0; i < r1; ++i)
                for(j = 0; j < c2; ++j)
                    for(k = 0; k < c1; ++k) // c1==r2
                    {
                        mult2[i*c2+j] += ap[i*c1+k] * bp[k*c2+j];
                    }
        */



        if (p1.n_cols == p2.n_rows)
        {
            for (int i = 0; i < p1.n_rows; i++)
                for (int j = 0; j < p2.n_cols; j++)
                    for (int k = 0; k < p1.n_cols; k++)
                        tmp.ptr[i * p2.n_cols + j] = p1.ptr[i * p1.n_cols + k] * p2.ptr[k * p2.n_cols + j];
        }
        return tmp;
    }
    else
    {
        cout << "Cant multiply p1[" << p1.n_rows << "," << p1.n_cols << "] by p2[" << p2.n_rows << "," << p2.n_cols << "]" << endl;
        exit(1);
    }
    newmat dumy;
    return dumy;
}
newmat mult(newmat p1, double p2)
{
    newmat tmp(p1.n_rows, p1.n_cols);
    for (int i = 0; i < p1.n_rows; i++)
    {
        for (int j = 0; j < p1.n_cols; j++)
            tmp.ptr[i * p1.n_cols + j] = p1.ptr[i * p1.n_cols + j] * p2;
    }
    return tmp;
}
newmat add(newmat p1, double p2)
{
    newmat tmp(p1.n_rows, p1.n_cols);
    for (int i = 0; i < p1.n_rows; i++)
    {
        for (int j = 0; j < p1.n_cols; j++)
            tmp.ptr[i * p1.n_cols + j] = p1.ptr[i * p1.n_cols + j] + p2;
    }
    return tmp;
}
newmat matadd(newmat p1, newmat p2)
{
    newmat tmp(p1.n_rows, p1.n_cols);
    for (int i = 0; i < p1.n_rows; i++)
    {
        for (int j = 0; j < p1.n_cols; j++)
            tmp.ptr[i * p1.n_cols + j] = p1.ptr[i * p1.n_cols + j] + p2.ptr[i * p1.n_cols + j];
    }
    return tmp;
}
#endif
int backprop(int y0)
{

    //tgt0.insert_cols(nodes[OutputLayer], 1);
    //  double err = accu((tgt - final) % (tgt - final)) *0.5;
    double err = 0;
    for (int i = 0; i < tgt.n_rows; i++)
        for (int j = 0; j < tgt.n_cols - 1; j++) // last ele in tgt is bias so dont include in err function
        {
            err += (tgt.ptr[i * tgt.n_cols + j] - actuation[OutputLayer].ptr[i * tgt.n_cols + j]) * (tgt.ptr[i * tgt.n_cols + j] - actuation[OutputLayer].ptr[i * tgt.n_cols + j]);
        }
    err *= 0.5;

    if (abs(err) < EPSILON)
    {
        int val = tgt.index_max_row(0, 0, 9);
#ifdef SAMPLEFREQ
        if ((y0 + 1) % SAMPLEFREQ == 0)
            cout << "---------------------------------- BACK PROPAGATION  sample=" <<
            y0 + 1 << " err=" << err << "<epsilon, for tgt '" << val <<
            "' so error is acceptable, returning" << endl << flush;
#endif
        err_summary.ptr[val] = err;
        return 1;
    }

#ifdef SAMPLEFREQ
    if ((y0 + 1) % SAMPLEFREQ == 0)
        cout << "------------------------------------ BACK PROPAGATION sample=" <<
        y0 + 1 << endl << flush;
#endif
    ftick[OutputLayer].set_mult1_add2_scalars(actuation[OutputLayer], -1.0, 1.0);                            //  ftick[OutputLayer] = -actuation[OutputLayer] + 1;
//output( ftick[OutputLayer], " ftick[OutputLayer] 1");
    ftick[OutputLayer].piecewisemult(actuation[OutputLayer]);	// element wise multiply                //  ftick[OutputLayer] = ftick[OutputLayer] % (actuation[OutputLayer]);      
//output( ftick[OutputLayer], " ftick[OutputLayer] 2");
    deltafn[OutputLayer].set_diff2_piecewisemult3(tgt, actuation[OutputLayer], ftick[OutputLayer]);  //  deltafn[OutputLayer] = (tgt0 - actuation[OutputLayer]) % (ftick[OutputLayer]);
//output( deltafn[OutputLayer], " deltafn[OutputLayer] ");

    for (int i = OutputLayer - 1; i >= 0; i--)
    {
        deltafn_t[i + 1].set_transpose(deltafn[i + 1]);
//output( deltafn_t[i + 1], " deltafn_t[i + 1]");
        weight_updates[i].set_matmult(deltafn_t[i + 1], actuation[i]);            // weight_updates[i] = deltafn[i + 1].t() *actuation[i];
//output( weight_updates[i], "weight_updates[i]");
        new_layer_weights[i].set_mult1_add2_mat(weight_updates[i], eta, layer_weights[i]);// new_layer_weights[i] = layer_weights[i] + (eta *weight_updates[i]);
//output( new_layer_weights[i], "new_layer_weights[i]");
        ftick[i].set_mult1_add2_scalars(actuation[i], -1.0, 1.0);                              //  ftick[i] = -actuation[i] + 1;
//output( ftick[i], "ftick[i]");
        ftick[i].piecewisemult(actuation[i]);	// element wise multiply          //  ftick[i] = ftick[i] % (actuation[i]); 
//output( ftick[i], "ftick[i]2");
        deltafn[i].set_matmult(deltafn[i + 1], layer_weights[i]);                       // deltafn[i] = deltafn[i + 1] *layer_weights[i];
//output( deltafn[i], "deltafn[i]1");
        deltafn[i].piecewisemult(ftick[i]);                                             //  deltafn[i] = deltafn[i] % ftick[i];
//output( deltafn[i], "deltafn[i]2");
    }
    for (int i = 0; i < OutputLayer; i++)
    {
        for (int j = 0; j < layer_weights[i].n_rows; j++)
            for (int k = 0; k < layer_weights[i].n_cols; k++)
                layer_weights[i].ptr[j * layer_weights[i].n_cols + k] = new_layer_weights[i].ptr[j * layer_weights[i].n_cols + k];
    }
    return 0;
}


void forward_feed(unsigned char*& imgdata, unsigned char*& labdata, bool train,
    int samples)
{
    int tot_correct = 0;
    int tot_wrong = 0;
    int correct_num = -1;
    int best_guess = -1;
    int num_correct[OUTPUT_LINES] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    int num_wrong[OUTPUT_LINES] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    int chosen_wrongly[OUTPUT_LINES][OUTPUT_LINES] = {
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
         { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
         { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
         { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
         { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
         { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
         { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
         { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
         { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
         { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }
    };
    int num_tested = 0;
    int epochs;
    if (train)
    {
        epochs = EPOCHS;
    }
    else
    {
        epochs = 1;
    }
    for (int y = 0; y < samples; y++)
    {
#ifdef SAMPLEFREQ
        if ((y + 1) % SAMPLEFREQ == 0)
        {
            cout << "------------------------------------ FORWARD FEED OF ";
            if (train)
                cout << "TRAINING";
            else cout << "TEST    ";
            cout << " SAMPLE # " << y + 1 << endl << flush;
        }
#endif
        load_an_image(y, imgdata, actuation[0], tgt, labdata, epochs);

        int tgtval = tgt.index_max_row(0, 0, 9);
        for (int e = 0; e < epochs; e++)
        {
            for (int i = 0; i < OutputLayer; i++)	// only n-1 transitions between n layers
            {
#ifdef SERIAL_ONLY


                layer_weights_t[i].set_transpose(layer_weights[i]);
               // SerialMatrixVectorMultiply(netin[i].ptr,
               //     actuation[i].ptr, actuation[i].n_rows,
               //     layer_weights_t[i].ptr, layer_weights_t[i].n_rows, layer_weights_t[i].n_cols);
                netin[i].set_matmult(actuation[i], layer_weights_t[i], actuation[i].n_cols);

              //  for (int j = 0; j < netin[i].n_cols; j++)
              //  {
              //      netin[i].ptr[j] /= actuation[i].n_cols;
              //  }
#ifdef SAMPLEFREQ
                if ((y + 1) % SAMPLEFREQ == 0)
                    cout << "Netin serial (" << netin[i].n_rows << "," << netin[i].n_cols <<
                    ")= " << netin[i].prtstr() << endl << flush;
#endif
#else
                layer_weights_t[i].set_transpose(layer_weights[i]);
                netin[i].set_matmult(actuation[i], layer_weights_t[i], actuation[i].n_cols);
                //PreMatMul(actuation[i], layer_weights_t[i], netin[i], actuation[i].n_cols);

#ifdef SAMPLEFREQ
                cout << "Netin Parallel " << netin[i].n_rows << "," << netin[i].n_cols <<
                    ")= " << netin[i].prtstr() << endl << flush;
#endif

#endif
                sigmoid3(netin[i], actuation[i + 1]);
            }
#ifdef SAMPLEFREQ
            if ((y + 1) % SAMPLEFREQ == 0)
            {
                std::cout << "Final output : " << endl << std::setw(7) << fixed <<
                    showpoint << actuation[OutputLayer].prtstr() <<
                    " Sample: " << y + 1 << std::endl << flush;
                std::cout << "Expec output : " << endl << std::setw(7) << fixed <<
                    showpoint << tgt.prtstr() << " Sample: " << y + 1 <<
                    std::endl << flush;
            }
#endif
            //////////////////////////// forward feed end
            if (train)
            {
                // printout intermediate result
           //     int outval = actuation[OutputLayer].subvec(0, 9).index_max();
                int outval = actuation[OutputLayer].index_max_row(0, 0, 9);
#ifdef SAMPLEFREQ
                if ((y + 1) % SAMPLEFREQ == 0)
                {
                    std::cout << "Train output : " << endl << std::setw(7) << fixed <<
                        showpoint << actuation[OutputLayer].prtstr() <<
                        " Sample: " << y + 1 << std::endl << flush;
                    // Below just figures out the order in which to print the "A"ctal
                    // result and "O"bjective result
                    // (or "*" if correct) in the output line.
                    // So tgtval is correct if lastval==firstval(they are indicies, and
                    // will be equal if tgtval==outval)
                    int firstval = tgtval < outval ? tgtval : outval;
                    int lastval = tgtval > outval ? tgtval : outval;
                    string firststr = tgtval == firstval ?
                        to_string(firstval) + string("T") :
                        to_string(firstval) + string("O");
                    string laststr = tgtval == lastval ? to_string(lastval) + "T" :
                        to_string(lastval) + "O";
                    if (firstval == lastval)
                        firststr = "*" + to_string(firstval);	// correct
                    for (int z1 = 0; z1 < firstval; z1++)
                        cout << "         ";
                    cout << "       " << firststr;
                    for (int z1 = 0; z1 < lastval - firstval - 1; z1++)
                        cout << "         ";
                    if (firstval != lastval)
                        cout << "       " << laststr;	// expected
                    cout << endl << flush;
                }
#endif
                if (backprop(y) == 1)
                    break;	// exit i/epoch loop and goto next sample (as error function is
               // within limits for this tgt)
            }
        }

        if (!train)
        {
            correct_num = tgt.index_max_row(0, 0, 9);
            best_guess = actuation[OutputLayer].index_max_row(0, 0, 9);

            if (best_guess == correct_num)
            {
                num_correct[correct_num]++;
                tot_correct++;
            }
            else
            {
                num_wrong[correct_num]++;
                chosen_wrongly[correct_num][best_guess]++;
                tot_wrong++;
            }
            num_tested++;
        }
        if (!train)
        {
            std::cout << "Final output : " << endl << std::setw(7) << fixed <<
                showpoint << actuation[OutputLayer].prtstr() <<
                " Sample: " << y + 1 << std::endl << flush;
            for (int z1 = 0; z1 < actuation[OutputLayer].index_max_row(0, 0, 9); z1++)
                cout << "         ";
            cout << "       ^" << endl << flush;
            std::cout << "Expec output : " << endl << std::setw(7) << fixed <<
                showpoint << tgt.prtstr() << " Sample: " << y + 1 <<
                std::endl << flush;
        }
    }
    if (!train)
    {
        confusion_matrix << "Tested         " << num_tested << " samples" << endl <<
            flush;
        confusion_matrix << "Tested Correct " << tot_correct << " samples" << endl <<
            flush;
        confusion_matrix << "Tested Wrong   " << tot_wrong << " samples" << endl <<
            endl << endl << "  " << flush;
        for (int i = 0; i < OUTPUT_LINES; i++)
            confusion_matrix << "     " << dec << std::setw(6) << "'" << i << "'";
        confusion_matrix << "<-- ANN chose" << endl << flush;
        confusion_matrix << "------------------------------------------------------"
            "------------------------------------------------------"
            "-----------------------------";
        double colsum[OUTPUT_LINES] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
        double rowsum[OUTPUT_LINES] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
        string blanks = "                    ";
        for (int i = 0; i < OUTPUT_LINES; i++)
        {
            string correct_size = to_string(num_correct[i]);
            confusion_matrix << endl << setw(4) << i << "  |";
            for (int j = 0; j < OUTPUT_LINES; j++)
            {
                rowsum[i] += chosen_wrongly[i][j];
                colsum[j] += chosen_wrongly[i][j];
                if (i == j)
                    confusion_matrix << std::setw(6) << "[" << num_correct[i] << "]" <<
                    blanks.substr(0, 5 - correct_size.length()) <<
                    "|";
                else
                    confusion_matrix << std::setw(7) << chosen_wrongly[i][j] << "     |";
            }
            float pctg = 0;
            if (tot_wrong != 0)
                pctg = (float)(rowsum[i]) / (float)(tot_wrong) * 100.0f;
            confusion_matrix << "  " << setw(7) << std::setw(7);
            confusion_matrix.copyfmt(init);
            confusion_matrix << rowsum[i];
            confusion_matrix << setw(7) << "     " << fixed << showpoint << pctg <<
                "%" << endl << flush;
            confusion_matrix.copyfmt(init);
            confusion_matrix << "----------------------------------------------------"
                "----------------------------------------------------"
                "---------------------------------";
        }
        confusion_matrix << endl << "   ^   ";
        for (int i = 0; i < OUTPUT_LINES; i++)
            confusion_matrix << dec << std::setw(7) << colsum[i] << "      ";
        confusion_matrix << endl << "Target   ";
        for (int i = 0; i < OUTPUT_LINES; i++)
        {
            float pctg = 0;
            if (tot_wrong != 0)
                pctg = (float)(colsum[i]) / (float)(tot_wrong) * 100.0f;
            confusion_matrix << dec << setw(7) << fixed << showpoint << pctg <<
                "%     ";
            confusion_matrix.copyfmt(init);
        }
        confusion_matrix << endl << endl << endl << endl << endl <<
            "Correct selections:" << endl << flush;
        confusion_matrix << "       ";
        for (int i = 0; i < OUTPUT_LINES; i++)
            confusion_matrix << dec << std::setw(6) << "'" << i << "'     ";
        confusion_matrix << endl << "       ";
        for (int i = 0; i < OUTPUT_LINES; i++)
        {
            confusion_matrix << std::setw(7) << num_correct[i] << "      ";
        }
        confusion_matrix << endl << endl << "Incorrect selections:" << endl <<
            flush;
        confusion_matrix << "       ";
        for (int i = 0; i < OUTPUT_LINES; i++)
            confusion_matrix << dec << std::setw(6) << "'" << i << "'     ";
        confusion_matrix << endl << "       ";
        for (int i = 0; i < OUTPUT_LINES; i++)
        {
            confusion_matrix << std::setw(7) << num_wrong[i] << "      ";
        }
        confusion_matrix << endl << endl << flush;
        float pctg =
            (float)(tot_correct) / (float)(tot_correct + tot_wrong) * 100.0f;
        confusion_matrix << "Total Correct : " << std::setw(7) << fixed << showpoint <<
            pctg << "%     " << endl << endl << flush;
        cout << confusion_matrix.str() << flush;
        confusion_matrix.copyfmt(init);
        cout.copyfmt(init);
    }
}

#ifdef WANT_TO_LOAD_WEIGHTS
void load_weights(string fname)
{
    ifstream iFile;
    cout << "Loading weights from file : " << fname << endl << flush;
    iFile.open(fname, ios::in);
    string aline;
    int olaycnt = 0;

    vector<string> strs;

    if (fname.substr(0, 4) == "post")
        stringstream confusion_matrix2;
    getline(iFile, aline);
    boost::split(strs, aline, boost::is_any_of("="));
    if (strs.size() > 1)
        lays = stoi(strs[1]);
    cout << "Layer # " << ++olaycnt << " Has " << lays << " nodes" << endl;
    while (iFile.good())
    {
        getline(iFile, aline);
        if (aline.find("NodesInLayer") != std::string::npos)
        {
            nd2[t] = x;
            x = 0;
            t++;

            boost::split(strs, aline, boost::is_any_of("="));
            if (strs.size() > 1)
                nd[t] = stoi(strs[1]);
            cout << "Layer # " << ++olaycnt << " Has " << nd[t] << "layers" << endl;
        }
        else if ((aline.find("Error Summary") != std::string::npos))
        {
            nd2[t] = x;
            x = 0;
            t++;
            break;
        }
        else if (aline.find("LayerBiases") == std::string::npos)
        {
            boost::trim(aline);
            boost::split(strs, aline, boost::is_any_of(" "));
            boost::algorithm::split(strs, aline, boost::is_any_of("\t "),
                boost::token_compress_on);
            for (int y = 0; y < strs.size(); y++)
            {
                if (strs[y].length() > 0)
                {
                    l2[t][x++] = stod(strs[y]);
                }
            }
        }
    }
    for (int i = 0; i < lays; i++)
    {
        for (int j = 0; j < nd2[i + 1]; j++)
        {
            int r = (j) / (nd[i + 1] + 1);
            int c = (j) % (nd[i + 1] + 1);
            layer_weights[i].ptr[r * (nd[i + 1] + 1) + c] = l2[i + 1][j];
        }
    }
}
#endif

void save_weights(string hdr)
{
    ofstream oFile;
    string fname = hdr + string("_weights_") + fid + string(".txt");
    cout << "Saving weights to file : " << fname << endl << flush;
    oFile.open(fname, ios::out);
    if (hdr.substr(0, 4) == "post")
        oFile << confusion_matrix.str();
    oFile << "NumberOfLayers=" << NumberOfLayers << endl << flush;
    for (int i = 0; i < OutputLayer; i++)
    {

        oFile << "NodesInLayer" << i << "=" << nodes[i] << endl << setprecision(20) << fixed << showpoint << flush;
        oFile << "    " << layer_weights[i].prtstr();
        oFile << endl;
        // ALTERNATIVELY
        //for (int j=0; j<layer_weights[i].n_rows ;j++)
        //{
        //   for (int k=0; k<layer_weights[i].n_cols; k++)
        //        oFile << setprecision(20) <<  fixed << showpoint << layer_weights[i](j,k) << " " <<flush;
        //    oFile << endl;
        // }
    }

    oFile << "EndFile" << endl << flush;
    oFile.close();
    cout.copyfmt(init);

}

int main2()
{
    size_t available, total;
    cudaMemGetInfo(&available, &total);
    int nDevices;

    cudaGetDeviceCount(&nDevices);
    cout << "avail=" << available << " total=" << total << " ndevices=" << nDevices <<endl;
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n",
            prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n",
            prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
            2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    }
#ifndef SERIAL_ONLY
    checkError(cudaMalloc((void**)&deviceA, 100000* sizeof(double)));
    checkError(cudaMalloc((void**)&deviceB, 100000* sizeof(double)));
    checkError(cudaMalloc((void**)&deviceC, 100000* sizeof(double)));
#endif
newmat a(4,3);
newmat b(3,4);
newmat c(4,4);

newmat d(4,3);
for (int i =0;i<a.n_rows;i++)
   for (int j=0;j<a.n_cols;j++)
   {
       a.ptr[i*a.n_cols+j] = i+j;
   }
for (int i =0;i<b.n_rows;i++)
   for (int j=0;j<b.n_cols;j++)
   {
     b.ptr[i*b.n_cols+j] = i*2+j*2;
   }

for (int i =0;i<c.n_rows;i++)
   for (int j=0;j<c.n_cols;j++)
   {
     c.ptr[i*c.n_cols+j] = i*(3+j+1); 
   }
a.add_scalar(5);
a.output_all_procs(cout);
//output(a);
//output(b);
//output(c);
c.set_matmult(a,b);
cout << "-------------------" << endl;
//output(c);
//output(d);
c.output_all_procs(cout);
//output(c);
//output(b);
exit(1);
}
int main()
{
   time_measurement main_time("main");
   time_measurement initialise_time("initialise");
   time_measurement train_time("train");
   time_measurement test_time("test");
#ifdef USE_CUBLAS
	// Create a handle for CUBLAS
	cublasCreate(&handle);
#endif
    size_t available, total;
    cudaMemGetInfo(&available, &total);
    confusion_matrix << "Info: Available Memory=" << available << " Total=" << total << endl;
    int nDevices;

    cudaGetDeviceCount(&nDevices);
    confusion_matrix << "Info: Number of devices available = " << nDevices << endl;
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        confusion_matrix << "Info: Device Number: "<< i << endl;
        confusion_matrix << "  Device name: " << prop.name << endl;
        confusion_matrix << "  Memory Clock Rate (KHz): "<< prop.memoryClockRate << endl;
        confusion_matrix << "  Memory Bus Width (bits): "<< prop.memoryBusWidth << "\n";
        double val=(double)2.0 * (double)prop.memoryClockRate * (double)(prop.memoryBusWidth / (double)8) / (double)1.0e6 ;
        confusion_matrix << "  Peak Memory Bandwidth (GB/s): "<< val << endl;
    }
    extern char** environ;
    string hname = "";
#ifdef WANT_TO_LOAD_WEIGHTS
    string weight_file_to_preload = "initial_random_values_weights_1637223695.txt";
#endif
    main_time.start_measurement();
    initialise_time.start_measurement();

    vector<string> strs;
    string bldver = string(__DATE__) + " at time " + string(__TIME__);
    cout << "--------------------------------  Build done on " << bldver << endl <<
        flush;
    init.copyfmt(cout);
    for (int i = 0; i < err_summary.n_cols; i++)
        err_summary.ptr[i] = -1.0;
   
        //NumberOfLayers = 4;
        NumberOfLayers = 3;
        nodes = new unsigned int[NumberOfLayers];
        nodes[0] = INPUT_LINES;
        nodes[1] = DEFAULT_HIDDEN1;
        nodes[2] = OUTPUT_LINES;
        //nodes[1] = DEFAULT_HIDDEN1;
        //nodes[2] = DEFAULT_HIDDEN2;
        //nodes[3] = OUTPUT_LINES;
        eta = ETA_DEFAULT;
        cout << "Using default setting of \"";
        for (int i = 0; i < NumberOfLayers; i++)
        {
            cout << nodes[i]; 
            if (i < NumberOfLayers - 1)
                cout << " "; 
        }
        cout << "\" " << endl << flush;
        cout << "And ETA=" << eta << endl << flush;;
    



    cout << "Number of Layers is " << NumberOfLayers << endl << flush;


#ifndef SERIAL_ONLY
         // set up CUDA timing structs
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
#endif

    OutputLayer = NumberOfLayers - 1;
    unsigned char* trainlabels;
    unsigned char* testlabels;
    unsigned char* traindata = load_file("train-images-idx3-ubyte",
        "train-labels-idx1-ubyte", &trainlabels);
    unsigned char* testdata = load_file("t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte", &testlabels);

    ///////////////////////////////////////////////
    //
    //  CREATE ARRAY OF MATRICES AND VECTORS
    //  AND SET WEIGHTS TO RANDOM (0<w < 1)
    //
    int max_mat = 0;
    int max_vec = 0;
    int bias_field = 1;

    for (int i = 0; i <= OutputLayer; i++)
    {
        max_vec = max(max_vec, (nodes[i] + bias_field));
        newmat rb3a(1, nodes[i] + bias_field);
        actuation.push_back(rb3a);

        newmat drb3(1, nodes[i] + bias_field);
        deltafn.push_back(drb3);

        newmat drb3_t(nodes[i] + bias_field, 1);
        deltafn_t.push_back(drb3_t);

        newmat frb3(1, nodes[i] + bias_field);
        ftick.push_back(frb3);

        if (i < OutputLayer)
        {
            max_mat =
                max(max_mat, (nodes[i] + bias_field) * (nodes[i + 1] + bias_field));
            // These buffers for the rowvec and mat structures below are done to ensure
            // the Armadillo matrix can be accessed directly and the library doesnt move
            // the memory around
            newmat rb3(1, (nodes[i + 1] + bias_field));

            // Create an array of matrices (one element for each layer) for the netin value
            // This holds the sum of weighted signals, for each node, that gets squashed to 
            // produce the nodes output for next layer
            netin.push_back(rb3);
            // Create a buffer of required size for weights, in each layer
            // (plus two more, one for delta updates, and one for holding new weight to be
            // applied after backprop. These maybe consolidated later
            newmat tmpwgt3((nodes[i + 1] + bias_field), (nodes[i] + bias_field));
            newmat tmpwgt3_t((nodes[i] + bias_field), (nodes[i + 1] + bias_field));
            for (int p = 0; p < (nodes[i + 1] + bias_field) * (nodes[i] + bias_field); p++)
                tmpwgt3.ptr[p] = (double)rand() / (double)RAND_MAX;
            newmat tmpwgt30((nodes[i + 1] + bias_field), (nodes[i] + bias_field));
            newmat tmpwgt300((nodes[i + 1] + bias_field), (nodes[i] + bias_field));
            // create an array of three matrices (weights for forward prop)
            // and deltas and new values, for back propagation
            layer_weights.push_back(tmpwgt3);
            layer_weights_t.push_back(tmpwgt3_t);

            new_layer_weights.push_back(tmpwgt30);

            weight_updates.push_back(tmpwgt300);
        }
    }

    // Informational, the max value of matrix and vectors are record and used to reserve CUDA memory 
    confusion_matrix << "Max Matrix size " << max_mat << " Max vector size = " << max_vec <<
        endl << flush;

#ifdef WANT_TO_LOAD_WEIGHTS
    // this is a function to load previously saved weights, to either ensure constant initial values
    // if say moving platforms with different psudeo RNG, or to load post weights after training
    // This works, but only implemented atm, by direct code changes, no UI implemented
    // But note used in this project anyway
    confusion_matrix  << "Chosen to load saved weight file '" << weight_file_to_preload << "' , so loading it ....." << endl;
    load_weights(weight_file_to_preload);
#else
    // Save initial starting weights if required for later
    save_weights("initial_random_values");
#endif

#ifndef SERIAL_ONLY
    checkError(cudaMalloc((void**)&deviceA, max_mat * sizeof(double)));
    checkError(cudaMalloc((void**)&deviceB, max_mat * sizeof(double)));
    checkError(cudaMalloc((void**)&deviceC, max_mat * sizeof(double)));

#ifdef __CUDA_ARCH__
    confusion_matrix << "Built for CUDA ARCH == " << __CUDA_ARCH__ << endl;
#endif
#endif
    initialise_time.stop_measurement();
    ///////////////////////////////////////////////
    //
    // TRAIN THE DATA
    //
    train_time.start_measurement();
    confusion_matrix<< "Training on data started (epochs=" << EPOCHS << ")...." << endl <<
        flush;

    forward_feed(traindata, trainlabels, true, TRAININGSAMPLES);
    train_time.stop_measurement();

    confusion_matrix  << "Training complete" << endl << flush;
    ///////////////////////////////////////////////
    //
    // TEST THE DATA
    //
    test_time.start_measurement();
    confusion_matrix << "Testing of data started...." << endl << flush;

    forward_feed(testdata, testlabels, false, TESTINGSAMPLES);

    test_time.stop_measurement();

    time_output << "Testing complete" << endl << flush;

    main_time.stop_measurement();
    time_output << "Total Time       : " << std::setw(12) << main_time.accumulated_time() << " us" <<
        endl << flush;
    time_output << "Total Time       : " << std::setw(12) << main_time.last_time() << " us" <<
        endl << flush;
    time_output << "Initialise Time  : " << std::setw(12) << initialise_time.accumulated_time() << " us" <<
        endl << flush;
    time_output << "Total Train Time : " << std::setw(12) << train_time.accumulated_time() << " us" <<
        endl << flush;
    time_output << "Total Test Time  : " << std::setw(12) << test_time.accumulated_time() << " us" <<
        endl << flush;

    time_output << "Epochs in Training : " << EPOCHS << endl << flush;
    time_output << "Training Samples   : " << TRAININGSAMPLES << endl <<
        flush;
    time_output << "Testing Samples    : " << TESTINGSAMPLES << endl <<
        flush;
    time_output << "Epsilon  : " << EPSILON << endl << flush;
    time_output << "Eta      : " << eta << endl << flush;
    time_output << "Build ver: " << bldver << endl << flush;
    time_output << "Error Summary" << endl << flush;

    time_output << err_summary.prtstr() << endl << flush;


    for (int i = 0; i <= OutputLayer; i++)
    {
        if (i < OutputLayer)
        {
            netin[i].free_ele();
            layer_weights[i].free_ele();
            layer_weights_t[i].free_ele();
            new_layer_weights[i].free_ele();
            weight_updates[i].free_ele();
        }
        actuation[i].free_ele();
        deltafn[i].free_ele();
        deltafn_t[i].free_ele();
        if (i==OutputLayer)
         ftick[i].output_all_procs( time_output );
        ftick[i].free_ele();
    }

    confusion_matrix << time_output.str();
    cout << confusion_matrix.str();

    save_weights("post_training_weights");

    delete[] traindata;
    delete[] trainlabels;
    delete[] testdata;
    delete[] testlabels;
    //cout << "Min time for " << build_type << " call : " << Call_MinTime.count() << " ns" << endl;
    //cout << "Total time for all " << avgcnt << " calls is " << Avg_Time.count() << " ns" << endl;
    //cout << "Avg time for " << build_type << " call : " << (double)Avg_Time.count() / (double)avgcnt << " ns" << endl;
    //cout << "Avg time for " << build_type << " call : " << (double)Avg_Time.count() / (double)(avgcnt * 1e09) << " s" << endl;
    //cout << "Max time for " << build_type << " call : " << Call_MaxTime.count() << " ns" << endl;
    //cout << "Max time for " << build_type << " call : " << Call_MaxTime.count() / 1000000000 << " s" << endl;
    //auto EndChronoTime = std::chrono::high_resolution_clock::now();
    //auto TotalChronoTime = std::chrono::duration_cast<std::chrono::nanoseconds> (EndChronoTime - StartChronoTime);




#ifndef SERIAL_ONLY
    cout << "Used Tile Dimension of " << tile_dimension << endl;
    checkError(cudaFree(deviceA));
    checkError(cudaFree(deviceB));
    checkError(cudaFree(deviceC));

    checkError(cudaEventDestroy(start));
    checkError(cudaEventDestroy(stop));
#ifdef USE_CUBLAS
	// Destroy the handle
	cublasDestroy(handle);
#endif
#endif


}

