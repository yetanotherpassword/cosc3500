#ifndef _WIN64
#include <unistd.h>
#endif
#include <iomanip>

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
#include <signal.h>

// Application Parameters
#define INPUT_LINES 784
#define OUTPUT_LINES 10
#define MATRIX_SIDE 28
#define MAX_PIXEL_VAL 255.0f
#define IMAGE_OFFSET 16
#define DEFAULT_HIDDEN 30
#define DEFAULT_HIDDEN1 500
#define DEFAULT_HIDDEN2 300
#define DEFAULT_HIDDEN3 20
#define DEFAULT_HIDDEN4 20
#define DEFAULT_HIDDEN5 20
#define DEFAULT_HIDDEN6 15

#define ETA_DEFAULT 0.5f
#define EPSILON 1E-04
#define TRAININGSAMPLES 60000
#define TESTINGSAMPLES 10000
#define EPOCHS 64
#define THREADS_PER_2BLKDIM 32 
#define THREADS_PER_1BLKDIM 256
#define TILES 32
// Wrapper for cuda memcpy to ensure size is ok
#define MyCUDAMemCpy(A, B, C, D) if (C>max_bytes || C<=0) { cout << "Error on " << __LINE__ << " as cudaMemcpy attempt (" << C << ") is invalid for allocation of " << max_bytes << endl; exit(1); } else checkError(cudaMemcpy(A,B,C,D))
#define TRY try {
#define CATCH } catch (const std::exception& e) { ouch("Exception was caught, on line " + to_string(__LINE__) + " with message '" + e.what() + "'\n" ); } 

// How often to print samples, 1=All, 2=every second one, etc
// Undefine to remove output
#define SAMPLEFREQ 1000
//#undef SAMPLEFREQ

// SERIAL_ONLY macro passed at compile time to determine if building SERIAL or PARALLEL (default) vesrion
#ifdef SERIAL_ONLY
#define BUILT_TYPE "Serial"
#else
#define BUILT_TYPE "Parallel"
#endif

int max_mat = 0;
int max_vec = 0;
int max_bytes = 0;

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


using namespace std;

float mintime = std::numeric_limits<float>::max();
float maxtime = std::numeric_limits<float>::min();

std::chrono::milliseconds Process_MaxTime = std::chrono::milliseconds::min();
std::chrono::milliseconds Process_MinTime = std::chrono::milliseconds::max();
class time_measurement
{
public:
    std::chrono::milliseconds Call_MaxTime;
    std::chrono::milliseconds Call_MinTime;
    std::chrono::milliseconds TotalCallTime;
//    std::chrono::system_clock::time_point StartCallTime;
    
#ifndef _WIN64
    std::chrono::system_clock::time_point StartCallTime;
    std::chrono::system_clock::time_point EndCallTime;
#else
    std::chrono::steady_clock::time_point StartCallTime;
    std::chrono::steady_clock::time_point EndCallTime;
#endif
    std::chrono::milliseconds Tot_Time;
    int Tot_Cnt;
    int depth;
    bool in_measurement;
    string name;
    time_measurement(string n = "")
    {
        depth = 0;
        in_measurement = false;
        Call_MaxTime = std::chrono::milliseconds::min();
        Call_MinTime = std::chrono::milliseconds::max();
        Tot_Cnt = 0;
        Tot_Time = std::chrono::milliseconds::zero();
        name = n;
    };
    void start_measurement()
    {
        if (in_measurement)
        {
            depth++;
        }
        else
        {
            if (depth == 0)
            {
                depth = 1;
                in_measurement = true;
                StartCallTime =  std::chrono::high_resolution_clock::now();
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
                TotalCallTime = std::chrono::duration_cast<std::chrono::milliseconds> (EndCallTime - StartCallTime);

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
            s = "Never Called";
            return s;
        }
        else
        {
            int64_t t = Tot_Time.count() / (int64_t)Tot_Cnt;
            s = to_string(t);
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
        string s2 = avg_time();
        if (s2 == "Never Called")
        {
            s << "For " << name << ":" << endl;
            s << "   No measurements as wasnt called" << endl << flush;
        }
        else
        {
            string bt2;
            if (b == "Parallel")
                bt2 = "|Parallel|             ";
            else
                bt2 = "..Serial..";
            string blank = "                              ";
            string filler = blank.substr(name.length());
            s << "For " << name << bt2 << filler << endl;
            s << "   Min Time    : " << min_time() << " ms" << endl;
            s << "   Max Time    : " << max_time() << " ms" << endl;
            s << "   All Time    : " << accumulated_time() << " ms" << endl;
            s << "   No of Calls : " << number_of_calls() << "        " << endl;
            s << "   Avg Time    : " << s2 << " ms" << endl;
            s << "   Last Time   : " << last_time() << " ms" << endl << flush;
        }
        return s.str();
    };

};

#ifndef SERIAL_ONLY
class newmat;  // Forward declaration
double* LayerWeightsDevice;
double* ActuationDevice;
double* NetinDevice;
double* deviceA, * deviceB, * deviceC;
cudaEvent_t start, stop;
void PreMatMul(newmat& a, newmat& b, newmat& c, int norm);
__global__ void MatAddMat(double* A, double* B, double* C, int len) 
{
    int idx = blockIdx.x * blockDim.x  + threadIdx.x;
    if (idx < len)
    {
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void MatSubMat(double* A, double* B, double* C, int len) 
{


    int idx = blockIdx.x * blockDim.x  + threadIdx.x;
    if (idx < len)
    {
        C[idx] = A[idx] - B[idx];
    }

}

__global__ void MatMultMat(double* A, double* B, double* C, int ARows, int ACols, int BRows, int BCols, int CRows, int CCols) 
{

    double CValue = 0;

    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    for (int k = 0; k < (THREADS_PER_2BLKDIM + ACols - 1) / THREADS_PER_2BLKDIM; k++) 
    {

        for (int n = 0; n < THREADS_PER_2BLKDIM; ++n)
            if ((k * THREADS_PER_2BLKDIM + n < ACols && Row < ARows) && (k * THREADS_PER_2BLKDIM + n < BRows && Col < BCols))
                CValue += A[Row * ACols + k * THREADS_PER_2BLKDIM + n] * B[(k * THREADS_PER_2BLKDIM + n) * BCols + Col];

    }

    if (Row < CRows && Col < CCols) C[((blockIdx.y * blockDim.y + threadIdx.y) * CCols) + (blockIdx.x * blockDim.x) + threadIdx.x] = CValue;
}


__global__ void transpose(double* A, double* C, int r, int c)
{
        int ele = blockDim.x  * blockIdx.x + threadIdx.x;
        int col = ele % c;
        int row = ele / c;
        
        if(col < c && row < r)
        {
                C[row + col*r] = A[col + row*c];
        }
}


__global__ void MatMultMatEleWise(double* A, double* B, double* C, int max)
{
         int ele = blockDim.x  * blockIdx.x + threadIdx.x;
         if (ele < max)
            C[ele] = A[ele] * B[ele];
}

__global__ void MatAddScalar(double scalar, double* C, int n)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i<n)
        C[i] = C[i] + scalar;
}


__global__ void MatDivScalar(double scalar, double* C, int n)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i<n)
        C[i] = C[i] / scalar;
}


__global__ void MatMultScalar(double scalar, double* C, int n)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i<n)
        C[i] = C[i] * scalar;
}



#else
///////////////////////////////////////////////////////////////////
// Serial implementation of the matrix-vector multiply function
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
            }
            Y[i * m_nc + j] = Y[i * m_nc + j] / (double)norm;
        }
    }

}
#endif
// Associative Array type Mapping of times to names (for discrete function timing)
typedef std::map< std::string, time_measurement* > maptype;

void ouch (string s)
{
     cerr << s << endl << flush;
     exit(1);
}

class newmat 
{
public:
    double* ptr;
    int n_rows;
    int n_cols;
    bool destruct; // flag to invoke destructor or not

    static time_measurement* timeptr;
    static map< string, time_measurement* >* timers;  // timers are static, ie one copy for all instantiations

    const string build_type = BUILT_TYPE;
    void delete_timers()
    {
       for(maptype::iterator iter = timers->begin(); iter != timers->end(); ++iter)
       {
          delete (*timers)[iter->first];
       }
       delete timers;
    };
    void create_new_time_meas(string s)
    {
      try
      {
        timeptr = new time_measurement(s);
        if (timeptr == NULL)
        {
            ouch("Error: Failed to allocate memory for timeptr in create_new_time_meas");
        }
      }
      catch (const std::exception& e) 
      { 
         ouch("Exception was caught, on line " + to_string(__LINE__) + " with message '" + e.what() + "'\n" );
      }
        (*timers)[s] = timeptr;
    }
    void init_timers()
    {
        if (timers == NULL)
        {
          try
          {
            timers = new  map< string, time_measurement* >;
            if (timers == NULL)
            {
                ouch("Error: Failed to allocate memory for timers in init_timers");
            }
          }
          catch (const std::exception& e) 
          { 
             ouch("Exception was caught, on line " + to_string(__LINE__) + " with message '" + e.what() + "'\n" );
          }
        }
        if (timeptr == NULL)
        {
            create_new_time_meas("set_transpose");
            create_new_time_meas("add_mat");
            create_new_time_meas("sub_mat");
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

    newmat(int r=0, int c=0, bool d=false)
    {
        init_timers();
        n_rows = r;
        n_cols = c;
        destruct = d;
        if (r != 0 && c != 0)
        {
          try
          {
           ptr = new double[r * c];
           if (ptr == NULL)
           {
               ouch("Error: Failed to allocate memory (newmat constructor) at line "+to_string(__LINE__));
           }
          }
          catch (const std::exception& e) 
          { 
             ouch("Exception was caught, on line " + to_string(__LINE__) + " with message '" + e.what() + "'\n" );
          }
        }
        else
        {
           ptr = NULL;
           destruct = false;
           n_rows = 0;
           n_cols = 0;
        }
    };

    ~newmat()
    {
        if (destruct)
        {
            delete[] ptr;
            ptr = NULL;
            n_rows = 0;
            n_cols = 0;
        }
    };

    void set_serial_transpose(newmat & tmp)
    {
        if (n_rows == tmp.n_cols && n_cols == tmp.n_rows)
        {
            for (int i = 0; i < tmp.n_rows; i++)
                for (int j = 0; j < tmp.n_cols; j++)
                    ptr[j * tmp.n_rows + i] = tmp.ptr[i * tmp.n_cols + j];
        }
    };

    void set_transpose(newmat &  tmp)
    {
        (*timers)["set_transpose"]->start_measurement();
        if (n_rows == tmp.n_cols && n_cols == tmp.n_rows)
        {
#ifdef SERIAL_ONLY
            for (int i = 0; i < tmp.n_rows; i++)
                for (int j = 0; j < tmp.n_cols; j++)
                    ptr[j * tmp.n_rows + i] = tmp.ptr[i * tmp.n_cols + j];
#else
            int onedLen = tmp.n_rows * tmp.n_cols;
            MyCUDAMemCpy(deviceA, tmp.ptr, onedLen * sizeof(double), cudaMemcpyHostToDevice);

            int threads=TILES;
            int gridSize = (onedLen + TILES - 1) / TILES; 
              
            transpose <<< gridSize, threads >> > (deviceA, deviceC, tmp.n_rows, tmp.n_cols);

            checkError(cudaDeviceSynchronize());

            MyCUDAMemCpy(ptr, deviceC, onedLen * sizeof(double), cudaMemcpyDeviceToHost);
            
#endif
        }
        else
        {
            cout << "Cant transpose Matrix[" << tmp.n_rows << "," << tmp.n_cols << "] into given Matrix[" << n_rows << "," << n_cols << "]" << endl << flush;
            exit(1);
        }
        (*timers)["set_transpose"]->stop_measurement();
    };

    void add_mat(newmat & m1)
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
        int threads =THREADS_PER_1BLKDIM;
        int dimGrid= ((onedLen + THREADS_PER_1BLKDIM - 1) / THREADS_PER_1BLKDIM);

        MyCUDAMemCpy(deviceA, ptr, onedLen * sizeof(double), cudaMemcpyHostToDevice);
        MyCUDAMemCpy(deviceB, m1.ptr, onedLen * sizeof(double), cudaMemcpyHostToDevice);

        MatAddMat << <   dimGrid, threads >> > (deviceA, deviceB, deviceC, onedLen);

        checkError(cudaDeviceSynchronize());

        MyCUDAMemCpy(ptr, deviceC, onedLen * sizeof(double), cudaMemcpyDeviceToHost);
#endif
        (*timers)["add_mat"]->stop_measurement();
    };

    void sub_mat(newmat & m1)
    {
        (*timers)["sub_mat"]->start_measurement();
        if (m1.n_rows != n_rows || m1.n_cols != n_cols)
        {
            cout << "Cant submat m1[" << m1.n_rows << "," << m1.n_cols << "] from *this[" << n_rows << "," << n_cols << "]" << endl << flush;
            exit(1);
        }
#ifdef SERIAL_ONLY
        for (int i = 0; i < n_rows; i++)
            for (int j = 0; j < n_cols; j++)
                ptr[i * n_cols + j] -= m1.ptr[i * n_cols + j];
#else
        int onedLen = n_rows * n_cols;


        int threads =THREADS_PER_1BLKDIM;
        int dimGrid= ((onedLen + THREADS_PER_1BLKDIM - 1) / THREADS_PER_1BLKDIM);

        MyCUDAMemCpy(deviceA, ptr, onedLen * sizeof(double), cudaMemcpyHostToDevice);
        MyCUDAMemCpy(deviceB, m1.ptr, onedLen * sizeof(double), cudaMemcpyHostToDevice);

        MatSubMat << <   dimGrid, threads >> > (deviceA, deviceB, deviceC, onedLen);

        checkError(cudaDeviceSynchronize());

        MyCUDAMemCpy(ptr, deviceC, onedLen * sizeof(double), cudaMemcpyDeviceToHost);
#endif
        (*timers)["sub_mat"]->stop_measurement();
    };

    void add_scalar(double d)
    {
        (*timers)["add_scalar"]->start_measurement();

#ifdef SERIAL_ONLY
        for (int i = 0; i < n_rows; i++)
            for (int j = 0; j < n_cols; j++)
                ptr[i * n_cols + j] += d;
#else
        int blockSize, minGridSize;
        int onedLen = n_rows * n_cols;
        MyCUDAMemCpy(deviceC, ptr, onedLen * sizeof(double), cudaMemcpyHostToDevice);
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, MatAddScalar, 0, onedLen);

        // Round up according to array size 
        int gridSize = (onedLen + blockSize - 1) / blockSize; 

        MatAddScalar << < gridSize, blockSize >> > (d, deviceC, onedLen);

        checkError(cudaDeviceSynchronize());

        MyCUDAMemCpy(ptr, deviceC, onedLen * sizeof(double), cudaMemcpyDeviceToHost);
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
        int blockSize, minGridSize;
        int onedLen = n_rows * n_cols;
        MyCUDAMemCpy(deviceC, ptr, onedLen * sizeof(double), cudaMemcpyHostToDevice);
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, MatAddScalar, 0, onedLen);

        // Round up according to array size 
        int gridSize = (onedLen + blockSize - 1) / blockSize; 

        MatDivScalar << < gridSize, blockSize >> > (d, deviceC, onedLen);

        checkError(cudaDeviceSynchronize());

        MyCUDAMemCpy(ptr, deviceC, onedLen * sizeof(double), cudaMemcpyDeviceToHost);
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
        int blockSize, minGridSize;
        int onedLen = n_rows * n_cols;
        MyCUDAMemCpy(deviceC, ptr, onedLen * sizeof(double), cudaMemcpyHostToDevice);
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, MatAddScalar, 0, onedLen);

        // Round up according to array size 
        int gridSize = (onedLen + blockSize - 1) / blockSize; 

        MatMultScalar << < gridSize, blockSize >> > (d, deviceC, onedLen);

        checkError(cudaDeviceSynchronize());

        MyCUDAMemCpy(ptr, deviceC, onedLen * sizeof(double), cudaMemcpyDeviceToHost);
#endif
        (*timers)["mult_scalar"]->stop_measurement();
    };

    void set_mult1_add2_scalars(newmat & y1, double d1, double d2)
    {
        (*timers)["set_mult1_add2_scalars"]->start_measurement();

        if (n_rows != y1.n_rows || n_cols != y1.n_cols)
        {
            cout << "Cant store y1[" << y1.n_rows << "," << y1.n_cols 
                 << " in *this[" << n_rows << "," << n_cols << "]" << endl;
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

    void set_mult1_add2_mat(newmat & y1, double d1, newmat & y2)
    {
        (*timers)["set_mult1_add2_mat"]->start_measurement();
        if (n_rows != y1.n_rows || n_rows != y2.n_rows || n_cols != y1.n_cols || n_cols != y2.n_cols)
        {
            cout << "Cant add y1[" << y1.n_rows << "," << y1.n_cols << "] to y2[" 
                 << y2.n_rows << "," << y2.n_cols << "] and store in *this[" 
                 << n_rows << "," << n_cols << "]" << endl;
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

//////////////////////////////////////////////////////////////////////
//
// This serial function ONLY used for testing the parallel version
//
    void set_matmult_testing_only(newmat & p1, newmat & p2, int norm = 1)
    {
        if (p1.n_cols == p2.n_rows)
        {
            if (n_rows != p1.n_rows || n_cols != p2.n_cols)
            {
                cout << "Resultant matrix wont hold result, fixing by realloc in set_matmult" << endl;
                free_ele();
                n_rows = p1.n_rows;
                n_cols = p2.n_cols;
          try
          {
                ptr = new double[n_rows * n_cols];
                if (ptr == NULL)
                {
                    ouch("Error: Failed to allocate memory (temp var in set_matmult_testing_only)");
                }
          }
          catch (const std::exception& e) 
          { 
             ouch("Exception was caught, on line " + to_string(__LINE__) + " with message '" + e.what() + "'\n" );
          }
            }

            for (int i = 0; i < p1.n_rows; ++i)
            {
                 for (int j = 0; j < p2.n_cols; ++j)
                 {
                    ptr[i * p2.n_cols + j] = 0;
                    for (int k = 0; k < p2.n_rows; ++k) // c1==r2
                    {
                       ptr[i * p2.n_cols + j] += p1.ptr[i * p2.n_rows + k] * p2.ptr[k * p2.n_cols + j];
              
                    }
                    ptr[i * p2.n_cols + j] = ptr[i * p2.n_cols + j] / (double)norm;
                  }
            }
        }
        else
        {
            cout << "Cant multiply p1[" << p1.n_rows << "," << p1.n_cols << "] by p2[" << p2.n_rows << "," << p2.n_cols << "]" << endl;
            exit(1);
        }
    };
//
//////////////////////////////////////////////////////////////////////

    void set_matmult(newmat & p1, newmat & p2, int norm = 1)
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
           try
           {
                ptr = new double[n_rows * n_cols];
                if (ptr == NULL)
                {
                    ouch("Error: Failed to re-allocate memory (temp var in set_matmult)");
                }
          }
          catch (const std::exception& e) 
          { 
             ouch("Exception was caught, on line " + to_string(__LINE__) + " with message '" + e.what() + "'\n" );
          }
            }
#ifdef SERIAL_ONLY
            SerialMatrixVectorMultiply(ptr, p1.ptr, p1.n_rows, p2.ptr, p2.n_rows, p2.n_cols, norm);
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

    void piecewisemult(newmat & p1)
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
        MyCUDAMemCpy(deviceA, ptr, onedLen * sizeof(double), cudaMemcpyHostToDevice);
        MyCUDAMemCpy(deviceB, p1.ptr, onedLen * sizeof(double), cudaMemcpyHostToDevice);

        int blockSize, minGridSize;
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, MatDivScalar, 0, onedLen);

        // Round up according to array size
        int gridSize = (onedLen + blockSize - 1) / blockSize;

        MatMultMatEleWise << <  gridSize, blockSize >> > (deviceA, deviceB, deviceC, onedLen);

        checkError(cudaDeviceSynchronize());

        MyCUDAMemCpy(ptr, deviceC, onedLen * sizeof(double), cudaMemcpyDeviceToHost);
#endif

        (*timers)["piecewisemult"]->stop_measurement();
    };

    void set_diff2_piecewisemult3(newmat & p1, newmat & p2, newmat & p3)
    {
        (*timers)["set_diff2_piecewisemult3"]->start_measurement();

        if (n_rows != p1.n_rows || n_rows != p2.n_rows || n_cols != p1.n_cols || n_cols != p2.n_cols)
        {
            cout << "Cant diff p2[" << p2.n_rows << "," << p2.n_cols << "] from p1[" 
                 << p1.n_rows << "," << p1.n_cols << "] and store in *this[" 
                 << n_rows << "," << n_cols << "]" << endl;
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
        MyCUDAMemCpy(deviceA, p1.ptr, onedLen * sizeof(double), cudaMemcpyHostToDevice);
        MyCUDAMemCpy(deviceB, p2.ptr, onedLen * sizeof(double), cudaMemcpyHostToDevice);

        int blockSize, minGridSize;
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, MatDivScalar, 0, onedLen);
  
        int gridSize = (onedLen + blockSize - 1) / blockSize;

        MatSubMat << < gridSize, blockSize >> > (deviceA, deviceB, deviceC, onedLen);
        
        checkError(cudaDeviceSynchronize());

        MyCUDAMemCpy(ptr, deviceC, onedLen * sizeof(double), cudaMemcpyDeviceToHost);

#endif
        piecewisemult(p3);

        (*timers)["set_diff2_piecewisemult3"]->stop_measurement();
    };
//////////////////////////////////////////////
//
// Print Functions
    string prtstr(string q="   ")
    {
        stringstream s;
        s << "";

        for (int i = 0; i < n_rows; i++)
        {
            for (int j = 0; j < n_cols; j++)
            {
                s<<  q << ptr[i*n_cols+j];
            }
            s << endl;
        }
        return s.str();
    };
    string prtstr_less_last_col(string q="   ")
    {
        stringstream s;
        s << "";

        for (int i = 0; i < n_rows; i++)
        {
            for (int j = 0; j < n_cols-1; j++)
            {
                s<<  q << ptr[i*n_cols+j];
            }
            s << endl;
        }
        return s.str();
    };
//
//////////////////////////////////////////////

    void free_ele()
    {
        (*timers)["free_ele"]->start_measurement();
        if (ptr != NULL)
        {
            delete[] ptr;
            ptr = NULL;
        }
        n_rows = 0;
        n_cols = 0;
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
    void output_all_procs(ostream& s)
    {
        int toplen = 60;
        for (maptype::iterator t_it = timers->begin(); t_it != timers->end(); t_it++)
        {
            string outp = (*timers)[t_it->first]->output_all_times(build_type);
            int fill = toplen - outp.length();
            s << outp;
            for (int i = 0; i < fill; i++)
                s << " ";
            s << endl;

        }
    };
};
maptype* newmat::timers = NULL;
time_measurement* newmat::timeptr = NULL;


#ifndef SERIAL_ONLY
void PreMatMul(newmat& a, newmat& b, newmat& c, int norm)
{
    if ((c.n_rows != a.n_rows) || (a.n_cols != b.n_rows) || (c.n_cols != b.n_cols))
    {
        cout << "Incorrect dimensions passed to PreMatMul" << endl;
        cout << "c(" << c.n_rows << "," << c.n_cols << ") is to be set to "
            << "a(" << a.n_rows << "," << a.n_cols << ") * "
            << "b(" << b.n_rows << "," << b.n_cols << ")  "
            << endl;
        exit(1);
    }

    MyCUDAMemCpy(deviceA, a.ptr, c.n_rows * a.n_cols * sizeof(double), cudaMemcpyHostToDevice);
    MyCUDAMemCpy(deviceB, b.ptr, a.n_cols * c.n_cols * sizeof(double), cudaMemcpyHostToDevice);

    dim3 dimBlock(THREADS_PER_2BLKDIM, THREADS_PER_2BLKDIM, 1);
    dim3 dimGrid;

    dimGrid.x = (c.n_cols + dimBlock.x - 1) / dimBlock.x;
    dimGrid.y = (c.n_rows + dimBlock.y - 1) / dimBlock.y;


    
    MatMultMat << < dimGrid, dimBlock >> > (deviceA, deviceB, deviceC, c.n_rows, a.n_cols, a.n_cols, c.n_cols, c.n_rows, c.n_cols);

   
    checkError(cudaDeviceSynchronize());

    int onedLen = c.n_rows * c.n_cols;
    if (norm != 1)
    {
        int blockSize, minGridSize;
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, MatDivScalar, 0, onedLen);

        // Round up according to array size 
        int gridSize = (onedLen + blockSize - 1) / blockSize; 

        MatDivScalar << <  gridSize, blockSize >> > ((double)norm, deviceC, onedLen);

        checkError(cudaDeviceSynchronize());

    }

    MyCUDAMemCpy(c.ptr, deviceC, onedLen * sizeof(double), cudaMemcpyDeviceToHost);

}
#endif
std::time_t result = std::time(nullptr);
string fid = to_string(result);
unsigned int NumberOfLayers=0;
unsigned int OutputLayer=0;
unsigned int* nodes=NULL;
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
newmat err_summary(1, OUTPUT_LINES);

time_measurement main_time("main");
time_measurement initialise_time("initialise");
time_measurement train_time("train");
time_measurement test_time("test");


ios init(NULL);
stringstream confusion_matrix;
stringstream time_output;
string bldver;

#ifdef WANT_TO_LOAD_WEIGHTS
// Used for loading weights from file (if ever required)
double l2[10][50000];
int nd[100];
int nd2[100];
int lays;
int t = 0;
int x = 0;
#endif

void delete_all()
{
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
        ftick[i].free_ele();
    }
}

void early_exit(string msg) {
    if (train_time.in_measurement)
    {
        train_time.stop_measurement();
    }
    else if (test_time.in_measurement)
    {
        test_time.stop_measurement();
    }

    main_time.stop_measurement();
    if (msg.length() > 0)
        cout << msg << endl;
    time_output << "Total Time       : " << std::setw(12) << main_time.accumulated_time() << " ms" <<
        endl << flush;
    time_output << "Total Time       : " << std::setw(12) << main_time.last_time() << " ms" <<
        endl << flush;
    time_output << "Initialise Time  : " << std::setw(12) << initialise_time.accumulated_time() << " ms" <<
        endl << flush;
    time_output << "Total Train Time : " << std::setw(12) << train_time.accumulated_time() << " ms" <<
        endl << flush;
    time_output << "Total Test Time  : " << std::setw(12) << test_time.accumulated_time() << " ms" <<
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

    delete_all();
    newmat dummy;
    dummy.output_all_procs(time_output);
    dummy.delete_timers();

    confusion_matrix << time_output.str();
    cout << confusion_matrix.str();
}


void signal_callback_handler(int signum) {
    cout << "Caught signal " << signum << endl;
    // Terminate program
    early_exit("********* Caught Signal " + to_string(signum) + " Exiting early *******");
    exit(0);
}


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
      try
      {
        size = inFile.tellg();
        memblock = new unsigned char[size];
        if (memblock == NULL)
        {
            ouch("Error: Failed to allocate memory (for '"+filename+"') in load_file");
        }
        inFile.seekg(0, ios::beg);
        inFile.read((char*)memblock, size);
        inFile.close();

        cout << "the entire file content is in memory, all " << size <<
            " bytes of it" << endl << flush;
      }
      catch (const std::exception& e) 
      { 
         ouch("Exception was caught, on line " + to_string(__LINE__) + " with message '" + e.what() + "'\n" );
      }
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
      try
      {
        size = inFile.tellg();
        *labs = new unsigned char[size];
        if (*labs == NULL)
        {
            ouch("Error: Failed to allocate memory (for '"+labels+"') in load_file");
        }
        inFile.seekg(0, ios::beg);
        inFile.read((char*)*labs, size);
        inFile.close();

        cout << "the entire file content is in memory, all " << size <<
            " bytes of it" << endl << flush;
      }
      catch (const std::exception& e) 
      { 
         ouch("Exception was caught, on line " + to_string(__LINE__) + " with message '" + e.what() + "'\n" );
      }
    }
    inFile.close();
    return memblock;
}

void load_an_image(int seq, unsigned char*& mptr, newmat& img, newmat& t,
    unsigned char*& lp, int e)
{
    int start = (INPUT_LINES * seq) + IMAGE_OFFSET;
   // double greyval = MAX_PIXEL_VAL;

    for (int i = 0; i < INPUT_LINES; i++)
    {
       // img.ptr[i] = ((double)mptr[start + i]) / greyval;  // 87.81%
         if (mptr[start + i] == 0)  // 88.54%
             img.ptr[i] = 0;
         else
             img.ptr[i] = 1;
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
void output(newmat & t, string g = "")
{
    cout << g << endl;
    cout << t.prtstr();
}

double accu0(newmat & m1)
{
    double tmp = 0;
    for (int i = 0; i < m1.n_rows; i++)
        for (int j = 0; j < m1.n_cols; j++)
            tmp += m1.ptr[i * m1.n_cols + j];
    return tmp;
}


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
        if (val >= 0 && val <= 9)
            err_summary.ptr[val] = err;
        else
            ouch ("Error: Problem with val="+to_string(val)+" on line "+to_string(__LINE__)+" in backprop");
        return 1;
    }

#ifdef SAMPLEFREQ
    if ((y0 + 1) % SAMPLEFREQ == 0)
        cout << "------------------------------------ BACK PROPAGATION sample=" <<
        y0 + 1 << endl << flush;
#endif
TRY
    ftick[OutputLayer].set_mult1_add2_scalars(actuation[OutputLayer], -1.0, 1.0);                            //  ftick[OutputLayer] = -actuation[OutputLayer] + 1;
CATCH
//output( ftick[OutputLayer], " ftick[OutputLayer] 1");
TRY
    ftick[OutputLayer].piecewisemult(actuation[OutputLayer]);	// element wise multiply                //  ftick[OutputLayer] = ftick[OutputLayer] % (actuation[OutputLayer]);      
CATCH
//output( ftick[OutputLayer], " ftick[OutputLayer] 2");
TRY
    deltafn[OutputLayer].set_diff2_piecewisemult3(tgt, actuation[OutputLayer], ftick[OutputLayer]);  //  deltafn[OutputLayer] = (tgt0 - actuation[OutputLayer]) % (ftick[OutputLayer]);
CATCH
//output( deltafn[OutputLayer], " deltafn[OutputLayer] ");

    for (int i = OutputLayer - 1; i >= 0; i--)
    {
TRY
        deltafn_t[i + 1].set_transpose(deltafn[i + 1]);
CATCH
        //output( deltafn_t[i + 1], " deltafn_t[i + 1]");
TRY
        weight_updates[i].set_matmult(deltafn_t[i + 1], actuation[i]);            // weight_updates[i] = deltafn[i + 1].t() *actuation[i];
CATCH
//output( weight_updates[i], "weight_updates[i]");
TRY
        new_layer_weights[i].set_mult1_add2_mat(weight_updates[i], eta, layer_weights[i]);// new_layer_weights[i] = layer_weights[i] + (eta *weight_updates[i]);
CATCH
//output( new_layer_weights[i], "new_layer_weights[i]");
TRY
        ftick[i].set_mult1_add2_scalars(actuation[i], -1.0, 1.0);                              //  ftick[i] = -actuation[i] + 1;
CATCH
//output( ftick[i], "ftick[i]");
TRY
        ftick[i].piecewisemult(actuation[i]);	// element wise multiply          //  ftick[i] = ftick[i] % (actuation[i]); 
CATCH
//output( ftick[i], "ftick[i]2");
TRY
        deltafn[i].set_matmult(deltafn[i + 1], layer_weights[i]);                       // deltafn[i] = deltafn[i + 1] *layer_weights[i];
CATCH
//output( deltafn[i], "deltafn[i]1");
TRY
        deltafn[i].piecewisemult(ftick[i]);                                             //  deltafn[i] = deltafn[i] % ftick[i];
CATCH
//output( deltafn[i], "deltafn[i]2");
    }
    for (int i = 0; i < OutputLayer; i++)
    {
        for (int j = 0; j < layer_weights[i].n_rows; j++)
            for (int k = 0; k < layer_weights[i].n_cols; k++)
            {
TRY
                layer_weights[i].ptr[j * layer_weights[i].n_cols + k] = new_layer_weights[i].ptr[j * layer_weights[i].n_cols + k];
CATCH
            }
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
#ifdef SAMPLEFREQ2
                if ((y + 1) % SAMPLEFREQ == 0)
                    cout << "Netin serial (" << netin[i].n_rows << "," << netin[i].n_cols <<
                    ")= " << netin[i].prtstr() << endl << flush;
#endif
#else
                layer_weights_t[i].set_transpose(layer_weights[i]);
                netin[i].set_matmult(actuation[i], layer_weights_t[i], actuation[i].n_cols);
                //PreMatMul(actuation[i], layer_weights_t[i], netin[i], actuation[i].n_cols);

#ifdef SAMPLEFREQ2
                if ((y + 1) % SAMPLEFREQ == 0)
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
                    showpoint << actuation[OutputLayer].prtstr_less_last_col() <<
                    " Sample: " << y + 1 << std::endl << flush;
                std::cout << "Expec output : " << endl << std::setw(7) << fixed <<
                showpoint << tgt.prtstr_less_last_col("          ") << " Sample: " << y + 1 <<
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
                        showpoint << actuation[OutputLayer].prtstr_less_last_col() <<
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
                        cout << "           ";
                    cout << "      " << firststr;
                    for (int z1 = 0; z1 < lastval - firstval - 1; z1++)
                        cout << "               ";
                    if (firstval != lastval)
                        cout << "         " << laststr;	// expected
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
                showpoint << actuation[OutputLayer].prtstr_less_last_col() <<
                " Sample: " << y + 1 << std::endl << flush;
            for (int z1 = 0; z1 < actuation[OutputLayer].index_max_row(0, 0, 9); z1++)
                cout << "         ";
            cout << "       ^" << endl << flush;
            std::cout << "Expec output : " << endl << std::setw(7) << fixed <<
                showpoint << tgt.prtstr_less_last_col("          ") << " Sample: " << y + 1 <<
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
            confusion_matrix << dec << std::setw(7) << colsum[i] << "        ";
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

int main_unit_tests()
{
    const int r1=4000;
const int c1=3000;

#ifndef SERIAL_ONLY
stringstream suffix;
int max = c1>r1?c1:r1;
max_bytes = max*sizeof(double);
    size_t available, total;
    cudaMemGetInfo(&available, &total);
    int nDevices;

    cudaGetDeviceCount(&nDevices);
    suffix << "Availiable Memory of GPU=" << available << " Total Mem=" << total << " Number of Devices=" << nDevices << endl;
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        suffix << "Device Number: " << i << endl;
        suffix << "  Device name: " << prop.name << endl;
        suffix << "  Memory Clock Rate (KHz): " << prop.memoryClockRate << endl;
        suffix << "  Memory Bus Width (bits): " << prop.memoryBusWidth << endl;
        suffix << "  Peak Memory Bandwidth (GB/s): " << 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6 << endl << endl;
    }
    checkError(cudaMalloc((void**)&deviceA, max_bytes));
    checkError(cudaMalloc((void**)&deviceB, max_bytes));
    checkError(cudaMalloc((void**)&deviceC, max_bytes));
#endif


    newmat a(r1, c1);
    newmat b(c1, r1);
    newmat c(r1, r1);
        newmat c0(r1, r1);
        newmat d3(r1, c1);
    newmat d2(r1, c1);
        newmat d5(r1, c1);
    newmat d1(r1, c1);
        newmat d0(r1, c1);
            newmat d4(r1, c1);
                    newmat d6(r1, c1);
            newmat d7(r1, c1);
                        newmat d8(r1, c1);
    newmat d(r1, c1);
    newmat e(r1, c1);

    cout << "Running version : " << e.build_type << endl;

    for (int i = 0; i < a.n_rows; i++)
        for (int j = 0; j < a.n_cols; j++)
        {
            a.ptr[i * a.n_cols + j] = 5;
            d.ptr[i * a.n_cols + j] = (i + 2) * 2 + j * 3;
            d2.ptr[i * a.n_cols + j] = (i + 2) * 2 + j * 3;
            e.ptr[i * a.n_cols + j] = (i + 2) * 2 + j * 3;
            d1.ptr[i * a.n_cols + j] = i * (3 + j + 1);
                        d0.ptr[i * a.n_cols + j] = i * (3 + j + 1);
            d3.ptr[i * a.n_cols + j] = d1.ptr[i * a.n_cols + j] * d2.ptr[i * a.n_cols + j];
            d4.ptr[i * a.n_cols + j] = d0.ptr[i * a.n_cols + j] - d2.ptr[i * a.n_cols + j];
                        d5.ptr[i * a.n_cols + j] = d2.ptr[i * a.n_cols + j] /10;
                                    d6.ptr[i * a.n_cols + j] = i * (4+ j -3);
                                    d7.ptr[i * a.n_cols + j] = d6.ptr[i * a.n_cols + j] *9 +17;
        }
    for (int i = 0; i < b.n_rows; i++)
        for (int j = 0; j < b.n_cols; j++)
        {
            b.ptr[i * b.n_cols + j] = i * 2 + j * 2;
        }

    for (int i = 0; i < c.n_rows; i++)
        for (int j = 0; j < c.n_cols; j++)
        {
            c.ptr[i * c.n_cols + j] = i * (3 + j + 1);

        }
    e.add_mat(a);
  
    int err = 0;
    for (int i = 0; i < a.n_rows; i++)
        for (int j = 0; j < a.n_cols; j++)
            if (e.ptr[i * a.n_cols + j] != d.ptr[i * d.n_cols + j] + 5)
            {
                cout << "e+5=" << e.ptr[i * e.n_cols + j] << " != d+5=" << d.ptr[i * d.n_cols + j] + 5 << endl;
                err = 1;
                break;
            }
    if (err == 1)
        cout << "Error occured in add_mat" << endl;
    else
        cout << "No Error occured in add_mat" << endl;
    

    d.add_scalar(5);

     err = 0;
    for (int i = 0; i < a.n_rows; i++)
        for (int j = 0; j < a.n_cols; j++)
            if (e.ptr[i * a.n_cols + j] != d.ptr[i * d.n_cols + j] )
            {
                cout << "e+5=" << e.ptr[i * e.n_cols + j] << " != d+5=" << d.ptr[i * d.n_cols + j]  << endl;
                err = 1;
                break;
            }
    if (err == 1)
        cout << "Error occured in add_scalar" << endl;
    else
        cout << "No Error occured in add_scalar" << endl;





err=0;
    b.set_transpose(d);
    for (int i = 0; i < b.n_rows; i++)
        for (int j = 0; j < b.n_cols; j++)
            if (b.ptr[i * b.n_cols + j] != d.ptr[j * d.n_cols + i])
            {
                cout << "error in transpose b.ptr["<<i<<","<<j <<"]==idx["<<i * b.n_cols + j << "] != d" << "== "<< b.ptr[i * b.n_cols + j] << "!="  << d.ptr[j * d.n_rows + i] << endl;
                err = 1;
                break;
            }
    if (err == 1)
    {
        cout << "Error occured in set_transpose" << endl;
        output(b,"b");
        output(d,"d");
    }
    else
        cout << "No Error occured in set_transpose" << endl;

d1.piecewisemult(d2);
err=0;
for (int i = 0; i < d1.n_rows; i++)
        for (int j = 0; j < d1.n_cols; j++)
            if (d1.ptr[i * d1.n_cols + j] != d3.ptr[ i* d1.n_cols + j])
            {
                cout << "error in piecewisemult d1.ptr["<<i<<","<<j <<"]==idx["<<i * d1.n_cols + j<< "] != d3" << "== "<< d1.ptr[i * d1.n_cols + j] << "!="  << d3.ptr[ i* d1.n_cols + j]  << endl;
                err = 1;
                break;
            }


    if (err == 1)
    {
        cout << "Error occured in piecewisemult" << endl;
        output(d1,"d1");
        output(d3,"d3");
    }
    else
        cout << "No Error occured in piecewisemult" << endl;




  d0.sub_mat(d2);

  err=0;
for (int i = 0; i < d0.n_rows; i++)
        for (int j = 0; j < d1.n_cols; j++)
            if (d0.ptr[i * d0.n_cols + j] != d4.ptr[ i* d0.n_cols + j])
            {
                cout << "error in sub_mat d0.ptr["<<i<<","<<j <<"]==idx["<<i * d0.n_cols + j<< "] != d3" << "== "<< d0.ptr[i * d0.n_cols + j] << "!="  << d4.ptr[ i* d0.n_cols + j]  << endl;
                err = 1;
                break;
            }


    if (err == 1)
    {
        cout << "Error occured in sub_mat" << endl;
        output(d0,"d0");
        output(d4,"d4");
    }
    else
        cout << "No Error occured in sub_mat" << endl;

d2.div_scalar(10);
 err=0;
for (int i = 0; i < d2.n_rows; i++)
        for (int j = 0; j < d2.n_cols; j++)
            if (d2.ptr[i * d2.n_cols + j] != d5.ptr[ i* d2.n_cols + j])
            {
                cout << "error in div_scalar d2.ptr["<<i<<","<<j <<"]==idx["<<i * d2.n_cols + j<< "] != d5" << "== "<< d5.ptr[i * d2.n_cols + j] << "!="  << d5.ptr[ i* d2.n_cols + j]  << endl;
                err = 1;
                break;
            }


    if (err == 1)
    {
        cout << "Error occured in div_scalar" << endl;
        output(d2,"d2");
        output(d5,"d5");
    }
    else
        cout << "No Error occured in div_scalar" << endl;


d8.set_mult1_add2_scalars(d6,9,17);
 err=0;
for (int i = 0; i < d8.n_rows; i++)
        for (int j = 0; j < d8.n_cols; j++)
            if (d8.ptr[i * d8.n_cols + j] != d7.ptr[ i* d8.n_cols + j])
            {
                cout << "error in set_mult1_add2_scalars d8.ptr["<<i<<","<<j <<"]==idx["<<i * d8.n_cols + j<< "] != d7" << "== "<< d8.ptr[i * d8.n_cols + j] << "!="  << d7.ptr[ i* d8.n_cols + j]  << endl;
                err = 1;
                break;
            }


    if (err == 1)
    {
        cout << "Error occured in set_mult1_add2_scalars" << endl;
        output(d8,"d8");
        output(d7,"d7");
    }
    else
        cout << "No Error occured in set_mult1_add2_scalars" << endl;



    c.set_matmult(a, b);
  
    c0.set_matmult_testing_only(a,b);

 err=0;
for (int i = 0; i < c0.n_rows; i++)
        for (int j = 0; j < c0.n_cols; j++)
            if (c.ptr[i * c0.n_cols + j] != c0.ptr[ i* c0.n_cols + j])
            {
                cout << "error in set_matmult c.ptr["<<i<<","<<j <<"]==idx["<<i * c0.n_cols + j<< "] != c0" << "== "<< c.ptr[i * c0.n_cols + j] << "!="  << c0.ptr[ i* c0.n_cols + j]  << endl;
                err = 1;
                break;
            }


    if (err == 1)
    {
        cout << "Error occured in set_matmult" << endl;
        output(c,"c");
        output(c0,"c0");
    }
    else
        cout << "No Error occured in set_matmult" << endl;




      cout << "-------------------" << endl;
    c.output_all_procs(cout);
    cout << "Tests performed on matrices of sizes " << r1 << " x " << c1 << endl;

    #ifndef SERIAL_ONLY
     cout << suffix.str();
     #endif
     return (0);
    
}
int main()
{

    signal(SIGINT, signal_callback_handler);

    size_t available, total;
    cudaMemGetInfo(&available, &total);
    confusion_matrix << "Info: Available Memory=" << available << " Total=" << total << endl;
    int nDevices;

    cudaGetDeviceCount(&nDevices);
    confusion_matrix << "Info: Number of devices available = " << nDevices << endl;
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        confusion_matrix << "Info: Device Number: " << i << endl;
        confusion_matrix << "  Device name: " << prop.name << endl;
        confusion_matrix << "  Memory Clock Rate (KHz): " << prop.memoryClockRate << endl;
        confusion_matrix << "  Memory Bus Width (bits): " << prop.memoryBusWidth << "\n";
        double val = (double)2.0 * (double)prop.memoryClockRate * (double)(prop.memoryBusWidth / (double)8) / (double)1.0e6;
        confusion_matrix << "  Peak Memory Bandwidth (GB/s): " << val << endl;
    }
    extern char** environ;
    string hname = "";
#ifdef WANT_TO_LOAD_WEIGHTS
    string weight_file_to_preload = "initial_random_values_weights_1637223695.txt";
#endif
    main_time.start_measurement();
    initialise_time.start_measurement();

    vector<string> strs;
    bldver = string(__DATE__) + " at time " + string(__TIME__);
    cout << "--------------------------------  Build done on " << bldver << endl <<
        flush;
    init.copyfmt(cout);
    for (int i = 0; i < err_summary.n_cols; i++)
        err_summary.ptr[i] = -1.0;

    NumberOfLayers = 4;

    try
    {
        nodes = new unsigned int[NumberOfLayers];
        if (nodes == NULL)
        {
            ouch("Error: Failed to allocate memory for nodes in main");
        }
    }
    catch (const std::exception& e) 
    { 
        ouch("Exception was caught, on line " + to_string(__LINE__) + " with message '" + e.what() + "'\n" );
    }
    int c=0;
    nodes[c++] = INPUT_LINES;
//    nodes[c++] = DEFAULT_HIDDEN;
    nodes[c++] = DEFAULT_HIDDEN1;
    nodes[c++] = DEFAULT_HIDDEN2;
//    nodes[c++] = DEFAULT_HIDDEN3;
//    nodes[c++] = DEFAULT_HIDDEN4;
//    nodes[c++] = DEFAULT_HIDDEN5;
//    nodes[c++] = DEFAULT_HIDDEN6;
    nodes[c++] = OUTPUT_LINES;
    if (c != NumberOfLayers)
        ouch("Check of setup appears incorrect, as "+to_string(c)+" != "+to_string(NumberOfLayers)+" See line "+to_string(__LINE__));
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
    cout << "Training epochs set to " << EPOCHS << endl;
    cout << "Build type is : " << BUILT_TYPE << endl;

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
            // Save the biggest matrix size (to ensure cuda memory allocation is sufficient)
            max_mat =
                max(max_mat, (nodes[i] + bias_field) * (nodes[i + 1] + bias_field));

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
    confusion_matrix << "Max Matrix size " << max_mat 
                     << " Max vector size = " << max_vec 
                     << endl << flush;

#ifdef WANT_TO_LOAD_WEIGHTS
    // this is a function to load previously saved weights, to either ensure constant initial values
    // if say moving platforms with different psudeo RNG, or to load post weights after training
    // This works, but only implemented atm, by direct code changes, no UI implemented
    // But note used in this project anyway
    confusion_matrix << "Chosen to load saved weight file '" << weight_file_to_preload << "' , so loading it ....." << endl;
    load_weights(weight_file_to_preload);
#else
    // Save initial starting weights if required for later
    save_weights("initial_random_values");
#endif

#ifndef SERIAL_ONLY
    max_bytes = max_mat * sizeof(double);
    checkError(cudaMalloc((void**)&deviceA, max_bytes));
    checkError(cudaMalloc((void**)&deviceB, max_bytes));
    checkError(cudaMalloc((void**)&deviceC, max_bytes));

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
    confusion_matrix << "Training on data started (epochs=" << EPOCHS << ")...." << endl <<
        flush;

    forward_feed(traindata, trainlabels, true, TRAININGSAMPLES);
    train_time.stop_measurement();

    confusion_matrix << "Training complete" << endl << flush;
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
    time_output << "Total Time       : " << std::setw(12) << main_time.accumulated_time() << " ms" <<
        endl << flush;
    time_output << "Total Time       : " << std::setw(12) << main_time.last_time() << " ms" <<
        endl << flush;
    time_output << "Initialise Time  : " << std::setw(12) << initialise_time.accumulated_time() << " ms" <<
        endl << flush;
    time_output << "Total Train Time : " << std::setw(12) << train_time.accumulated_time() << " ms" <<
        endl << flush;
    time_output << "Total Test Time  : " << std::setw(12) << test_time.accumulated_time() << " ms" <<
        endl << flush;

    time_output << "Epochs in Training : " << EPOCHS << endl << flush;
    time_output << "Training Samples   : " << TRAININGSAMPLES << endl <<
        flush;
    time_output << "Testing Samples    : " << TESTINGSAMPLES << endl <<
        flush;
    time_output << "Epsilon  : " << EPSILON << endl << flush;
    time_output << "Eta      : " << eta << endl << flush;
    time_output << "Build ver: " << bldver << endl << flush;
    time_output << "Error Summary (-1 means corresponding cost did not go less than Epsilon)" << endl << flush;

    time_output << err_summary.prtstr() << endl << flush;

    delete_all();
    newmat dummy;
    dummy.output_all_procs(time_output);
    dummy.delete_timers();

    confusion_matrix << time_output.str();
    cout << confusion_matrix.str();

    save_weights("post_training_weights");

    delete[] traindata;
    delete[] trainlabels;
    delete[] testdata;
    delete[] testlabels;

#ifndef SERIAL_ONLY
    checkError(cudaFree(deviceA));
    checkError(cudaFree(deviceB));
    checkError(cudaFree(deviceC));

    checkError(cudaEventDestroy(start));
    checkError(cudaEventDestroy(stop));
#endif

    return (0);
}

