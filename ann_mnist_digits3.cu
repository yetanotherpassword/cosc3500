#include <iomanip>
#include <cmath>
#include <chrono>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/split.hpp>
#include <vector>
#include <limits>
#include <sstream>
#include <fstream>
#include <iostream>
// Application Parameters
#define DEFTHREADS 256
#define INPUT_LINES 784
#define OUTPUT_LINES 10
#define MATRIX_SIDE 28
#define MAX_PIXEL_VAL 255.0f
#define IMAGE_OFFSET 16
#define DEFAULT_HIDDEN 30
#define ETA_DEFAULT 0.5f
#define EPSILON 1E-04
#define TRAININGSAMPLES 60000
#define TESTINGSAMPLES 10000
#define EPOCHS 1

// How often to print samples, 1=All, 2=every second one, etc
// Undefine or define to very large number to remove output
#define SAMPLEFREQ 1
#undef SAMPLEFREQ


/*
 * ALLAN CAMPTON
 * COSC3500 Milestone 2 Parallel Version
 *
 * To perform a full build and run from scratch, do the following
 *
     unzip Project_AC.zip
     cd ~/cosc3500/
     unzip mnist.zip
     unxz armadillo-10.6.2.tar.xz
     tar xvf armadillo-10.6.2.tar
     cd armadillo-10.6.2/
     mkdir build
     cd build
     cmake ..
     make
     cd ../..
     make
     sbatch ./goslurm.sh ann_mnist_digits_cuda    #Run parallel version (with default settings)
     sbatch ./goslurm.sh ann_mnist_digits_serial  #Run serial version for comparison 

       #Note for armadillo build
       #Made lib static and issue with MKL on Centos
       #Below changes done in my git, but may need to do if download from
       #http://sourceforge.net/projects/arma/files/armadillo-10.6.2.tar.xz
       #sed -i "s/add_library( armadillo/add_library( armadillo STATIC/" CMakeLists.txt
      #sed -i "s/include(ARMA_FindMKL)/#include(ARMA_FindMKL)/" CMakeLists.txt
 */


int thrds = DEFTHREADS;

using namespace std;

float mintime = std::numeric_limits<float>::max();
float maxtime = std::numeric_limits<float>::min();

std::chrono::microseconds Process_MaxTime = std::chrono::microseconds::min();
std::chrono::microseconds Process_MinTime = std::chrono::microseconds::max();

#ifndef SERIAL_ONLY
double *LayerWeightsDevice;
double *ActuationDevice;
double *NetinDevice;
cudaEvent_t start, stop;
int tile_dimension = 8; 
#endif

class Matrix {
  public:
   double * index;
   int rows;
   int cols;
   string name;
   void zeroize()
   {
        if ((index != NULL) && (rows>0) && (cols>0))
           for (int i=0;i<rows;i++)
             for (int j=0;j<cols;j++)
                  index[i*cols+j]=0;
   };
   int index_max_row(int r, int start, int stop)
   {
        int idx=0;
        double max=  std::numeric_limits<double>::min();
        if (((r<rows) && (r>=0)) && (start>=0) && (start < cols) && (stop >=0) && (stop<cols) && (start<=stop))
          for (int i =r; i<=r;i++)
             for (int j =start; j<=stop;j++)
               if (index[i*cols+j] > max)
               {
                  idx=i*cols+j; 
                  max =  index[i*cols+j];
               }
        return idx;
   };

   Matrix(int r, int c, char *s="") 
   {
       name=string(s);
       index=NULL;
       if ((r>0) && (c>0))
       {
          rows = r;
          cols = c;
          index = new double[r*c];
       }
       else
          cout << "Error: Non-Zero Positive numbers only: Passed row=" << r << " and col=" << c <<endl;
   };
/*
   ~Matrix()
   {
      if (index != NULL)
           delete[] index;
   };
*/
   string prtstr(string s="")
   {
       stringstream ss;
       ss << s << endl;
       if (index !=NULL)
          for (int i=0;i<rows;i++)
          {
             for (int j=0;j < cols;j++)
                  ss << "   " << index[i*cols+j] ;
             ss << endl;
          }
       return ss.str();
   }
   void prt(string s)
   {
       cout << s << endl;
       if (index !=NULL)
          for (int i=0;i<rows;i++)
          {
             for (int j=0;j < cols;j++)
                  cout << "   " << index[i*cols+j] ;
             cout << endl;
          }
   }
   void free_ele()
   {
       if (index != NULL)
           delete[] index;
   };

const Matrix operator+ (int d)
   {

     Matrix tmp(rows,cols,"tmp");
         for (int i = 0; i < rows; ++i)  // m_nc == y_nc
         {
             for (int j = 0; j < cols; ++j)     // m_nr == x_nc
             {
                  tmp.index[i] = index[i*cols+j] + (double)  d;
             }
         }
     if (name=="tmp")
           delete [] index;
     return tmp;
  };

Matrix operator+ (double d)
const    {

     Matrix tmp(rows,cols,"tmp");
         for (int i = 0; i < rows; ++i)  // m_nc == y_nc
         {
             for (int j = 0; j < cols; ++j)     // m_nr == x_nc
             {
                  tmp.index[i] = index[i*cols+j] +  d;
             }
         }
     if (name=="tmp")
           delete [] index;
     return tmp;
  };


const Matrix operator+ (const Matrix  m2)
 const  {

     Matrix tmp(rows,cols,"tmp");
     if ((cols == m2.cols) && (rows==m2.rows))
     {
         for (int i = 0; i < m2.rows; ++i)  // m_nc == y_nc
         {
             for (int j = 0; j < m2.cols; ++j)     // m_nr == x_nc
             {
                  tmp.index[i] = index[i*m2.cols+j] +  m2.index[i*m2.cols+j];
             }
         }
     }
     if (m2.name=="tmp")
           delete [] m2.index;
     return tmp;
  };
Matrix operator+ (Matrix  m2)
   {

     Matrix tmp(rows,cols,"tmp");
     if ((cols == m2.cols) && (rows==m2.rows))
     {
         for (int i = 0; i < m2.rows; ++i)  // m_nc == y_nc
         {
             for (int j = 0; j < m2.cols; ++j)     // m_nr == x_nc
             {
                  tmp.index[i] = index[i*m2.cols+j] +  m2.index[i*m2.cols+j];
             }
         }
     }
     if (m2.name=="tmp")
           delete [] m2.index;
     return tmp;
  };

Matrix operator- (int d)
   {

     Matrix tmp(rows,cols,"tmp");
         for (int i = 0; i < rows; ++i)  // m_nc == y_nc
         {
             for (int j = 0; j < cols; ++j)     // m_nr == x_nc
             {
                  tmp.index[i] = index[i*cols+j] - (double)  d;
             }
         }
     if (name=="tmp")
           delete [] index;
     return tmp;
  };

Matrix operator- (double d)
   {

     Matrix tmp(rows,cols,"tmp");
         for (int i = 0; i < rows; ++i)  // m_nc == y_nc
         {
             for (int j = 0; j < cols; ++j)     // m_nr == x_nc
             {
                  tmp.index[i] = index[i*cols+j] -  d;
             }
         }
     if (name=="tmp")
           delete [] index;
     return tmp;
  };

const Matrix operator* (const int d)
   {

     Matrix tmp(rows,cols,"tmp");
         for (int i = 0; i < rows; ++i)  // m_nc == y_nc
         {
             for (int j = 0; j < cols; ++j)     // m_nr == x_nc
             {
                  tmp.index[i] = index[i*cols+j] * (double) d;
             }
         }
     if (name=="tmp")
           delete [] index;
     return tmp;
  };
const Matrix operator* (const double d)
   {

     Matrix tmp(rows,cols,"tmp");
         for (int i = 0; i < rows; ++i)  // m_nc == y_nc
         {
             for (int j = 0; j < cols; ++j)     // m_nr == x_nc
             {
                  tmp.index[i] = index[i*cols+j] *  d;
             }
         }
     if (name=="tmp")
           delete [] index;
     return tmp;
  };

Matrix operator- (Matrix  m2)
   {

     Matrix tmp(rows,cols,"tmp");
     if ((cols == m2.cols) && (rows==m2.rows))
     {
         for (int i = 0; i < m2.rows; ++i)  // m_nc == y_nc
         {
             for (int j = 0; j < m2.cols; ++j)     // m_nr == x_nc
             {
                  tmp.index[i] = index[i*m2.cols+j] -  m2.index[i*m2.cols+j];
             }
         }
     }
     if (m2.name=="tmp")
           delete [] m2.index;
     return tmp;
  };
//element wise multiply is %
const Matrix operator% (const Matrix  m2)
{

     Matrix tmp(rows,cols,"tmp");
 if ((cols == m2.cols)  && (rows == m2.rows))
 {
     for (int i = 0; i < m2.cols; ++i)     // m_nc == y_nc
     {
          for (int j = 0; j < m2.rows; ++j)        // m_nr == x_nc
          {
               tmp.index[j *m2.cols + i] = m2.index[j *m2.cols + i] *index[j *m2.cols + i];
          }
     }
 }
     if (name == "tmp")
          delete [] index;
     if (m2.name == "tmp")
          delete [] m2.index;
     return tmp;
};


const Matrix operator* (const Matrix m2)
   {

     Matrix tmp(rows,m2.cols,"tmp");
     if (cols == m2.rows)
     {
       for (int i = 0; i < m2.cols; ++i)  // m_nc == y_nc
       {
          tmp.index[i] = 0;
cout << "y["<<i<<"]=0" << endl;
          for (int j = 0; j < m2.rows; ++j)     // m_nr == x_nc
          {
cout << "y["<<i<<"] += M["<<j<<"*"<<m2.cols<<"+"<<i<<"] * x["<<j<<"] ===="<<  m2.index[j *m2.cols + i] << "*" << index[j] << endl;
               tmp.index[i] += m2.index[j *m2.cols + i] *index[j];
          }
       }
     }
     if (m2.name == "tmp")
          delete [] m2.index;
     return tmp;
  };





   Matrix& operator= (const Matrix m2)
   {
    // do the copy
       if (rows==0 && cols==0 && index==NULL)
       {
            index = new double[m2.rows*m2.cols];
            cols = m2.cols;
            rows = m2.rows;
       }
       if ((m2.rows == rows) && (m2.cols == cols))
          for (int r=0;r<m2.rows;r++)
              for (int c=0;c<m2.cols;c++)
                 index[r*m2.cols+c] = m2.index[r*m2.cols+c];
       if (m2.name=="tmp")
          delete [] m2.index;
       return *this;
   };

};

std::time_t result = std::time(nullptr);
string fid = to_string(result);
unsigned int NumberOfLayers;
unsigned int OutputLayer;
unsigned int *nodes;
double eta;	// Learning factor

vector<Matrix> netin;
vector<Matrix> actuation;
vector<Matrix> deltafn;
vector<Matrix> ftick;
vector<Matrix> layer_weights;
vector<Matrix> weight_updates;
vector<Matrix> new_layer_weights;


ios init(NULL);
stringstream confusion_matrix;
Matrix err_summary(1,OUTPUT_LINES);


#ifdef WANT_TO_LOAD_WEIGHTS
// Used for loading weights from file (if ever required)
double l2[10][50000];
int nd[100];
int nd2[100];
int lays;
int t = 0;
int x = 0;
#endif

#ifdef SERIAL_ONLY
string build_type = "Serial";
#else
string build_type = "Parallel";
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
__global__ 
void VectorMatrixMultiply(double* act, double* lwgts, double* net, int actuation_rows, int actuation_cols, int layer_weights_rows, int layer_weights_cols, int netin_rows, int netin_cols, int tile_dimension) 
{

    double netin_accum = 0;

    int row = blockIdx.y*tile_dimension + threadIdx.y;
    int col = blockIdx.x*tile_dimension + threadIdx.x;

    for (int i = 0; i < (tile_dimension + actuation_cols - 1)/tile_dimension; i++) 
    {
        for (int j = 0; j < tile_dimension; ++j) 
            if ((i*tile_dimension + j < actuation_cols && row < actuation_rows) && (i*tile_dimension + j < layer_weights_rows && col < layer_weights_cols))
                netin_accum += act[row*actuation_cols + i*tile_dimension + j] * lwgts[(i*tile_dimension + j)*layer_weights_cols + col];
    }

    if (row < netin_rows && col < netin_cols) 
    //      net[((blockIdx.y * blockDim.y + threadIdx.y)*netin_cols)+(blockIdx.x*blockDim.x)+threadIdx.x]=netin_accum;
       net[((blockIdx.y * blockDim.y + threadIdx.y)*netin_cols)+(blockIdx.x*blockDim.x)+threadIdx.x]= 5;
}

#ifndef SERIAL_ONLY
int InitiateCUDAVectorMatrixMultiply3(int i) 
{

    float time;
    int x_dimension = 1;
    // CUDA Grid is based on Matrix Dimensions
    int y_dimension = layer_weights[i].rows;
    int z_dimension = layer_weights[i].cols;

    int netin_rows = x_dimension;
    int netin_cols = z_dimension;

    int actuation_rows = x_dimension;
    int actuation_cols = y_dimension;

    int layer_weights_rows = y_dimension;
    int layer_weights_cols = z_dimension;

    dim3 dimBlock(tile_dimension, tile_dimension, 1);
    dim3 dimGrid;

cout <<    "i=" << i << " dimGrid.x = ("<<netin_cols<<" + "<<dimBlock.x <<"- 1)/ "<<dimBlock.x<<endl;
cout <<    "i=" << i << " dimGrid.y = ("<<netin_rows<<" + "<<dimBlock.y <<"- 1)/ "<<dimBlock.y<<endl;

    dimGrid.x = (netin_cols + dimBlock.x - 1)/dimBlock.x;
    dimGrid.y = (netin_rows + dimBlock.y - 1)/dimBlock.y;
    
    checkError(cudaMemcpy(ActuationDevice, actuation[i].index, x_dimension*y_dimension*sizeof(double), cudaMemcpyHostToDevice));
    checkError(cudaMemcpy(LayerWeightsDevice,  layer_weights[i].index, y_dimension*z_dimension*sizeof(double), cudaMemcpyHostToDevice));

//    cudaEventRecord(start,0);
    auto StartChronoTime = std::chrono::high_resolution_clock::now();
cout << dimGrid.x << "," << dimGrid.y << "==Grid, Block==" << dimBlock.x << "," << dimBlock.y << " and tile_dimension="<< tile_dimension << endl;
    // CUDA Call to GPU /////////////////////////////////////////////////////////
    VectorMatrixMultiply<<<dimGrid, dimBlock>>>(ActuationDevice, LayerWeightsDevice, NetinDevice, actuation_rows, actuation_cols, layer_weights_cols, layer_weights_rows, netin_rows, netin_cols, tile_dimension);
    // CUDA Call to GPU /////////////////////////////////////////////////////////

    checkError(cudaDeviceSynchronize());

    auto EndChronoTime = std::chrono::high_resolution_clock::now();
    auto TotalChronoTime = std::chrono::duration_cast<std::chrono::microseconds > (          EndChronoTime - StartChronoTime);

    if (TotalChronoTime > Process_MaxTime)
       Process_MaxTime = TotalChronoTime;

    if (TotalChronoTime < Process_MinTime)
       Process_MinTime = TotalChronoTime;


//    cudaEventRecord(stop,0);
//    cudaEventSynchronize(stop);
//    cudaEventElapsedTime(&time, start, stop);

    if (time < mintime)
       mintime = time;
    if (time > maxtime)
       maxtime = time;
    checkError(cudaMemcpy(netin[i].index, NetinDevice, x_dimension*z_dimension*sizeof(double), cudaMemcpyDeviceToHost));
    for (int j=0;j<netin[i].cols;j++)
       netin[i].index[j] = netin[i].index[j] / actuation[i].cols;
    
    return 0;
}
#endif

void MultArmVM(double *V, double *M, double *R, int m_nr, int m_nc)
{
     double sum;
     for (int c = 0; c < m_nc; c++)
     {
          sum = 0;
          for (int r = 0; r < m_nr; r++)	// m_nr == v_nc &m_nc=r_nc
               sum += M[c *m_nr + r] *V[r];
          R[c] = sum;
     }
}

// implementation of the matrix-vector multiply function
void MatrixTranspVectorMultiply2(double *Y, const double *X, double *M,
          int m_nr, int m_nc)
{
     int t_r = m_nc;	// same as x_nc
     int t_c = m_nr;	// same as y_nc
    	// matrix passed in is m_nr x m_nc, need to transpose it to m_nr x m_nc
    	// and its stored in as columns so can maniuplate indexes
     for (int i = 0; i < t_r; ++i)
     {
         	// cout << "i="<<i<<" m_nc="<<m_nr<<" m_nr="<<m_nr<<endl;
          Y[i] = 0;
          int t = 0;
          for (int j = i; j < t_c * t_r; j += t_c)
          {

               Y[i] += M[i *t_c + j] *X[t++];
          }
     }
}
// implementation of the matrix-vector multiply function
void MatrixTranspVectorMultiply(double *Y, const double *X, double *M, int m_nr,
          int m_nc)
{
     for (int i = 0; i < m_nr; ++i)	// m_nc == y_nc
     {
          Y[i] = 0;
          for (int j = 0; j < m_nc; ++j)	// m_nr == x_nc
          {
               Y[i] += M[i *m_nc + j] *X[j];
          }
     }
}
// implementation of the matrix-vector multiply function
void SerialMatrixVectorMultiply(double *Y, double *X, double *M, int m_nr, int m_nc)
{
    auto StartChronoTime = std::chrono::high_resolution_clock::now();

  // Need to ensure Y vector passed has been zeroised
    for (int i=0;i<m_nr*m_nc ;i++)
    {
        int c1=i % m_nc;
        int r1=i / m_nc;
        Y[c1] += X[r1] * M[c1 *m_nr + r1];
    }
    auto EndChronoTime = std::chrono::high_resolution_clock::now();
    auto TotalChronoTime = std::chrono::duration_cast<std::chrono::microseconds > (          EndChronoTime - StartChronoTime);

    if (TotalChronoTime > Process_MaxTime)
       Process_MaxTime = TotalChronoTime;

    if (TotalChronoTime < Process_MinTime)
       Process_MinTime = TotalChronoTime;
}

void sigmoid3(Matrix & net, Matrix & out)
{
   int c=net.cols;
   for (int i=0;i<c;i++)
     out.index[i] = 1 / (1 + exp(-net.index[i]));
   out.index[c] = 1.0;	// add bias signal value
     //return out;
}

/////////////////////////////////////////////
//
// PRINT ROUTINES
//
void print_an_image_vals(unsigned char *c, int i)
{
     cout << "This is a : " << i << endl << flush;
     for (int i = 0; i < INPUT_LINES; i++)
     {
          if (i % MATRIX_SIDE == 0)
               cout << endl << flush;
          cout << hex << std::setfill('0') << std::setw(2) << (unsigned int) c[i] <<
               dec << " ";
     }
     cout << endl << flush;
}

void print_an_image(unsigned char *c, int i)
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

void print_images(unsigned char *c, int size)
{
     for (int i = IMAGE_OFFSET; i < size; i++)
     {
          if (((i - IMAGE_OFFSET) % MATRIX_SIDE) == 0)
               cout << endl << flush;
          if (((i - IMAGE_OFFSET) % INPUT_LINES) == 0)
               cout << endl << "Image : " << dec <<
               ((i - IMAGE_OFFSET) / INPUT_LINES) + 1 << endl << flush;
          cout << hex << std::setfill('0') << std::setw(2) << (unsigned int) c[i] <<
               " ";
     }
}

//
//
/////////////////////////////////////////////

unsigned char *load_file(string filename, string labels, unsigned char **labs)
{
     unsigned char *memblock;
     ifstream inFile;
     streampos size;

     cout << "Using file '" << filename << "'" << endl << flush;
    	//
    	// Load MNIST DIGIT IMAGES
    	//
     inFile.open(filename, ios:: in | ios::binary | ios::ate);
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
          inFile.read((char*) memblock, size);
          inFile.close();

          cout << "the entire file content is in memory, all " << size <<
               " bytes of it" << endl << flush;
     }
     inFile.close();
    	//
    	// Load MNIST DIGIT LABELS
    	//
     inFile.open(labels, ios:: in | ios::binary | ios::ate);
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
          inFile.read((char*) *labs, size);
          inFile.close();

          cout << "the entire file content is in memory, all " << size <<
               " bytes of it" << endl << flush;
     }
     inFile.close();
     return memblock;
}

void load_an_image(int seq, unsigned char* &mptr, Matrix &img, Matrix &t,
     unsigned char* &lp)
{
     int start = (INPUT_LINES *seq) + IMAGE_OFFSET;
     double greyval = MAX_PIXEL_VAL;

     for (int i = 0; i < INPUT_LINES; i++)
     {
          img.index[i] = ((double) mptr[start + i]) / greyval;
     }

     img.index[nodes[0]] = 1;      // set bias signal, so can multiply with[node weights |
        // bias weights] augmented matrix

     int img_is_digit = (int) lp[8 + seq];
#ifdef SAMPLEFREQ
     if ((seq + 1) % SAMPLEFREQ == 0)
     {
          cout << "For sample :" << seq + 1 << endl << flush;
          print_an_image(&mptr[start], img_is_digit);
     }
#endif
     t.zeroize();  // create the target vector (plus one for 'bias' bit)
     if (img_is_digit > 9)
     {
          cout << "Error: img_is_digit=" << img_is_digit << "seq=" << seq << endl;
          exit(1);
     }
     t.index[img_is_digit] = 1;    // set the target 'bit'
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
void output(Matrix t)
{
     t.prt("Test");
}
     
double accu(Matrix m1)
{
  double tmp=0;
    for (int i=0;i<m1.rows;i++)
      for (int j=0; i<m1.cols;j++)
         tmp += m1.index[i*m1.cols+j];
  return tmp;
}    
int backprop(Matrix & tgt, int y0)
{

     Matrix final = actuation[OutputLayer];
     final.cols--;
     Matrix tgt0 = tgt;
     //tgt0.insert_cols(nodes[OutputLayer], 1);
     double err = accu((tgt - final) % (tgt - final)) *0.5;
     if (abs(err) < EPSILON)
     {
          int val = tgt0.index_max_row(0,0,9);
#ifdef SAMPLEFREQ
          if ((y0 + 1) % SAMPLEFREQ == 0)
               cout << "---------------------------------- BACK PROPAGATION  sample=" <<
               y0 + 1 << " err=" << err << "<epsilon, for tgt '" << val <<
               "' so error is acceptable, returning" << endl << flush;
#endif
          err_summary.index[val] = err;
          return 1;
     }

#ifdef SAMPLEFREQ
     if ((y0 + 1) % SAMPLEFREQ == 0)
          cout << "------------------------------------ BACK PROPAGATION sample=" <<
          y0 + 1 << endl << flush;
#endif
     ftick[OutputLayer] = (actuation[OutputLayer]* (-1))  + 1;
     ftick[OutputLayer] =
          ftick[OutputLayer] % (actuation[OutputLayer]);	// element wise multiply
     deltafn[OutputLayer] = (tgt0 - actuation[OutputLayer]) % (ftick[OutputLayer]);

     for (int i = OutputLayer - 1; i >= 0; i--)
     {
          weight_updates[i] =  (deltafn[i + 1]*actuation[i]);
          new_layer_weights[i]  = layer_weights[i]*weight_updates[i]  + (weight_updates[i]*eta );
          ftick[i] = (actuation[i] *(-1)) + 1;
          ftick[i] = ftick[i] % (actuation[i]);	// element wise multiply
          deltafn[i] = deltafn[i + 1] *layer_weights[i];
          deltafn[i] = deltafn[i] % ftick[i];
     }
     for (int i = 0; i < OutputLayer; i++)
     {

          layer_weights[i] = new_layer_weights[i];
     }
     return 0;
}


void forward_feed(unsigned char* &imgdata, unsigned char* &labdata, bool train,
     int samples)
{
     Matrix tgt(1,OUTPUT_LINES+1);
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
     string intype;
     if (train)
     {
          intype = "TRAINING";
          epochs = EPOCHS;
     }
     else
     {
          epochs = 1;
          intype = "TEST    ";
     }
     for (int y = 0; y < samples; y++)
     {
#ifdef SAMPLEFREQ
          if ((y + 1) % SAMPLEFREQ == 0)
               cout << "------------------------------------ FORWARD FEED OF " << intype <<
               " SAMPLE # " << y + 1 << endl << flush;
#endif
          load_an_image(y, imgdata, actuation[0], tgt, labdata);

          int tgtval = tgt.index_max_row(0, 0, 9);
          for (int e = 0; e < epochs; e++)
          {
               for (int i = 0; i < OutputLayer; i++)	// only n-1 transitions between n layers
               {
#ifdef SERIAL_ONLY
                   SerialMatrixVectorMultiply(netin[i].index, 
                                               actuation[i].index, 
                                               layer_weights[i].index,  layer_weights[i].rows,  layer_weights[i].cols);


                    for (int j = 0; j < netin[i].cols; j++)
                    {
                        	      netin[i].index[j] /= actuation[i].cols;
                    }
#ifdef SAMPLEFREQ
                    if ((y + 1) % SAMPLEFREQ == 0)
                         cout << "Netin serial (" << netin[i].rows << "," << netin[i].cols <<
                         ")= " << netin[i].prtstr("NetIn Serial") << endl << flush;
#endif
#else
                   InitiateCUDAVectorMatrixMultiply3(i);

#ifdef SAMPLEFREQ
                         cout << "Netin Parallel " << netin[i].rows << "," << netin[i].cols <<
                         ")= " << netin[i].prtstr("Net In Parallel") << endl << flush;
#endif

#endif
                    sigmoid3(netin[i], actuation[i + 1]);
               }
#ifdef SAMPLEFREQ
               if ((y + 1) % SAMPLEFREQ == 0)
               {
                    std::cout << "Final output : " << endl << std::setw(7) << fixed <<
                         showpoint << actuation[OutputLayer].prtstr("Final Out") <<
                         " Sample: " << y + 1 << std::endl << flush;
                    std::cout << "Expec output : " << endl << std::setw(7) << fixed <<
                         showpoint << tgt.prtstr("Tgt3") << " Sample: " << y + 1 <<
                         std::endl << flush;
               }
#endif
              	//////////////////////////// forward feed end
               if (train)
               {
                   	// printout intermediate result
               //     int outval = actuation[OutputLayer].subvec(0, 9).index_max();
                    int outval =  actuation[OutputLayer].index_max_row(0, 0, 9);
#ifdef SAMPLEFREQ
                    if ((y + 1) % SAMPLEFREQ == 0)
                    {
                         std::cout << "Train output : " << endl << std::setw(7) << fixed <<
                              showpoint << actuation[OutputLayer].prtstr("") <<
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
                    if (backprop(tgt, y) == 1)
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
                    showpoint << actuation[OutputLayer].prtstr("B") <<
                    " Sample: " << y + 1 << std::endl << flush;
               for (int z1 = 0; z1 < actuation[OutputLayer].index_max_row(0, 0, 9); z1++)
                    cout << "         ";
               cout << "       ^" << endl << flush;
               std::cout << "Expec output : " << endl << std::setw(7) << fixed <<
                    showpoint << tgt.prtstr("") << " Sample: " << y + 1 <<
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
                    pctg = (float)(rowsum[i]) / (float)(tot_wrong) *100.0f;
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
                    pctg = (float)(colsum[i]) / (float)(tot_wrong) *100.0f;
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
               (float)(tot_correct) / (float)(tot_correct + tot_wrong) *100.0f;
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
     iFile.open(fname, ios:: in);
     string aline;

     vector<string> strs;

     if (fname.substr(0, 4) == "post")
          stringstream confusion_matrix2;
     getline(iFile, aline);
     boost::split(strs, aline, boost::is_any_of("="));
     if (strs.size() > 1)
          lays = stoi(strs[1]);
     cout << " Has " << lays << "layers" << endl;
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
               cout << " Has " << nd[t] << "layers" << endl;
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
               layer_weights[i](r, c) = l2[i + 1][j];
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
     oFile << "Error Summary" << endl << flush;

     oFile << err_summary.prtstr() << endl << flush;

     oFile << "EndFile" << endl << flush;
     oFile.close();
     cout.copyfmt(init);

}

int main(int argc, char *argv[])
{
     extern char **environ;
     string hname = "";
     //string y="initial_random_values_weights_11337071.txt";
     string y = "initial_random_values_weights_1636260202.txt";

     vector<string> strs;
     string bldver = string(__DATE__) + " at time " + string(__TIME__);
     cout << "--------------------------------  Build done on " << bldver << endl <<
          flush;
     init.copyfmt(cout);
     for (int i=0;i<err_summary.cols;i++)
        err_summary.index[i] = -1.0;
     if (argc < 2)
     {
          NumberOfLayers = 3;
          nodes = new unsigned int[NumberOfLayers];
          nodes[0] = INPUT_LINES;
          nodes[1] = DEFAULT_HIDDEN;
          nodes[2] = OUTPUT_LINES;
          eta = ETA_DEFAULT;
          cout << "Using default setting of \"" << nodes[0] << " " << nodes[1] << " " <<
               nodes[2] << "\" " << endl << flush;
          cout << "And ETA=" << eta << endl << flush;;
     }
     else if (argc < 6)
     {
          cout << "Usage: " << argv[0] << " ETA IN H1[H2 H3 ...] OUT THREADS" << endl <<
               flush;
          cout << "       Where ETA is the learning factor, &" << endl << flush;
          cout
               <<
               "       Where number of parameters after ETA is the number of layers" <<
               endl << flush;
          cout << "       Must have a minimum of 3, i.e. IN H1 OUT" << endl << flush;
          cout << "       And the parameters themselves are numbers, " << endl <<
               flush;
          cout << "       indicating the number of nodes in that layer." << endl <<
               flush;
          cout << "       e.g. \"" << argv[0] << " " << ETA_DEFAULT << " " <<
               INPUT_LINES << " " << DEFAULT_HIDDEN << " " << OUTPUT_LINES << " " <<
               DEFTHREADS << "\" " << endl << flush;
          cout << "       and is the default, if no params supplied." << endl <<
               flush;
          exit(1);
     }
     else
     {
          NumberOfLayers = argc - 3;
          nodes = new unsigned int[NumberOfLayers];
          eta = stod(string(argv[1]));
          if (eta <= 0)
          {
               cout << "Error: ETA must be positive, usually less than 1" << endl <<
                    flush;
               exit(1);
          }
          for (int i = 2; i < argc - 1; i++)
          {
               int p = stoi(string(argv[i]));
               if (p > 0)
               {
                    nodes[i - 2] = stoi(string(argv[i]));
               }
               else
               {
                    cout << "Error in parameter " << i << " - must be positive" << endl <<
                         flush;
                    exit(1);
               }
          }
          thrds = stoi(argv[argc - 1]);
     }
     cout << "Threads chosen is " << thrds << endl << flush;
     cout << "Number of Layers is " << NumberOfLayers << endl << flush;

    	// netptrs = new double *[NumberOfLayers];
    	// Use slurm job number if avaiable (else defaults to epoch time) for file ids
    	// created
     for (char **current = environ; *current; current++)
     {
          string tmp = *current;
          boost::split(strs, tmp, boost::is_any_of("="));
          if ((strs[0] == "SLURM_JOBID") || (strs[0] == "SLURM_JOB_ID"))
          {
               if (strs[1].length() > 0)
               {
                    fid = strs[1];
               }
          }
          else if (strs[0] == "HOSTNAME")
          {
               if (strs[1].length() > 0)
               {
                    hname = strs[1];
               }
          }
     }

#ifndef SERIAL_ONLY
 // set up CUDA timing structs
     cudaEventCreate(&start);
     cudaEventCreate(&stop);
#endif

     OutputLayer = NumberOfLayers - 1;
     unsigned char *trainlabels;
     unsigned char *testlabels;
     unsigned char *traindata = load_file("train-images-idx3-ubyte",
          "train-labels-idx1-ubyte", &trainlabels);
     unsigned char *testdata = load_file("t10k-images-idx3-ubyte",
          "t10k-labels-idx1-ubyte", &testlabels);
     auto StartTime = std::chrono::high_resolution_clock::now();

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
          Matrix rb3a(1, nodes[i] + bias_field);
          actuation.push_back(rb3a);

          Matrix drb3(1, nodes[i] + bias_field);
          deltafn.push_back(drb3);

          Matrix frb3(1, nodes[i] + bias_field);
          ftick.push_back(frb3);

          if (i < OutputLayer)
          {
               max_mat =
                    max(max_mat, (nodes[i] + bias_field) *(nodes[i + 1] + bias_field));
               // These buffers for the rowvec and mat structures below are done to ensure
               // the Armadillo matrix can be accessed directly and the library doesnt move
               // the memory around
               Matrix rb3(1,( nodes[i+1] + bias_field));

               // Create an array of matrices (one element for each layer) for the netin value
               // This holds the sum of weighted signals, for each node, that gets squashed to 
               // produce the nodes output for next layer
               netin.push_back(rb3);
               // Create a buffer of required size for weights, in each layer
               // (plus two more, one for delta updates, and one for holding new weight to be
               // applied after backprop. These maybe consolidated later
               Matrix tmpwgt3((nodes[i + 1] + bias_field),( nodes[i] + bias_field));
               for (int p=0;p<(nodes[i + 1] + bias_field)*( nodes[i] + bias_field);p++)
                  tmpwgt3.index[p] = rand()/RAND_MAX;
               Matrix tmpwgt30((nodes[i + 1] + bias_field),( nodes[i] + bias_field));
               Matrix tmpwgt300((nodes[i + 1] + bias_field),( nodes[i] + bias_field));
               // create an array of three matrices (weights for forward prop)
               // and deltas and new values, for back propagation
               layer_weights.push_back(tmpwgt3);

               new_layer_weights.push_back(tmpwgt30);

               weight_updates.push_back(tmpwgt300);
          }
     }
     // Save initial starting weights if required for later
     save_weights("initial_random_values");

    // Informational, the max value of matrix and vectors are record and used to reserve CUDA memory 
     cout << "Max Matrix size " << max_mat << " Max vector size = " << max_vec <<
          endl << flush;

#ifdef WANT_TO_LOAD_WEIGHTS
     // this is a function to load previously saved weights, to either ensure constant initial values
     // if say moving platforms with different psudeo RNG, or to load post weights after training
     // This works, but only implemented atm, by direct code changes, no UI implemented
     // But note used in this project anyway
     load_weights(y);
#endif

#ifndef SERIAL_ONLY
     checkError(cudaMalloc(&ActuationDevice, max_vec* sizeof(double)));
     checkError(cudaMalloc(&NetinDevice, max_vec* sizeof(double)));
     checkError(cudaMalloc(&LayerWeightsDevice, max_mat* sizeof(double)));
#ifdef __CUDA_ARCH__
     cout << "Built for CUDA ARCH == " << __CUDA_ARCH__ << endl;
#endif
#endif
    	///////////////////////////////////////////////
    	//
    	// TRAIN THE DATA
    	//
     auto StartTrainTime = std::chrono::high_resolution_clock::now();
     cout << "Training on data started (epochs=" << EPOCHS << ")...." << endl <<
          flush;

     forward_feed(traindata, trainlabels, true, TRAININGSAMPLES);
     auto EndTrainTime = std::chrono::high_resolution_clock::now();

     cout << "Training complete" << endl << flush;
    	///////////////////////////////////////////////
    	//
    	// TEST THE DATA
    	//
     cout << "Testing of data started...." << endl << flush;
     auto StartTestTime = std::chrono::high_resolution_clock::now();

     forward_feed(testdata, testlabels, false, TESTINGSAMPLES);

     auto EndTestTime = std::chrono::high_resolution_clock::now();

     cout << "Testing complete" << endl << flush;

     auto TotalTime = std::chrono::duration_cast<std::chrono::microseconds > (          EndTestTime - StartTime);
     auto TrainTime = std::chrono::duration_cast<std::chrono::microseconds > (          EndTrainTime - StartTrainTime);
     auto TestTime = std::chrono::duration_cast<std::chrono::microseconds > (          EndTestTime - StartTestTime);

     cout << "Total Time       : " << std::setw(12) << TotalTime.count() << " us" <<
          endl << flush;
     cout << "Total Train Time : " << std::setw(12) << TrainTime.count() << " us" <<
          endl << flush;
     cout << "Total Test Time  : " << std::setw(12) << TestTime.count() << " us" <<
          endl << flush;

     confusion_matrix << "Epochs in Training : " << EPOCHS << endl << flush;
     confusion_matrix << "Training Samples   : " << TRAININGSAMPLES << endl <<
          flush;
     confusion_matrix << "Testing Samples    : " << TESTINGSAMPLES << endl <<
          flush;
     confusion_matrix << endl << endl << "Total Time       : " << std::setw(12) <<
          TotalTime.count() << " us" << endl << flush;
     confusion_matrix << "Total Train Time : " << std::setw(12) <<
          TrainTime.count() << " us" << endl << flush;
     confusion_matrix << "Total Test Time  : " << std::setw(12) << TestTime.count() <<
          " us" << endl << flush;
     confusion_matrix << endl << endl << "Total Time       : " << std::setw(12) <<
          TotalTime.count() / 1000000 << " s" << endl << flush;
     confusion_matrix << "Total Train Time : " << std::setw(12) <<
          TrainTime.count() / 1000000 << " s" << endl << flush;
     confusion_matrix << "Total Test Time  : " << std::setw(12) <<
          TestTime.count() / 1000000 << " s" << endl << flush;
     confusion_matrix << endl << endl << "Total Time       : " << std::setw(12) <<
          TotalTime.count() / 60000000 << " min" << endl << flush;
     confusion_matrix << "Total Train Time : " << std::setw(12) <<
          TrainTime.count() / 60000000 << " min" << endl << flush;
     confusion_matrix << "Total Test Time  : " << std::setw(12) <<
          TestTime.count() / 60000000 << " min" << endl << flush;
     confusion_matrix << "Epsilon  : " << EPSILON << endl << flush;
     confusion_matrix << "Eta      : " << eta << endl << flush;
     confusion_matrix << "Build ver: " << bldver << endl << flush;
     save_weights("post_training_weights");

     delete[] traindata;
     delete[] trainlabels;
     delete[] testdata;
     delete[] testlabels;

     if (mintime == std::numeric_limits<float>::max())
        cout << "Problem recording min time for " << build_type << endl;
     else
     {
        cout << "Min time for " <<  build_type << " call : " << mintime  << " us" << endl;
        cout << "Min time for " <<  build_type << " call : " << mintime/1000000  << " s" << endl;
        cout << "Min time for " <<  build_type << " call : " << mintime/60000000  << " min" << endl;
     }
     if (maxtime == std::numeric_limits<float>::min())
        cout << "Problem recording max time for " << build_type << endl;
     else
     {
        cout << "Max time for " <<  build_type << " call : " << mintime  << " us" << endl;
        cout << "Max time for " <<  build_type << " call : " << mintime/1000000  << " s" << endl;
        cout << "Max time for " <<  build_type << " call : " << mintime/60000000  << " min" << endl;
     }
#ifndef SERIAL_ONLY
     cout << "Max time for CUDA call : " << Process_MaxTime.count() << " us (Measured by CUDA API)" << endl;
     cout << "Min time for CUDA call : " << Process_MinTime.count() << " us (Measured by CUDA API)" << endl;
     cout << "Used Tile Dimension of " << tile_dimension << endl;

     for (int i=0;i<netin.size();i++)
         netin[i].free_ele();


     checkError(cudaFree(LayerWeightsDevice));
     checkError(cudaFree(ActuationDevice));
     checkError(cudaFree(NetinDevice));

     checkError(cudaEventDestroy(start));
     checkError(cudaEventDestroy(stop));
#endif


}
