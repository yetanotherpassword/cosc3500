#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <boost/algorithm/string.hpp>
#include <vector>

#undef DEBUGON
#define DEFTHREADS 256
#define ARMA_64BIT_WORD
#define INPUT_LINES 784
#define OUTPUT_LINES 10
#define MATRIX_SIDE 28
#define MAX_PIXEL_VAL 255.0f
#define IMAGE_OFFSET 16
#define DEFAULT_HIDDEN 30
#define ETA_DEFAULT 0.5f

#define SAMPLEFREQ 1
//#undef SAMPLEFREQ

#define EPOCHS 1
#define EPSILON 1E-04
#define TRAININGSAMPLES 60000
#define TESTINGSAMPLES 10000
#define BLOCK_HEIGHT 1024
#define BLOCK_WIDTH 64
#define SPACE_GAP "   "
#define ONLYIF if (false)
/*
 * ALLAN CAMPTON
 * COSC3500 Milestone 1 Serial Version
 *
 * To perform a full build and run from scratch, do the following
 *
 *    git clone git://github.com/yetanotherpassword/cosc3500
 *    cd ~/cosc3500/
 *    unzip mnist.zip
 *    unxz armadillo-10.6.2.tar.xz
 *    tar xvf armadillo-10.6.2.tar
 *    cd armadillo-10.6.2/
 *    	#Made lib static and issue with MKL on Centos
 *    	#Below changes done in my git, but may need to do if download from
 *    	#http://sourceforge.net/projects/arma/files/armadillo-10.6.2.tar.xz
 *    	#sed -i "s/add_library( armadillo/add_library( armadillo STATIC/" CMakeLists.txt
 *    	#sed -i "s/include(ARMA_FindMKL)/#include(ARMA_FindMKL)/" CMakeLists.txt
 *    mkdir build
 *    cd build
 *    cmake ..
 *    make
 *    cd ../..
 *    make
 *    sbatch ./goslurm.sh ann_mnist_digits
 */
// g++ armo.cpp -g -o armo -std=c++11 -O2 -larmadillo

// requires armodillo (see above)

    using namespace std;
    int thrds=DEFTHREADS; 

    double * LayerWeightsDevice;
    double * ActuationDevice;
    double * NetinDevice;
  double * dev_A;
  double * dev_in;
  double * dev_out;
float mintime=1000000;
float maxtime=-10;

    std::time_t result = std::time(nullptr);
    string fid = to_string(result);
    unsigned int NumberOfLayers;
    unsigned int OutputLayer;
    unsigned int * nodes;
    double eta;               // Learning factor
    typedef struct vect
    {
       double * v;
       int size;
    } vect;
    typedef struct mat
    {
       double * m;
       int rows;
       int cols;
    } matx;
    std::vector<vect> netin;
    vector<vect> actuation;
    vector<vect> deltafn;
    vector<vect> ftick;
    vector<matx> layer_weights;
    vector<matx> weight_updates;
    vector<matx> new_layer_weights;
    vect tgt, tgt0;
    ios init(NULL);
    stringstream confusion_matrix;
    vect err_summary;

 void MultArmVM(double * V, double * M, double * R, int m_nr, int m_nc)
 {
  double sum;
  for (int c=0; c < m_nc; c++)
  {
    sum=0;
    for (int r = 0; r < m_nr; r++)  // m_nr == v_nc & m_nc=r_nc
       sum += M[c*m_nr+r] * V[r];
    R[c] = sum;
  }
 }

void sigmoid( vect  & net, vect & actplus)
{
    if (net.size != actplus.size)
    {
        cout << "Error net.size=="<<net.size<< " doesnt match actplus.size=="<<actplus.size<<endl;
        exit(1);
    }
    for (int i=0;i<net.size-1;i++)
    {
        actplus.v[i] = 1/(1+exp(-net.v[i])); 
        ONLYIF cout << "1/1(+exp(-" << net.v[i] << "))" << endl;
    }
    actplus.v[net.size-1] = 1.0;          // add bias signal value
}

/////////////////////////////////////////////
//
// DEBUGGING ROUTINES
//
void print_an_image_vals(unsigned char * c, int i)
{
    cout << "This is a : " << i << endl << flush;
    for (int i=0;i<INPUT_LINES;i++)
    {
       if (i%MATRIX_SIDE==0)
         cout << endl << flush;
       cout  << hex << std::setfill('0') << std::setw(2) << (unsigned int)c[i] << dec << " ";
    }
    cout << endl << flush;
}
   
void print_an_image(unsigned char * c, int i)
{
    cout << "This is a : " << i << endl << flush;
    for (int i=0;i<INPUT_LINES;i++)
    {
       if (i%MATRIX_SIDE==0)
         cout << endl << flush;
       if (c[i]==0)
          cout  << "  ";
       else if (c[i]<128)
          cout << "xx";
       else
          cout << "XX";
    }
    cout << endl << flush;
}
   

void print_images(unsigned char * c,  int size)
{
    for (int i=IMAGE_OFFSET;i<size;i++)
    {
       if (((i-IMAGE_OFFSET)%MATRIX_SIDE)==0)
           cout << endl << flush;
       if (((i-IMAGE_OFFSET)%INPUT_LINES)==0)
           cout << endl << "Image : " << dec << ((i-IMAGE_OFFSET)/INPUT_LINES)+1 << endl << flush;
       cout << hex << std::setfill('0') << std::setw(2) << (unsigned int)c[i] << " ";
    }
}

//
//
/////////////////////////////////////////////

unsigned char * load_file(string filename, string labels, unsigned char * * labs)
{
    unsigned char * memblock;
    ifstream inFile;
    streampos size;

    cout << "Using file '" << filename << "'" << endl << flush;
//
// Load MNIST DIGIT IMAGES
//
    inFile.open(filename, ios::in|ios::binary|ios::ate);
    if (!inFile) {
        cout << "Unable to open file '" << filename << "'" << endl << flush;
        exit(1); // terminate with error
    }
    else
    {
       cout << "OK opened '" << filename << "' Successfully" << endl << flush;
    }

    if (inFile.is_open())
    {
        size = inFile.tellg();
        memblock = new unsigned char [size];
        inFile.seekg (0, ios::beg);
        inFile.read ((char *)memblock, size);
        inFile.close();

        cout << "the entire file content is in memory, all " << size << " bytes of it" << endl << flush;
    }
    inFile.close();
//
// Load MNIST DIGIT LABELS
//
    inFile.open(labels, ios::in|ios::binary|ios::ate);
    if (!inFile) {
        cout << "Unable to open file '" << labels << "'" << endl << flush;
        exit(1); // terminate with error
    }
    else
    {
       cout << "OK opened '" << labels << "' Successfully" << endl << flush;
    }

    if (inFile.is_open())
    {
        size = inFile.tellg();
        *labs = new unsigned char [size];
        inFile.seekg (0, ios::beg);
        inFile.read ((char *) *labs, size);
        inFile.close();

        cout << "the entire file content is in memory, all " << size << " bytes of it" << endl << flush;
    }
    inFile.close();
    return memblock;

}

void load_an_image(int seq, unsigned char * &mptr, vect & img, vect & t, unsigned char * &lp)
{
    int start=(INPUT_LINES*seq)+IMAGE_OFFSET;
    double greyval=MAX_PIXEL_VAL;

    for (int i=0;i<INPUT_LINES;i++)
    {
        img.v[i] = ((double ) mptr[start+i])/greyval;
    }

    img.v[nodes[0]]=1;          // set bias signal, so can multiply with [node weights | bias weights] augmented matrix

    int img_is_digit=(int) lp[8+seq];
#ifdef SAMPLEFREQ
    if ((seq+1) % SAMPLEFREQ ==0)
    {
       cout << "For sample :" << seq+1 << endl << flush;
       print_an_image(&mptr[start], img_is_digit);
    }
#endif
    for (int i=0;i<t.size;i++)
       t.v[i]=0;

    if (img_is_digit>9)
    {
       cout << "Error: img_is_digit=" << img_is_digit << "seq=" << seq  << endl;
       exit(1);
    }

    t.v[img_is_digit]=1;               // set the target 'bit'

}

int idxmatchval(vect q, int i)
{
   int idx=-1;
   for (int w=0;w<q.size;w++)
      if (q.v[w] == i)
      {
           idx=w;
           break;
      }
   if (idx == -1)
   {
       cout << "Error: No match !!!" << endl;
       exit(1);
   }
   return idx;
}

int idxmaxval(vect & q, int l)
{
   int len=q.size;
   len = len > 10?10:len;
   double val=-100000;
   int idx=-1;
   for (int w=0;w<len;w++)
      if (q.v[w] > val)
      {
           idx=w;
           val = q.v[w];
      }
   if (idx == -1)
   {
       cout << "Error: No maximum !!! Called from line "<< l << " and len=" << len<< endl;
       for (int w=0;w<len;w++)
           cout << q.v[w] << SPACE_GAP;
       cout << endl;
          
       exit(1);
   }
 cout << "returning idx=" << idx << " as "<< q.v[idx] << " is biggest eg " << q.v[0] <<"," <<q.v[1] <<"," <<q.v[2]<<"," <<q.v[3]<<"," <<q.v[4]<<"," <<q.v[5]<<"," <<q.v[6]<<"," <<q.v[7]<<"," <<q.v[8]<<"," <<q.v[9] << endl;
   return idx;
}

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
    for(int i=0; i<n; i++)
      c = c + x[i] * A[xIndex + m * i];
    y[xIndex] = c;
  }
}

float matVecNaive (vect &n, vect & a, mat & l)
{

  double * out=n.v;
  double * in=a.v;
  double * A = l.m;
  int mrows=l.rows;
  int mcols=l.cols;
  int nsize=n.size;
  int asize=a.size;
  cout << " Going to multiply act(1," << asize << ") X layerw("<<mrows<<","<<mcols<<") to get netin (1x" << nsize << ")" << endl;
  
  
  // set up threading and blocking variables
  cudaDeviceProp dp;
  cudaGetDeviceProperties(&dp,0);
  unsigned int max_threads_per_block = dp.maxThreadsPerBlock;
//cout << "max_threads_per_block=" << max_threads_per_block << endl;
  int threads_perblockm = min(mrows, max_threads_per_block);
//cout << "threads_perblockm=" << threads_perblockm << endl;
  dim3 threadsPerBlockm(threads_perblockm);
  int num_blocksm = (int)ceil((double)mrows/(double)threads_perblockm);
//cout << "num_blocksm=" << num_blocksm << endl;
  dim3 numBlocksm(num_blocksm);

  // set up timing
  cudaEvent_t start, stop;
  float time;
  checkError(cudaEventCreate(&start));
  checkError(cudaEventCreate(&stop));
  checkError(cudaEventRecord(start,0));

 checkError(cudaMemcpy(dev_A, A,  mrows*mcols*sizeof(double), cudaMemcpyHostToDevice));
 checkError(cudaMemcpy(dev_in, in,  mrows*sizeof(double), cudaMemcpyHostToDevice));


  // execute kernel
  gen_matvec <<< numBlocksm, threadsPerBlockm >>>((double*)dev_A, (double*)dev_in, (double*)dev_out, mrows, mcols);
    //gen_matvec<<<  numBlocksm, threadsPerBlockm  >>> (mrows, mcols, LayerWeightsDevice, NetinDevice, ActuationDevice);
  checkError(cudaDeviceSynchronize());
 checkError(cudaMemcpy(out, dev_out,  mcols*sizeof(double), cudaMemcpyDeviceToHost));
  checkError(cudaEventRecord(stop,0));
  checkError(cudaEventSynchronize(stop));
  checkError(cudaEventElapsedTime(&time, start, stop));
  checkError(cudaEventDestroy(start));
  checkError(cudaEventDestroy(stop));

//cout << "out="<< out[0] << " "<< out[1] << endl;
  return time;
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

    for (int i = xindex; i < nc; i+= xstride)
    {
        Y[i] = 0;
        for (int j = yindex; j < nr; j+= ystride)
        {
             Y[i] += M[i * nc + j] * X[j];
        }
        Y[i] = Y[i] / nr;
    }

     __syncthreads();
}

void VectorMultiplyMatrix(vect & Y, vect & X, matx & M)
{
   if (Y.size == M.rows && M.cols == X.size)
   {
      for (int i = 0; i < M.rows; ++i)
      {
         Y.v[i] = 0;
         for (int j = 0; j < M.cols; ++j)
         {
            Y.v[i] += M.m[i*M.cols+j] * X.v[j];
         }
      }
   }
   else
   {
      cout << "Error: Trying to multiply a (1x"<<  X.size << ") with ("<<M.cols<<"x"<<M.rows<<") (Transposed) and store into a (1x"<<Y.size<<") !"<<endl;
      exit(1);
   }
}

void VectorMultiplyVector(matx & M, vect & Y, vect & X)
//void MatrixTranspMultiplyVector(vect & Y, vect & X, matx & M)
{
cout << "multiplying y=("<< Y.size << "x1) to X (1x" <<X.size<<") = (" << M.rows << "x" << M.cols << ")"<< endl;
// to multiply a (31x1) with (1x11) to get a (31x11) Matrix !

   if ( M.rows == Y.size && M.cols == X.size)
   {
      for (int i = 0; i < M.rows; ++i)
      {
         for (int j = 0; j < M.cols; ++j)
         {
            M.m[i*M.cols+j] = Y.v[i] * X.v[j];
            ONLYIF cout << "M["<<i<<"," << j<< "] =  X["<<j<<"] * Y[" << i << "]" << SPACE_GAP << "M="<<M.m[i*M.rows+j]<< " X=" << X.v[j] << " Y="<<  Y.v[i] << endl;
       ONLYIF     if (std::isnan(M.m[i]))
               cout << "NAN ERROR Y=" << Y.v[i] << "X=" << X.v[j] << " i=" << i << " j=" << endl;
         }
      }
   }
   else
   {
      cout << "Error: Trying to multiply a ("<<  X.size << "x1) with (1x" << Y.size <<") to get a ("<<M.rows<<"x"<<M.cols<<") Matrix !"<<endl;
      exit(1);
   }
}
void VectorMultiplyMatrixTransp(vect & Y, vect & X, matx & M)
{
ONLYIF cout << "multiplying x=(1x"<< X.size << ") to M ("<<M.cols<<"x"<<M.rows<<") = (1x" << Y.size << ")"<< endl;
   if (X.size == M.cols && M.rows == Y.size)
   {
      for (int i = 0; i < M.rows; ++i)
      {
         Y.v[i] = 0;
         for (int j = 0; j < M.cols; ++j)
         {
            Y.v[i] += M.m[i*M.cols+j] * X.v[j];
     ONLYIF        cout << "Y["<<i<<"] += M["<<i<<","<<j<<")* X["<<j<<"] =>"<< Y.v[i] << "=" << M.m[i*M.cols+j] << "*" << X.v[j] << endl;
        ONLYIF     if (std::isnan(Y.v[i]))
               cout << "NAN ERROR M=" << M.m[i*M.cols+j] << "X=" << X.v[j] << " i=" << i << " j=" <<  endl;
                
         }
      }
   }
   else
   {
      cout << "Error: Trying to multiply a (1x"<<  X.size << ") with ("<<M.cols<<"x"<<M.rows<<") (Transposed) and store into a (1x"<<Y.size<<") !"<<endl;
      exit(1);
   }
}
void MatrixVectorMultiply(vect & Y, vect & X, matx & M)
{
   if (X.size == M.rows && M.cols == Y.size)
   {
      for (int i = 0; i < M.rows; ++i)
      {
         Y.v[i] = 0;
         for (int j = 0; j < M.cols; ++j)
         {
            Y.v[i] += M.m[i*M.cols+j] * X.v[j];
         }
      }
   }
   else
   {
      cout << "Error: Trying to multiply a (1x"<<  X.size << ") with ("<<M.rows<<"x"<<M.cols<<") and store into a (1x"<<Y.size<<") !"<<endl;
      exit(1);
   }
}
int backprop(vect & tgt0, int y0)
{

        vect final;
        final.size = actuation[OutputLayer].size-1;
        final.v=new double[final.size];
        memcpy(final.v, actuation[OutputLayer].v, final.size);


        double err=0;
        for (int i=0;i<final.size;i++)
            err += (tgt0.v[i] - final.v[i]) * (tgt0.v[i] - final.v[i]) * 0.5;

        if (abs(err) < EPSILON)
        {
             int val=idxmaxval(tgt0, __LINE__);
#ifdef SAMPLEFREQ
             if ( (y0+1) % SAMPLEFREQ == 0) 
                cout << "---------------------------------- BACK PROPAGATION  sample=" << y0+1 <<" err=" << err << " < epsilon, for tgt '"<< val <<"' so error is acceptable, returning" << endl << flush;
#endif
             err_summary.v[val] = err;
             return 1;
        }

#ifdef SAMPLEFREQ
        if ( (y0+1) % SAMPLEFREQ == 0) 
          cout << "------------------------------------ BACK PROPAGATION sample="<< y0+1 << endl << flush;
#endif        
        
        for (int i=0;i< ftick[OutputLayer].size;i++)
        {
          if (i<tgt0.size)
          {
           ftick[OutputLayer].v[i] = (1-actuation[OutputLayer].v[i] ) * actuation[OutputLayer].v[i];
ONLYIF if (std::isnan(ftick[OutputLayer].v[i])) {
  cout << "Error ISNAN at ftick OutputLayer="<< OutputLayer << " i=" << i << endl; exit(1);}
           deltafn[OutputLayer].v[i]  =  (tgt0.v[i] - actuation[OutputLayer].v[i])*(ftick[OutputLayer].v[i]);
ONLYIF cout << deltafn[OutputLayer].v[i] << " = " << "(" << tgt0.v[i] << " - " << actuation[OutputLayer].v[i] << ") * " << ftick[OutputLayer].v[i]<<endl;
ONLYIF if (std::isnan(deltafn[OutputLayer].v[i])) {
  cout << "Error ISNAN at deltafn OutputLayer="<< OutputLayer << " i=" << i << endl; exit(1);}
          }
        }


        for (int i=OutputLayer-1;i>=0;i--)
        {
ONLYIF cout << "WUP("<<weight_updates[i].rows<<"x"<<weight_updates[i].cols<<") = DFN("<< deltafn[i+1].size << ") * actuation("<< actuation[i].size<< ")"<<endl;
            //VectorMultiplyVector(weight_updates[i], deltafn[i+1], actuation[i]);
            VectorMultiplyVector(weight_updates[i], actuation[i], deltafn[i+1]);
            
            VectorMultiplyMatrix(deltafn[i], deltafn[i+1], layer_weights[i]);
           // weight_updates[i]  =  deltafn[i+1].t() * actuation[i];
            for (int h=0;h<layer_weights[i].cols*layer_weights[i].rows;h++)
               new_layer_weights[i].m[h]  =  layer_weights[i].m[h] + (eta *  weight_updates[i].m[h]) ;
            for (int h=0;h<actuation[i].size;h++)
            {
                ftick[i].v[h] = (1 - actuation[i].v[h]) * actuation[i].v[h];
            //ftick[i] = -actuation[i] + 1;
            //ftick[i] = ftick[i] % (actuation[i]);  //element wise multiply
                deltafn[i].v[h] = deltafn[i].v[h] * ftick[i].v[h];
            }
            //deltafn[i] = deltafn[i+1]*layer_weights[i];

        }
        for (int i=0;i<OutputLayer;i++)
        {
          for (int j=0;j<layer_weights[i].rows*layer_weights[i].cols;j++)
           layer_weights[i].m[j] =  new_layer_weights[i].m[j];
        }
        return 0;
}
#if 0
// implementation of the matrix-vector multiply function
void MatrixVectorMultiply(rowvec  &n, rowvec  &a, mat  &m, double * ret)
{
    int mcols = m.n_rows;
    int mrows = m.n_cols;
    int m_biggest = max(mcols, mrows);
mat t=m.t();
rowvec res=a*t;
double * aptr=a.memptr();
double * nptr=n.memptr();
double * mptr=m.memptr();

    int   threadsPerBlock0 =256;
    //int   threadsPerBlock =thrds;
    //int   blocksPerGrid = (m_biggest + threadsPerBlock- 1) / threadsPerBlock;
    int blocksPerGrid0=m_biggest/threadsPerBlock0+1;
    //blocksPerGrid=4;
    //threadsPerBlock=256;
 
//    dim3 threadsPerBlock(16, 16);
//    dim3 numBlocks((N + threadsPerBlock.x -1) / threadsPerBlock.x, (N+threadsPerBlock.y -1) / threadsPerBlock.y);


  cudaDeviceProp dp;
  cudaGetDeviceProperties(&dp,0);
  unsigned int max_threads_per_block = dp.maxThreadsPerBlock;
  int threads_perblockm = min(mrows, max_threads_per_block);
  dim3 threadsPerBlockm(threads_perblockm);
  int num_blocksm = (int)ceil((double)mrows/(double)threads_perblockm);
  dim3 numBlocksm(num_blocksm);

  cout << "max_threads_per_block=" << max_threads_per_block << endl;
  cout << "threads_perblockm=" << threads_perblockm << endl;
  cout << "num_blocksm=" << num_blocksm << endl;

  // set up timing
  cudaEvent_t start, stop;
  float time;
  checkError(cudaEventCreate(&start));
  checkError(cudaEventCreate(&stop));
  checkError(cudaEventRecord(start,0));
 checkError(cudaMalloc( &LayerWeightsDevice, mrows*mcols*sizeof(double)));
 checkError(cudaMalloc( &ActuationDevice, mrows*sizeof(double)));
 checkError(cudaMalloc( &NetinDevice, mcols*sizeof(double)));


    checkError(cudaMemcpy(ActuationDevice, a.memptr(), mrows * sizeof(double), cudaMemcpyHostToDevice));
    checkError(cudaMemcpy(LayerWeightsDevice, m.memptr(), mrows * mcols * sizeof(double), cudaMemcpyHostToDevice));
     cout << "Calling gen_matvec2 || CUDA_MatrixVectorMultiply <<<" << num_blocksm << "," << threads_perblockm<< ">>> ("<<mrows<<","<<mcols<<","<<LayerWeightsDevice<<","<<NetinDevice<<","<<ActuationDevice<<endl;

    gen_matvec2<<<  numBlocksm, threadsPerBlockm  >>> (mrows, mcols, LayerWeightsDevice, NetinDevice, ActuationDevice);

    //checkError(cudaDeviceSynchronize());
  checkError(cudaDeviceSynchronize());
  checkError(cudaEventRecord(stop,0));
  checkError(cudaEventSynchronize(stop));
  checkError(cudaEventElapsedTime(&time, start, stop));
  checkError(cudaEventDestroy(start));
  checkError(cudaEventDestroy(stop));


    checkError(cudaMemcpy(ret, NetinDevice, mcols * sizeof(double), cudaMemcpyDeviceToHost));
cout <<"ret ================="<<endl;
  for (int x=0; x<mcols; x++)
  {
    cout << ret[x] << " ";
  }
  cout << endl;


}
#endif
void forward_feed(unsigned char * &imgdata, unsigned char * &labdata, bool train, int samples)
{
    int tot_correct=0;
    int tot_wrong=0;
    int correct_num=-1;
    int best_guess=-1;
    int num_correct[OUTPUT_LINES]={0,0,0,0,0,0,0,0,0,0};
    int num_wrong[OUTPUT_LINES]={0,0,0,0,0,0,0,0,0,0};
    int chosen_wrongly[OUTPUT_LINES][OUTPUT_LINES]={{ 0,0,0,0,0,0,0,0,0,0},
                                { 0,0,0,0,0,0,0,0,0,0},
                                { 0,0,0,0,0,0,0,0,0,0},
                                { 0,0,0,0,0,0,0,0,0,0},
                                { 0,0,0,0,0,0,0,0,0,0},
                                { 0,0,0,0,0,0,0,0,0,0},
                                { 0,0,0,0,0,0,0,0,0,0},
                                { 0,0,0,0,0,0,0,0,0,0},
                                { 0,0,0,0,0,0,0,0,0,0},
                                { 0,0,0,0,0,0,0,0,0,0}} ;
    int num_tested = 0;
    int epochs;
    string intype;
    if (train)
    {
       intype="TRAINING";
       epochs=EPOCHS;
    }
    else
    {
       epochs=1;
       intype="TEST    ";
    }
    for (int y=0;y<samples;y++)
    {
#ifdef SAMPLEFREQ
        if ( (y+1) % SAMPLEFREQ == 0)
           cout << "------------------------------------ FORWARD FEED OF "<<intype <<" SAMPLE # "<< y+1 << endl << flush;
#endif
        load_an_image(y, imgdata, actuation[0], tgt, labdata);
        int tgtval = idxmatchval(tgt, 1);
        for (int e=0;e<epochs;e++)
        {
            for (int i=0;i<OutputLayer;i++)  // only n-1 transitions between n layers
            {
               // cout << "------------------------------------ All inputs into L" << i << endl << flush;
                // sum layer 1 weighted input
#ifdef SERIAL_ONLY
                //netin[i] =  (actuation[i] * layer_weights[i].t())/actuation[i].n_cols;
                
                VectorMultiplyMatrix(netin[i],  actuation[i], layer_weights[i]);
#else
             //   MatrixVectorMultiply(netin[i],  actuation[i], layer_weights[i], netptrs[i]);
                float t = matVecNaive (  netin[i],  actuation[i], layer_weights[i]);
                //memcpy(netptrs[i], nettemp, actuation[i].n_cols * sizeof(double));

                if (t > maxtime)
                   maxtime=t;
                if (t < mintime )
                   mintime=t;

#endif    
                for (int k=0;k< netin[i].size;k++)
                   netin[i].v[k] = netin[i].v[k] / actuation[i].size;
                sigmoid(netin[i],  actuation[i+1]);
            }
#ifdef SAMPLEFREQ
            if ( (y+1) % SAMPLEFREQ == 0)
            {
               std::cout << "Final output : " << endl <<   std::setw(7) << fixed << showpoint << actuation[OutputLayer].v[0] << SPACE_GAP 
                                                                                              << actuation[OutputLayer].v[1] << SPACE_GAP
                                                                                              << actuation[OutputLayer].v[2] << SPACE_GAP
                                                                                              << actuation[OutputLayer].v[3] << SPACE_GAP
                                                                                              << actuation[OutputLayer].v[4] << SPACE_GAP
                                                                                              << actuation[OutputLayer].v[5] << SPACE_GAP
                                                                                              << actuation[OutputLayer].v[6] << SPACE_GAP
                                                                                              << actuation[OutputLayer].v[7] << SPACE_GAP
                                                                                              << actuation[OutputLayer].v[8] << SPACE_GAP
                                                                                              << actuation[OutputLayer].v[9] << SPACE_GAP
                                                                                               << " Sample: " << y+1 <<std::endl << flush;
               std::cout << "Expec output : " << endl  <<  std::setw(7) << fixed << showpoint << tgt.v[0] <<  SPACE_GAP
                                                                                              << tgt.v[1] <<  SPACE_GAP
                                                                                              << tgt.v[2] <<  SPACE_GAP
                                                                                              << tgt.v[3] <<  SPACE_GAP
                                                                                              << tgt.v[4] <<  SPACE_GAP
                                                                                              << tgt.v[5] <<  SPACE_GAP
                                                                                              << tgt.v[6] <<  SPACE_GAP
                                                                                              << tgt.v[7] <<  SPACE_GAP
                                                                                              << tgt.v[8] <<  SPACE_GAP
                                                                                              << tgt.v[9] <<  SPACE_GAP
                                                                                              << " Sample: " << y+1 << std::endl << flush;
            }
#endif            
                    //////////////////////////// forward feed end
            if (train)
            {
                 // printout intermediate result
                  int outval = idxmaxval(actuation[OutputLayer],__LINE__);
#ifdef SAMPLEFREQ
                  if ( (y+1) % SAMPLEFREQ == 0)
                  {
                      std::cout << "Train output : " << endl  <<  std::setw(7) << fixed << showpoint  << actuation[OutputLayer].v[0] <<  SPACE_GAP
                                                                                              << actuation[OutputLayer].v[1] << SPACE_GAP
                                                                                              << actuation[OutputLayer].v[2] << SPACE_GAP
                                                                                              << actuation[OutputLayer].v[3] << SPACE_GAP
                                                                                              << actuation[OutputLayer].v[4] << SPACE_GAP
                                                                                              << actuation[OutputLayer].v[5] << SPACE_GAP
                                                                                              << actuation[OutputLayer].v[6] << SPACE_GAP
                                                                                              << actuation[OutputLayer].v[7] << SPACE_GAP
                                                                                              << actuation[OutputLayer].v[8] << SPACE_GAP
                                                                                              << actuation[OutputLayer].v[9] << SPACE_GAP
                                                                                               << " Sample: " << y+1 <<std::endl << flush;
                  // Below just figures out the order in which to print the "A"ctal result and "O"bjective result
                  // (or "*" if correct) in the output line.
                  // So tgtval is correct if lastval==firstval(they are indicies, and will be equal if tgtval==outval)
                     int firstval= tgtval<outval?tgtval:outval;
                     int lastval= tgtval>outval?tgtval:outval;
                     string firststr= tgtval == firstval ? to_string(firstval)+string("T"):to_string(firstval)+string("O");
                     string laststr= tgtval == lastval? to_string(lastval)+"T":to_string(lastval)+"O";
                     if (firstval==lastval)
                        firststr="*"+to_string(firstval); // correct
                     for (int z1 = 0; z1 < firstval; z1++)
                         cout << "         "; 
                     cout << "       " << firststr;   
                     for (int z1 = 0; z1 < lastval- firstval-1; z1++)  
                         cout << "         "; 
                     if (firstval!= lastval)
                         cout << "       " << laststr;  // expected   
                     cout << endl << flush;      
                  }
#endif
                  if (backprop(tgt, y) == 1)
                     break;  // exit i/epoch loop and goto next sample (as error function is within limits for this tgt)
            }
        }



        if (!train)
        {

            correct_num = idxmatchval(tgt, 1);
            best_guess = idxmaxval(actuation[OutputLayer],__LINE__);

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
        if (!train ) 
        {
          std::cout << "Final output : " << endl  << std::setw(7) << fixed << showpoint << actuation[OutputLayer].v[0] << SPACE_GAP 
                                                                                        << actuation[OutputLayer].v[1] << SPACE_GAP
                                                                                        << actuation[OutputLayer].v[2] << SPACE_GAP
                                                                                        << actuation[OutputLayer].v[3] << SPACE_GAP
                                                                                        << actuation[OutputLayer].v[4] << SPACE_GAP
                                                                                        << actuation[OutputLayer].v[5] << SPACE_GAP
                                                                                        << actuation[OutputLayer].v[6] << SPACE_GAP
                                                                                        << actuation[OutputLayer].v[7] << SPACE_GAP
                                                                                        << actuation[OutputLayer].v[8] << SPACE_GAP
                                                                                        << actuation[OutputLayer].v[9] << SPACE_GAP
                                                                                            << " Sample: " << y+1 <<std::endl << flush;
            best_guess = idxmaxval(actuation[OutputLayer],__LINE__);
            correct_num = idxmatchval(tgt, 1);
          int maxw = best_guess>correct_num?best_guess:correct_num;
          for (int z1=0;z1<maxw;z1++)
             cout << "         ";
          cout << "       ^" << endl << flush;
          std::cout << "Expec output : " << endl <<  std::setw(7) << fixed << showpoint << tgt.v[0] << SPACE_GAP
                                                                                        << tgt.v[1] << SPACE_GAP
                                                                                        << tgt.v[2] << SPACE_GAP
                                                                                        << tgt.v[3] << SPACE_GAP
                                                                                        << tgt.v[4] << SPACE_GAP
                                                                                        << tgt.v[5] << SPACE_GAP
                                                                                        << tgt.v[6] << SPACE_GAP
                                                                                        << tgt.v[7] << SPACE_GAP
                                                                                        << tgt.v[8] << SPACE_GAP
                                                                                        << tgt.v[9] << SPACE_GAP
                                                                                      << " Sample: " << y+1 << std::endl << flush;
        }

    }
    if (!train)
    {
         confusion_matrix << "Tested         " << num_tested << " samples"<<endl << flush;
         confusion_matrix << "Tested Correct " << tot_correct << " samples"<<endl << flush;
         confusion_matrix << "Tested Wrong   " << tot_wrong<< " samples"<<endl << endl << endl << "  " << flush;
         for (int i=0;i<OUTPUT_LINES;i++)
             confusion_matrix  <<  "     "<< dec << std::setw(6) << "'" <<i << "'";
         confusion_matrix << " <-- ANN chose" << endl << flush;
         confusion_matrix << "-----------------------------------------------------------------------------------------------------------------------------------------" ;
         double colsum[OUTPUT_LINES]={0,0,0,0,0,0,0,0,0,0};
         double rowsum[OUTPUT_LINES]={0,0,0,0,0,0,0,0,0,0};
         string blanks="                    ";
         for (int i=0;i<OUTPUT_LINES;i++)
         {
            string correct_size=to_string(num_correct[i]);
            confusion_matrix << endl <<  setw(4)   << i << "  |";
            for (int j=0;j<OUTPUT_LINES;j++)
            {
                rowsum[i] +=  chosen_wrongly[i][j];
                colsum[j] +=  chosen_wrongly[i][j];
                if (i==j)
                   confusion_matrix  << std::setw(6) << "[" <<  num_correct[i] <<  "]" << blanks.substr(0, 5-correct_size.length()) << "|";
                else
                   confusion_matrix  << std::setw(7) << chosen_wrongly[i][j] <<  "     |";
            }
             float pctg=0;
             if (tot_wrong!=0)
                  pctg=(float)(rowsum[i])/ (float) (tot_wrong) * 100.0f;
            confusion_matrix << "  " <<  setw(7)  << std::setw(7) ;
            confusion_matrix.copyfmt(init);
            confusion_matrix <<rowsum[i] ;
            confusion_matrix <<  setw(7)   <<"     " <<  fixed << showpoint <<pctg  <<   "%"<<endl << flush;
            confusion_matrix.copyfmt(init);
            confusion_matrix << "-----------------------------------------------------------------------------------------------------------------------------------------" ;

         }
         confusion_matrix << endl << "   ^   " ;
         for (int i=0;i<OUTPUT_LINES;i++)
             confusion_matrix  << dec << std::setw(7) << colsum[i] << "      ";
         confusion_matrix << endl << "Target   ";
         for (int i=0;i<OUTPUT_LINES;i++)
         {
             float pctg=0;
             if (tot_wrong!=0)
                pctg=(float)(colsum[i])/ (float) (tot_wrong) * 100.0f;
            confusion_matrix << dec <<  setw(7) << fixed << showpoint <<  pctg  << "%     ";
             confusion_matrix.copyfmt(init);
         }
         confusion_matrix << endl << endl << endl << endl << endl << "Correct selections:" << endl << flush;
         confusion_matrix << "       ";
         for (int i=0;i<OUTPUT_LINES;i++)
             confusion_matrix  << dec << std::setw(6) << "'" << i << "'     ";
         confusion_matrix << endl << "       ";
         for (int i=0;i<OUTPUT_LINES;i++)
         {
                confusion_matrix  << std::setw(7) << num_correct[i] <<  "      ";
         }
         confusion_matrix << endl << endl << "Incorrect selections:" << endl << flush;
         confusion_matrix << "       ";
         for (int i=0;i<OUTPUT_LINES;i++)
             confusion_matrix  << dec << std::setw(6) << "'" << i << "'     ";
         confusion_matrix << endl << "       ";
         for (int i=0;i<OUTPUT_LINES;i++)
         {
                confusion_matrix  << std::setw(7) << num_wrong[i] <<  "      ";
         }
         confusion_matrix << endl << endl << flush; 
         float pctg=(float)(tot_correct)/ (float) (tot_correct+tot_wrong) * 100.0f;
         confusion_matrix << "Total Correct : " <<  std::setw(7) << fixed << showpoint  <<pctg << "%     " << endl << endl << flush;
         cout << confusion_matrix.str() << flush;
         confusion_matrix.copyfmt(init);
         cout.copyfmt(init);
    }
                
}




void save_weights(string hdr)
{
    ofstream oFile;
    string fname = hdr+string("_weights_") + fid +string(".txt");
    cout << "Saving weights to file : " << fname << endl << flush;
    oFile.open(fname, ios::out);
    if (hdr.substr(0,4)=="post")
       oFile << confusion_matrix.str();
    oFile << "NumberOfLayers=" << NumberOfLayers << endl << flush;
 
    for (int i=0; i< OutputLayer; i++)
    {

        oFile <<  "NodesInLayer"<<i<<"=" << nodes[i] << endl << flush;
        for (int j=0;j < layer_weights[i].cols*layer_weights[i].rows;j++)
        {
            oFile << layer_weights[i].m[j] << SPACE_GAP;
            if ((j+1) %  layer_weights[i].cols == 0) 
               oFile << endl;
        }
        oFile << endl;
    }
    oFile << "Error Summary" << endl << flush;

    for (int j=0;j < err_summary.size;j++)
       oFile << err_summary.v[j] << SPACE_GAP;
    oFile << endl << flush;

    oFile << "EndFile" << endl << flush;
    oFile.close();

}



int main (int argc, char *argv[])
{
   extern char **environ;
   string hname="";

    vector<string> strs;
    string bldver = string(__DATE__) + " at time " + string(__TIME__);
        cout << "--------------------------------  Build done on " << bldver << endl << flush;
        init.copyfmt(cout);
        if (argc < 2)
        {
            NumberOfLayers=3;
            nodes = new unsigned int [NumberOfLayers];
            nodes[0]=INPUT_LINES;
            nodes[1]=DEFAULT_HIDDEN;
            nodes[2]=OUTPUT_LINES;
            eta = ETA_DEFAULT;
            cout << "Using default setting of \"" << nodes[0] << " " << nodes[1] << " " << nodes[2]<<  "\" " << endl << flush;
            cout << "And ETA=" << eta << endl << flush;;
        }
        else if (argc < 6)
        {
             cout << "Usage: " << argv[0] << " ETA IN H1 [H2 H3 ...] OUT THREADS" << endl << flush;
             cout << "       Where ETA is the learning factor, &" << endl << flush;
             cout << "       Where number of parameters after ETA is the number of layers" << endl << flush;
             cout << "       Must have a minimum of 3, i.e. IN H1 OUT" << endl << flush;
             cout << "       And the parameters themselves are numbers, "<< endl << flush;
             cout << "       indicating the number of nodes in that layer." << endl << flush;
             cout << "       e.g. \"" << argv[0] <<  " "<< ETA_DEFAULT << " " << INPUT_LINES << " " << DEFAULT_HIDDEN << " " << OUTPUT_LINES << " " << DEFTHREADS << "\" " << endl << flush;
             cout << "       and is the default, if no params supplied." << endl << flush;
             exit (1);
        }
        else
        {
             NumberOfLayers = argc-3;
             nodes = new unsigned int [NumberOfLayers];
             eta = stod(string(argv[1]));
             if (eta <= 0)
             {
                   cout << "Error: ETA must be positive, usually less than 1" << endl << flush;
                   exit(1);
             }
             for (int i=2;i<argc-1;i++)
             {
                int p = stoi(string(argv[i]));
                if (p > 0)
                {
                   nodes[i-2] = stoi(string(argv[i]));
                }
                else
                {
                   cout << "Error in parameter " << i << " - must be positive" << endl << flush;
                   exit (1);
                }
             }
             thrds=stoi(argv[argc-1]);
        }
        cout << "Threads chosen is " << thrds << endl << flush;
        cout << "Number of Layers is " << NumberOfLayers << endl << flush;

    //netptrs = new double * [NumberOfLayers];
    // Use slurm job number if avaiable (else defaults to epoch time) for file ids created
    for(char **current = environ; *current; current++) {
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


    tgt.size=10;
    tgt.v = new double [tgt.size];

    tgt0.size=11;
    tgt0.v = new double [tgt0.size];

    err_summary.size = OUTPUT_LINES;
    err_summary.v = new double [err_summary.size];
    for (int q=0;q<err_summary.size;q++)
      err_summary.v[q] = -1.0;

    OutputLayer = NumberOfLayers -1;
    unsigned char * trainlabels; 
    unsigned char * testlabels; 
    unsigned char * traindata = load_file("train-images-idx3-ubyte", "train-labels-idx1-ubyte", &trainlabels);
    unsigned char * testdata = load_file("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", &testlabels);
    auto StartTime = std::chrono::high_resolution_clock::now();

///////////////////////////////////////////////
//
//  CREATE ARRAY OF MATRICES AND VECTORS
//  AND SET WEIGHTS TO RANDOM (0 < w < 1)
//
    int max_mat=0;
    int max_vec=0;
    int bias_field = 1;

    for (int i=0;i <= OutputLayer; i++)
    {
         max_vec=max(max_vec, (nodes[i]+bias_field));

         vect rbf;
         rbf.size = (nodes[i]+bias_field);
         rbf.v = new double [rbf.size];
         ftick.push_back(rbf);

         vect rbd;
         rbd.size = (nodes[i]+bias_field);
         rbd.v = new double [rbd.size];
         deltafn.push_back(rbd);

         vect rb;
         rb.size = (nodes[i]+bias_field);
         rb.v = new double [rb.size];

         actuation.push_back(rb); // size= nodes[i],1
         if (i<OutputLayer)
         {
            max_mat=max(max_mat, (nodes[i]+bias_field)*(nodes[i+1]+bias_field));
            //netptrs[i] = new double [nodes[i]+bias_field];
            //rowvec rb2 (netptrs[i], nodes[i+1]+bias_field, false, true);
            vect rb2;
            rb2.size = (nodes[i+1]+bias_field);
            rb2.v = new double [rb2.size];

            netin.push_back(rb2);   // size=nodes[i],1
            matx tmpwgt;
            tmpwgt.rows=nodes[i]+1;
            tmpwgt.cols=nodes[i+1]+1;
            tmpwgt.m =  new double [ tmpwgt.rows * tmpwgt.cols];
            for (int j=0;j<tmpwgt.rows * tmpwgt.cols;j++)
               tmpwgt.m[j] = ((double)rand())/(double)RAND_MAX;
            matx tmpwgtup;
            tmpwgtup.rows=nodes[i]+1;
            tmpwgtup.cols=nodes[i+1]+1;
            tmpwgtup.m =  new double [ tmpwgtup.rows * tmpwgtup.cols];

            matx tmpwgtnew;
            tmpwgtnew.rows=nodes[i]+1;
            tmpwgtnew.cols=nodes[i+1]+1;
            tmpwgtnew.m =  new double [ tmpwgtnew.rows * tmpwgtnew.cols];

            layer_weights.push_back( tmpwgt );
            new_layer_weights.push_back(tmpwgtnew);
            weight_updates.push_back(tmpwgtup);
          }
    }
    save_weights("initial_random_values");
   cout << "Max Matrix size " << max_mat << " Max vector size = " << max_vec << endl << flush;
   cout << "vector lens=" << netin.size() <<"," <<layer_weights.size() << "," <<actuation.size() << endl;

   checkError(cudaMalloc(&ActuationDevice, max_vec * sizeof(double)));
   checkError(cudaMalloc(&NetinDevice, max_vec * sizeof(double)));
   checkError(cudaMalloc(&LayerWeightsDevice, max_mat * sizeof(double)));
 checkError(cudaMalloc( &dev_A, max_mat*sizeof(double)));
 checkError(cudaMalloc( &dev_in, max_vec*sizeof(double)));
 checkError(cudaMalloc( &dev_out, max_vec*sizeof(double)));

#ifdef __CUDA_ARCH__ 
cout << "CUDA ARCH == " << __CUDA_ARCH__ << endl;
#endif

/////////////////////////////////////////////// 
//
// TRAIN THE DATA
//
    auto StartTrainTime = std::chrono::high_resolution_clock::now();
    cout << "Training on data started (epochs=" << EPOCHS << ")...." << endl << flush;

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

   auto TotalTime = std::chrono::duration_cast<std::chrono::microseconds>(EndTestTime-StartTime);
   auto TrainTime =  std::chrono::duration_cast<std::chrono::microseconds>(EndTrainTime-StartTrainTime);
   auto TestTime =  std::chrono::duration_cast<std::chrono::microseconds>(EndTestTime-StartTestTime);
 
    cout << "Total Time       : " <<    std::setw(12) << TotalTime.count() <<" us"<< endl << flush;
    cout << "Total Train Time : " << std::setw(12) <<    TrainTime.count() <<" us"<< endl << flush;
    cout << "Total Test Time  : " <<  std::setw(12) <<   TestTime.count() <<" us"<< endl << flush;

    confusion_matrix << "Epochs in Training : " << EPOCHS << endl << flush;
    confusion_matrix << "Training Samples   : " << TRAININGSAMPLES << endl << flush;
    confusion_matrix << "Testing Samples    : " << TESTINGSAMPLES << endl << flush;
    confusion_matrix << endl << endl <<  "Total Time       : " <<    std::setw(12) << TotalTime.count() <<" us"<< endl << flush; 
    confusion_matrix << "Total Train Time : " << std::setw(12) <<    TrainTime.count() <<" us"<< endl << flush;
    confusion_matrix  << "Total Test Time  : " <<  std::setw(12) <<   TestTime.count() <<" us"<< endl << flush;
    confusion_matrix << endl << endl <<  "Total Time       : " <<    std::setw(12) << TotalTime.count()/1000000 <<" s"<< endl << flush; 
    confusion_matrix << "Total Train Time : " << std::setw(12) <<    TrainTime.count()/1000000 <<" s"<< endl << flush;
    confusion_matrix  << "Total Test Time  : " <<  std::setw(12) <<   TestTime.count()/1000000 <<" s"<< endl << flush;
    confusion_matrix << endl << endl <<  "Total Time       : " <<    std::setw(12) << TotalTime.count()/60000000 <<" min"<< endl << flush; 
    confusion_matrix << "Total Train Time : " << std::setw(12) <<    TrainTime.count()/60000000 <<" min"<< endl << flush;
    confusion_matrix  << "Total Test Time  : " <<  std::setw(12) <<   TestTime.count()/60000000 <<" min"<< endl << flush;
    confusion_matrix << "Epsilon  : " << EPSILON << endl << flush;
    confusion_matrix << "Eta      : " << eta << endl << flush;
    confusion_matrix << "Build ver: " << bldver<<endl << flush;
    save_weights("post_training_weights");
        delete[] traindata;
        delete[] trainlabels;
        delete[] testdata;
        delete[] testlabels;
   checkError(cudaFree(LayerWeightsDevice));
   checkError(cudaFree(ActuationDevice));
   checkError(cudaFree(NetinDevice));
#ifndef SERIAL_ONLY
   cout << "Max time for CUDA call : " << maxtime << endl;
   cout << "Min time for CUDA call : " << mintime << endl;
#endif
}
