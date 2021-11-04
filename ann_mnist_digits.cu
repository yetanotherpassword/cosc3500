#include <iostream>
#include <iomanip>
#include <cmath>
#include <nvblas.h>
#include <cublas.h>
#define ARMA_ALLOW_FAKE_GCC
#include <armadillo>
#include <boost/algorithm/string.hpp>

#undef DEBUGON
#define DEFTHREADS 64
#define ARMA_64BIT_WORD
#define INPUT_LINES 784
#define OUTPUT_LINES 10
#define MATRIX_SIDE 28
#define MAX_PIXEL_VAL 255.0f
#define IMAGE_OFFSET 16
#define DEFAULT_HIDDEN 30
#define ETA_DEFAULT 0.5f

#define SAMPLEFREQ 1000
#define EPOCHS 512
#define EPSILON 1E-04
#define TRAININGSAMPLES 60000
#define TESTINGSAMPLES 10000
#define BLOCK_HEIGHT 1024
#define BLOCK_WIDTH 64

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

    int thrds=DEFTHREADS; 
    using namespace arma;
    using namespace std;

    double * LayerWeightsDevice;
    double * ActuationDevice;
    double * NetinDevice;
  double * dev_A;
  double * dev_in;
  double * dev_out;

    std::time_t result = std::time(nullptr);
    string fid = to_string(result);
    unsigned int NumberOfLayers;
    unsigned int OutputLayer;
    unsigned int * nodes;
    double eta;               // Learning factor
    vector<rowvec> netin;
    vector<rowvec> actuation;
    vector<rowvec> deltafn;
    vector<rowvec> ftick;
    vector<mat> layer_weights;
    vector<mat> weight_updates;
    vector<mat> new_layer_weights;
    mat tmpwgt; 
    ios init(NULL);
    int vec_start_idx[100];
    int mat_start_idx[100];
    double * netin2;
    double * actuation2;
    double * layer_weights2;
    stringstream confusion_matrix;
    rowvec err_summary=ones<rowvec>(OUTPUT_LINES) * (-1);

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


rowvec sigmoid( rowvec  & net)
{
    rowvec out = 1/(1+exp(-net));
    out(out.n_cols-1)=1.0;          // add bias signal value
    return out;
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

void load_an_image(int seq, unsigned char * &mptr, rowvec & img, rowvec & t, unsigned char * &lp)
{
    int start=(INPUT_LINES*seq)+IMAGE_OFFSET;
    double greyval=MAX_PIXEL_VAL;

    for (int i=0;i<INPUT_LINES;i++)
    {
        img(i) = ((double ) mptr[start+i])/greyval;
    }

    img(nodes[0])=1;          // set bias signal, so can multiply with [node weights | bias weights] augmented matrix

    int img_is_digit=(int) lp[8+seq];
    if ((seq+1) % SAMPLEFREQ ==0)
    {
       cout << "For sample :" << seq+1 << endl << flush;
       print_an_image(&mptr[start], img_is_digit);
    }

    t=zeros<rowvec>(OUTPUT_LINES); // create the target vector (plus one for 'bias' bit)
    t(img_is_digit)=1;               // set the target 'bit'

}
// For use with gdb
void output (mat t)
{
   cout << t << endl;
}
// For use with gdb
void output (rowvec t)
{
   cout << t << endl;
}

int backprop(rowvec tgt, int y0)
{

        rowvec final = actuation[OutputLayer];
        final.shed_col(nodes[OutputLayer]-1);
        rowvec tgt0  = tgt;
        tgt0.insert_cols(nodes[OutputLayer],1);
        double err = accu((tgt - final) %  (tgt - final))*0.5;
        if (abs(err) < EPSILON)
        {
             int val=tgt.index_max();
             if ( (y0+1) % SAMPLEFREQ == 0) 
                cout << "---------------------------------- BACK PROPAGATION  sample=" << y0+1 <<" err=" << err << " < epsilon, for tgt '"<< val <<"' so error is acceptable, returning" << endl << flush;
             err_summary(val) = err;
             return 1;
        }

        if ( (y0+1) % SAMPLEFREQ == 0) 
          cout << "------------------------------------ BACK PROPAGATION sample="<< y0+1 << endl << flush;
        
        ftick[OutputLayer] = -actuation[OutputLayer] + 1;
        ftick[OutputLayer] = ftick[OutputLayer] % (actuation[OutputLayer]);  //element wise multiply
        deltafn[OutputLayer]  =  (tgt0 - actuation[OutputLayer])%(ftick[OutputLayer]);

        for (int i=OutputLayer-1;i>=0;i--)
        {
  
            weight_updates[i]  =  deltafn[i+1].t() * actuation[i];
            new_layer_weights[i]  =  layer_weights[i] + (eta *  weight_updates[i]) ;
             
            ftick[i] = -actuation[i] + 1;
            ftick[i] = ftick[i] % (actuation[i]);  //element wise multiply
            deltafn[i] = deltafn[i+1]*layer_weights[i];
            deltafn[i] = deltafn[i] % ftick[i];

        }
        for (int i=0;i<OutputLayer;i++)
        {
           layer_weights[i] =  new_layer_weights[i];
        }
        return 0;
}

void checkError(cudaError_t e)
{
    if (e != cudaSuccess)
    {
        std::cerr << "CUDA error: " << int(e) << " : " << cudaGetErrorString(e) << '\n';
        abort();
    }
}



__global__ void gen_matvec2(const int m, const int n, double *A, double*y, double*x)
{
  unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
  if ( xIndex < n ){
    double c = 0.0f;
    for(int i=0; i<m; i++)
      c = c + x[i] * A[xIndex + n * i];
    y[xIndex] = c;
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

 checkError(cudaMemcpy(dev_A, A,  m*n*sizeof(double), cudaMemcpyHostToDevice));
 checkError(cudaMemcpy(dev_in, in,  m*sizeof(double), cudaMemcpyHostToDevice));


  // execute kernel
  gen_matvec <<< numBlocksm, threadsPerBlockm >>>((double*)dev_A, (double*)dev_in, (double*)dev_out, m, n);
    //gen_matvec<<<  numBlocksm, threadsPerBlockm  >>> (mrows, mcols, LayerWeightsDevice, NetinDevice, ActuationDevice);
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
  checkError(cudaThreadSynchronize());
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


void forward_feed(unsigned char * &imgdata, unsigned char * &labdata, bool train, int samples)
{
    rowvec tgt;
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
        if ( (y+1) % SAMPLEFREQ == 0)
           cout << "------------------------------------ FORWARD FEED OF "<<intype <<" SAMPLE # "<< y+1 << endl << flush;
        load_an_image(y, imgdata, actuation[0], tgt, labdata);
        int tgtval = tgt.subvec(0,9).index_max();

        for (int e=0;e<epochs;e++)
        {
            for (int i=0;i<OutputLayer;i++)  // only n-1 transitions between n layers
            {
               // cout << "------------------------------------ All inputs into L" << i << endl << flush;
                // sum layer 1 weighted input
#ifdef SERIAL_ONLY
                netin[i] =  (actuation[i] * layer_weights[i].t())/actuation[i].n_cols;
                cout << "Netin serial ("<<  netin[i].n_rows << "," <<  netin[i].n_cols << ")= "  << netin[i] << endl << flush;
#else
                //cout << "Netin2  ("<<  netin[i].n_rows << "," <<  netin[i].n_cols << ")= "  << netin[i] << endl << flush;
             //   MatrixVectorMultiply(netin[i],  actuation[i], layer_weights[i], netptrs[i]);
 matVecNaive (  netin[i].memptr(),  actuation[i].memptr(), layer_weights[i].memptr(), layer_weights[i].n_cols, layer_weights[i].n_rows) ;
                //memcpy(netptrs[i], nettemp, actuation[i].n_cols * sizeof(double));
   netin[i] = netin[i]/actuation[i].n_cols; 
                cout << "Netin2  ("<<  netin[i].n_rows << "," <<  netin[i].n_cols << ")= "  << netin[i] << endl << flush;
#endif    

                actuation[i+1] = sigmoid(netin[i]);
            }
            if ( (y+1) % SAMPLEFREQ == 0)
            {
               std::cout << "Final output : " << endl <<   std::setw(7) << fixed << showpoint << actuation[OutputLayer].subvec(0,9) << " Sample: " << y+1 <<std::endl << flush;
               std::cout << "Expec output : " << endl  <<  std::setw(7) << fixed << showpoint << tgt.subvec(0,9) << " Sample: " << y+1 << std::endl << flush;
            }
            
                    //////////////////////////// forward feed end
            if (train)
            {
                 // printout intermediate result
                  int outval = actuation[OutputLayer].subvec(0,9).index_max();
                  if ( (y+1) % SAMPLEFREQ == 0)
                  {
                      std::cout << "Train output : " << endl  <<  std::setw(7) << fixed << showpoint  << actuation[OutputLayer].subvec(0,9) << " Sample: " << y+1 << std::endl << flush;
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
                  if (backprop(tgt, y) == 1)
                     break;  // exit i/epoch loop and goto next sample (as error function is within limits for this tgt)
            }
        }



        if (!train)
        {
            correct_num = tgt.subvec(0,9).index_max();
            best_guess = actuation[OutputLayer].subvec(0,9).index_max();

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
        if (!train ||  (y+1) % SAMPLEFREQ == 0)
        {
          std::cout << "Final output : " << endl  << std::setw(7) << fixed << showpoint << actuation[OutputLayer].subvec(0,9) << " Sample: " << y+1 <<std::endl << flush;
          for (int z1=0;z1<actuation[OutputLayer].subvec(0,9).index_max();z1++)
             cout << "         ";
          cout << "       ^" << endl << flush;
          std::cout << "Expec output : " << endl <<  std::setw(7) << fixed << showpoint << tgt.subvec(0,9) << " Sample: " << y+1 << std::endl << flush;
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
        oFile << layer_weights[i] << endl << flush;
    }
    oFile << "Error Summary" << endl << flush;

    oFile << err_summary << endl << flush;

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
    vec_start_idx[0]=0;
    mat_start_idx[0]=0;
    int max_mat=0;
    int max_vec=0;
    int bias_field = 1;

    for (int i=0;i <= OutputLayer; i++)
    {
         max_vec=max(max_vec, (nodes[i]+bias_field));
         rowvec rb (nodes[i]+bias_field);
         actuation.push_back(rb); // size= nodes[i],1
         deltafn.push_back({});
         ftick.push_back({});
         if (i<OutputLayer)
         {
            max_mat=max(max_mat, (nodes[i]+bias_field)*(nodes[i+1]+bias_field));
            //netptrs[i] = new double [nodes[i]+bias_field];
            //rowvec rb2 (netptrs[i], nodes[i+1]+bias_field, false, true);
            rowvec rb2 (nodes[i+1]+bias_field);

            netin.push_back(rb2);   // size=nodes[i],1

            tmpwgt = randu<mat>( nodes[i+1]+1,nodes[i]+1); // network weights for each node + 1 node bias weight
            layer_weights.push_back( tmpwgt );
            new_layer_weights.push_back(tmpwgt);
            weight_updates.push_back(tmpwgt);
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
cout << "CUDA ARCH ++ " << __CUDA_ARCH__ << endl;
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
}
#if 0
#include <iostream>
using namespace std;

__global__
void kernel(float *a, float *b, float *c, int N, int M) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float sum = 0;
    if (tid < M) {
        for (int i = 0; i < N; i++)
            sum += a[i] * b[(i * M) + tid];
        c[tid] = sum;

    }
}

int main(void) {

    float *dev_a, *dev_b, *dev_c;

    int N = 785;
    int M = 31;

    float a[N];
    float b[N][M];
    float c[M];

    for (int i = 0; i < N; i++) {
        a[i] = (float) rand()/ (float) RAND_MAX;
        cout << "a[" << i << "]=" << a[i] << endl;
    }

    for (int i = 0; i < N; i++) {
        for (int e = 0; e < M; e++) {
            b[i][e] = (float) rand()/ (float) RAND_MAX;
        cout << "b[" << i << "][" << e << "]=" << b[i][e] << endl;
        }
    }

    cudaMalloc((void**) &dev_a, sizeof(float) * N);
    cudaMalloc((void**) &dev_b, sizeof(float) * N * M);
    cudaMalloc((void**) &dev_c, sizeof(float) * M);

    cudaMemcpy(dev_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, sizeof(float) * N * M, cudaMemcpyHostToDevice);

    kernel<<<M / 256 + 1, 256>>>(dev_a, dev_b, dev_c, N, M);

    cudaMemcpy(c, dev_c, sizeof(float) * M, cudaMemcpyDeviceToHost);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    for (int i = 0; i < M; i++) {
        //cout << c[i] << endl;
        cout << "c[" << i << "]=" << c[i] << endl;
    }
// for (int i=0;i<10;i++)
//   cout << (float) rand()/ (float) RAND_MAX << endl;
    return 0;
}
void matrix_mult (double  * m2, double * vin2, double * vout2, int idx)
{
   int ms=mat_start_idx[idx];
   int me=mat_start_idx[idx+1];
   int vs=vec_start_idx[idx];
   int ve=vec_start_idx[idx+1];
   int vl=ve-vs;
   int ml=me-ms;
   double tmp =0;

   for (int i=0;i<ml;i++)
   {
        int vi = i % vl;
        tmp += m2[ms+i]*vin2[vs+vi];
        if ((vi == 0) || (i+1) % vi == 0)
        {
           int vo2idx=(i/vl);
           vout2[vo2idx] = tmp;
           tmp =0;
        }
   }
    
}

void vector_mult (double  * v2, double i2, int idx)
{
   int s=vec_start_idx[idx];
   int e=vec_start_idx[idx+1];
   for (int i=s;i<e;i++)
        v2[i] = v2[i] * i2;
    
}

void vector_div (double  * v2, double i2, int idx)
{
    vector_mult(v2, 1/i2, idx); 
}

int index(int lyr, int row, int col)
{
   int olyr = lyr==0?0:nodes[lyr-1];
   int ret = olyr+row*(nodes[lyr]+1)+col;
   return ret;
}

void deindex(int val, int lyr, int & row, int & col)
{
   int olyr = lyr==0?0:nodes[lyr-1];
   int deval = val-olyr;
   col = deval % (nodes[lyr]+1);
   row =  deval / (nodes[lyr]+1);
}
// Kernel - Adding two matrices MatA and MatB


__global__ void gen_matvec(double *A, double*x, double*y, const int m, const int n) 
{
  unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
  if ( xIndex < m ){
    double c = 0.0f;
    for(int i=0; i<n; i++)
      c = c + x[i] * A[xIndex + m * i];
    y[xIndex] = c;
  }
}
double matVecNaive (double * out, double * in, double * A, const int m, const int n) {

  // set up threading and blocking variables
  cudaDeviceProp dp;
  cudaGetDeviceProperties(&dp,0);
  unsigned int max_threads_per_block = dp.maxThreadsPerBlock;

  int threads_perblockm = min(m, max_threads_per_block);
  dim3 threadsPerBlockm(threads_perblockm);
  int num_blocksm = (int)ceil((double)m/(double)threads_perblockm);
  dim3 numBlocksm(num_blocksm);

  // set up timing
  cudaEvent_t start, stop;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);

  // execute kernel
  gen_matvec <<< numBlocksm, threadsPerBlockm >>>((double*)A, (double*)in, (double*)out, m, n);

  cudaThreadSynchronize();
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return time;
}

__global__ void MatMulKernel(double *out, double *in, double *a, const int matrixHeight, const int matrixWidth) {
  // get variables for loop
  // copy section of b into shared mem
  // go through the threads vertically and sum them into a variable
  // atomic add these variables to the corresponding c index

  // looping is happening horizontally on the matrix
  // BLOCK_WIDTH is again horizontal
  // BLOCK_HEIGHT is going vertical
  // n / BLOCK_WIDTH blocks horizontally
  // m / BLOCK_HEIGHT block vertically

  // get variables for loop
  // variable for loop length: blockEltHeight
  __shared__ int blockElt;
  __shared__ int blockxInd;
  __shared__ int blockyInd;
  if (threadIdx.x == 0) {
    if ((blockIdx.x + 1) * BLOCK_WIDTH <= matrixWidth)
      blockElt = BLOCK_WIDTH;
    else blockElt = matrixWidth % BLOCK_WIDTH;
    blockxInd = blockIdx.x * BLOCK_WIDTH;
    blockyInd = blockIdx.y * BLOCK_HEIGHT;
  }
  
  __syncthreads();
  
  // copy section of b into shared mem
  // use the first BLOCK_WIDTH of thread
  __shared__ double b[BLOCK_WIDTH];

  if (threadIdx.x < blockElt) 
    b[threadIdx.x] = in[blockxInd + threadIdx.x];
  
  __syncthreads();

  // summing variable
  double cSum = (double) 0;
  int threadyInd = blockyInd + threadIdx.x;

  // make sure we are inside the matrix verticallly
  if (threadyInd < matrixHeight) {
  
    // go through the threads vertically and sum them into a variable
    for (int i=0; i<blockElt; i++)
      // A col index   : blockIdx.x * BLOCK_WIDTH + i : blockxInd + i
      // A row index  : blockIdx.y * BLOCK_HEIGHT + threadIdx.x : blockyInd + threadIdx.x : threadyInd
      // B index : b[i]

      // cSum = B index * ( A col index * matrixHeight + A row index)
      cSum += b[i] * a[(blockxInd + i) * (matrixHeight) + (threadyInd)];
      //printf("csum = %f\n", cSum);
    
    // atomic add these variables to the corresponding c index
    atomicAdd(out + threadyInd, cSum);
  }
  
}


void CUDA_MatrixVectorMultiply5(int nr, int nc, double* M, double* Y, double* X)
//void CUDA_MatrixVectorMultiply5(double* Y, double* X, double* M, int nr, int nc)
//double matVecMul (double * out, double * in, double * A, const int m, const int n)
{
  // set up threading and blocking variables
  cudaDeviceProp dp;
  cudaGetDeviceProperties(&dp,0);
  unsigned int max_threads_per_block = dp.maxThreadsPerBlock;

  int threads_perblockm = min(nr, max_threads_per_block);
  dim3 threadsPerBlockm(threads_perblockm);
  int num_blocksm = (int)ceil((double)nr/(double)threads_perblockm);
  dim3 numBlocksm(num_blocksm);

  int blockCols = (int) ceil(nc / (double) BLOCK_WIDTH);
  int blockRows = (int) ceil(nr / (double) BLOCK_HEIGHT);
  dim3 dimBlock(BLOCK_HEIGHT);
  dim3 dimGrid(blockCols, blockRows);

  int sharedMem = 3 * sizeof (int) + BLOCK_WIDTH * sizeof (double);

  cudaMalloc((void**) &LayerWeightsDevice, nr*nc*sizeof(double));
  cudaMalloc((void**) &ActuationDevice, nr);
  cudaMalloc((void**) &NetinDevice, nc);

  // copy elements from CPU to GPU
  cudaMemcpy(LayerWeightsDevice, M, nr*nc*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(ActuationDevice, X, nr*sizeof(double), cudaMemcpyHostToDevice);



  // set up timing
  cudaEvent_t start, stop;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);

  // execute kernels
  //zero_vector_double<<<numBlocksm, threadsPerBlockm>>>(out, m);
  MatMulKernel<<<dimGrid, dimBlock, sharedMem>>>(NetinDevice, ActuationDevice, LayerWeightsDevice, nr, nc);

  cudaThreadSynchronize();
  cudaMemcpy(Y, NetinDevice, nc*sizeof(double), cudaMemcpyDeviceToHost);
cout << "MatMulKernel=";
for (int i=0;i<nc;i++)
  cout  << Y[i] << " ";
cout << endl;
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

//  return time;
}

double matVecMulT (double * out, double * in, double * A, const int m, const int n)
{
  // set up threading and blocking variables
  cudaDeviceProp dp;
  cudaGetDeviceProperties(&dp,0);
  unsigned int max_threads_per_block = dp.maxThreadsPerBlock;

  int threads_perblockn = min(n, max_threads_per_block);
  dim3 threadsPerBlockn(threads_perblockn);
  int num_blocksn = (int)ceil((double)n/(double)threads_perblockn);
  dim3 numBlocksn(num_blocksn);

  int blockCols = (int) ceil(n / (double) BLOCK_HEIGHT);
  int blockRows = (int) ceil(m / (double) BLOCK_WIDTH);
  dim3 dimBlock(BLOCK_HEIGHT);
  dim3 dimGrid(blockCols, blockRows);

  int sharedMem = 3 * sizeof (int) + BLOCK_WIDTH * sizeof (double);

  // set up timing
  cudaEvent_t start, stop;
  double time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);

  // execute kernels
  //zero_vector_double<<<numBlocksn, threadsPerBlockn>>>(out, n);
  MatMulKernelT<<<dimGrid, dimBlock, sharedMem>>>(out, in, A, m, n);

  cudaThreadSynchronize();
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return time;
}
__global__ void CUDA_MatrixVectorMultiply4(int nr, int nc, double* M, double* Y, double* X)
{
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
double sum = 0.0;
for (int k = 0; k < nr; k++) {
sum += M[row*nr+k] * X[k * nr + col];
}
Y[row*nr+col] = sum;
}
__global__ void CUDA_MatrixVectorMultiply3(int nr, int nc, double* M, double* Y, double* X)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < nr && j < nc)
        Y[i][j] = M[i][j] + X[i][j];
}
__global__
void CUDA_MatrixVectorMultiply2 (int nr, int nc, double* M, double* Y, double* X)
{
    int tid=threadIdx.x+blockIdx.x*blockDim.x;
        double sum=0;
    if(tid<nc){
        for(int i=0; i<nr; i++)
            sum += X[i]*M[(i*nc)+tid];
        Y[tid]=sum/nr;
    }
}
/*
void MatrixVectorMultiply2(rowvec  &n, rowvec  &a, mat  &m, double * * ret)
{   
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;
    int i, j;
    double* devPtrA;
    double* a = 0;
    int M=m.n_rows;
    int N=m.n_cols;
    a = (double*)malloc (M * N * sizeof (*a));
    if (!a) {
        printf ("host memory allocation failed");
        return EXIT_FAILURE;
    }
    memcpy(a, m.memptr(), M* N* sizeof(double));
   // for (j = 0; j < N; j++) {
   //     for (i = 0; i < M; i++) {
   //         a[IDX2C(i,j,M)] = (float)(i * N + j + 1);
   //     }
    //}
    cudaStat = cudaMalloc ((void**)&devPtrA, M*N*sizeof(*a));
    if (cudaStat != cudaSuccess) {
        printf ("device memory allocation failed");
        return EXIT_FAILURE;
    }
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }
    stat = cublasSetMatrix (M, N, sizeof(*a), a, M, devPtrA, M);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data download failed");
        cudaFree (devPtrA);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }
    modify (handle, devPtrA, M, N, 1, 2, 16.0f, 12.0f);
    stat = cublasGetMatrix (M, N, sizeof(*a), devPtrA, M, a, M);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data upload failed");
        cudaFree (devPtrA);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }
    cudaFree (devPtrA);
    cublasDestroy(handle);
    for (j = 0; j < N; j++) {
        for (i = 0; i < M; i++) {
            printf ("%7.0f", a[IDX2C(i,j,M)]);
        }
        printf ("\n");
    }
    free(a);
    return EXIT_SUCCESS;
}
*/

void matrix_multiply_cuda(rowvec & n,  rowvec & a, mat & l)
{
double * q = (double *) l.memptr();
for (int y=0;y<l.n_cols;y++)
{
  for (int x=0;x<l.n_rows;x++)
  {
      //cout << netin[i](y,x) << " ";
      cout << q[y*l.n_rows+x]  << ",";
  }
  cout << endl << flush;
}
cout << "******************************** " << endl <<  layer_weights[i].t()  << endl << flush;
exit(1);
}

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
cout << A.n_rows << "," << A.n_cols << endl;
 matVecNaive (rbuf, vbuf, Abuf, 20, 2) ;
cout << "rbuf=" << rbuf[0] << " " << rbuf[1] << endl;
cout << "r=" << r << endl;
}
#endif
