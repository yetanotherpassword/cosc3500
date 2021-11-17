#include <iomanip>
#include <cmath>
#include <chrono>
//#include <boost/algorithm/string.hpp>
//#include <boost/algorithm/string/split.hpp>
#include <vector>
#include <limits>
#include <sstream>
#include <fstream>
#include <iostream>
#include <string>
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

std::chrono::microseconds Process_MaxTime = std::chrono::microseconds::min();
std::chrono::microseconds Process_MinTime = std::chrono::microseconds::max();
std::chrono::microseconds Call_MaxTime = std::chrono::microseconds::min();
std::chrono::microseconds Call_MinTime = std::chrono::microseconds::max();
std::chrono::microseconds Avg_Time;
int avgcnt=0;

#ifndef SERIAL_ONLY
double *LayerWeightsDevice;
double *ActuationDevice;
double *NetinDevice;
cudaEvent_t start, stop;
int tile_dimension = 8; 
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


#define TILE_DIM 16                     // Tile dimension


class newmat {
public:
   double * ptr;
   int n_rows;
   int n_cols;
   newmat(int r, int c)
   {
       n_rows=r;
       n_cols=c;
       ptr=new double [r*c];
   };
   string prtstr()
   { 
       string s="";
       for (int i=0;i<n_rows;i++)
       {
          for (int j=0;j<n_cols;j++)
		   s+= "   " + to_string(ptr[i*n_cols+j]);
	  s+= '\n';
       }
       return s;
   };
   void free_ele()
   {
       if (ptr != NULL)
          delete [] ptr;
   };
   void zeroize()
   {
       for (int i=0;i<n_rows;i++)
       {
           for (int j=0;j<n_cols;j++)
		  ptr[i*n_cols+j]=0.0;
       }
   };
   double * memptr()
   {
       return ptr;
   };
   int index_max_row(int r, int start, int stop)
   {
        int idx=0;
        double max=  std::numeric_limits<double>::min();
        if (((r<n_rows) && (r>=0)) && (start>=0) && (start < n_cols) && (stop >=0) && (stop<n_cols) && (start<=stop))
          for (int i =r; i<=r;i++)
             for (int j =start; j<=stop;j++)
               if (ptr[i*n_cols+j] > max)
               {
                  idx=i*n_cols+j;
                  max =  ptr[i*n_cols+j];
               }
        return idx;
   };

 } ;
 
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

void PreMatMul(newmat & a, newmat & b, newmat & c)
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

    auto StartChronoTime = std::chrono::high_resolution_clock::now();

                   auto StartCallTime = std::chrono::high_resolution_clock::now();

    MatMulNoShared<<<dimGrid , dimBlock>>>(deviceA , deviceB , deviceC , ARows , ACols, BRows ,BCols , CRows , CCols);
	
    checkError(cudaDeviceSynchronize());


    auto EndCallTime = std::chrono::high_resolution_clock::now();
    auto TotalCallTime = std::chrono::duration_cast<std::chrono::microseconds > (EndCallTime - StartCallTime);


    if (TotalCallTime > Call_MaxTime)
        Call_MaxTime = TotalCallTime;

    if (TotalCallTime < Call_MinTime)
        Call_MinTime = TotalCallTime;


    cudaMemcpy(hostC, deviceC, DIMX*DIMZ*sizeof(double), cudaMemcpyDeviceToHost);
    for (int j=0;j<31;j++)
      cout << hostC[j] << " " ;
    cout << endl;

    memcpy(c.memptr(), hostC, DIMX*DIMZ*sizeof(double));

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////





std::time_t result = std::time(nullptr);
string fid = to_string(result);
unsigned int NumberOfLayers;
unsigned int OutputLayer;
unsigned int *nodes;
double eta;	// Learning factor

vector<newmat> netin;
vector<newmat> actuation;
vector<newmat> deltafn;
vector<newmat> ftick;
vector<newmat> layer_weights;
vector<newmat> weight_updates;
vector<newmat> new_layer_weights;


ios init(NULL);
stringstream confusion_matrix;
newmat err_summary(1,OUTPUT_LINES);


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




// implementation of the matrix-vector multiply function
void SerialMatrixVectorMultiply(double *Y, double *X, double *M, int m_nr, int m_nc)
{
  // Need to ensure Y vector passed has been zeroised
    for (int i=0;i<m_nr*m_nc ;i++)
    {
        int c1=i % m_nc;
        int r1=i / m_nc;
        Y[c1] += X[r1] * M[c1 *m_nr + r1];
    }
}

void sigmoid3(newmat & net, newmat & out)
{
   int c=net.n_cols;
   for (int i=0;i<c;i++)
     out.ptr[i] = 1 / (1 + exp(-net.ptr[i]));
   out.ptr[c] = 1.0;	// add bias signal value
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

void load_an_image(int seq, unsigned char* &mptr, newmat &img, newmat &t,
     unsigned char* &lp)
{
     int start = (INPUT_LINES *seq) + IMAGE_OFFSET;
     double greyval = MAX_PIXEL_VAL;

     for (int i = 0; i < INPUT_LINES; i++)
     {
          img.ptr[i] = ((double) mptr[start + i]) / greyval;
     }

     img.ptr[nodes[0]] = 1;      // set bias signal, so can multiply with[node weights |
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
void output(newmat t)
{
     cout << t.prtstr();
}
     
double accu(newmat m1)
{
  double tmp=0;
    for (int i=0;i<m1.n_rows;i++)
      for (int j=0; i<m1.n_cols;j++)
         tmp += m1.ptr[i*m1.n_cols+j];
  return tmp;
}    
newmat diff (newmat p1, newmat p2)
{
  newmat tmp(p1.n_rows, p1.n_cols);
  for (int i=0;i<p1.n_rows;i++)
	 {
        for (int j=0;j<p1.n_cols;j++)
		   tmp.ptr[i*p1.n_cols+j]=p1.ptr[i*p1.n_cols+j] - p2.ptr[i*p1.n_cols+j];
	}
	return tmp;
}
newmat piecewisemult (newmat p1, newmat p2)
{
  newmat tmp(p1.n_rows, p1.n_cols);
  for (int i=0;i<p1.n_rows;i++)
	 {
        for (int j=0;j<p1.n_cols;j++)
		   tmp.ptr[i*p1.n_cols+j]=p1.ptr[i*p1.n_cols+j] * p2.ptr[i*p1.n_cols+j];
	}
	return tmp;
}
newmat matmult (newmat p1, newmat p2)
{
  newmat tmp(p1.n_rows, p2.n_cols);
  if (p1.n_cols == p2.n_rows)
  {
     for (int i=0;i<p1.n_rows;i++)
	 {
        for (int j=0;j<p1.n_cols;j++)
		   tmp.ptr[i*p1.n_cols+j]=p1.ptr[i*p1.n_cols+j] * p2.ptr[i*p1.n_cols+j];
	 }
  }
  return tmp;
}
newmat mult (newmat p1, double p2)
{
  newmat tmp(p1.n_rows, p1.n_cols);
  for (int i=0;i<p1.n_rows;i++)
	 {
        for (int j=0;j<p1.n_cols;j++)
		   tmp.ptr[i*p1.n_cols+j]=p1.ptr[i*p1.n_cols+j] * p2;
	}
	return tmp;
}
newmat add (newmat p1, double p2)
{
  newmat tmp(p1.n_rows, p1.n_cols);
  for (int i=0;i<p1.n_rows;i++)
	 {
        for (int j=0;j<p1.n_cols;j++)
		   tmp.ptr[i*p1.n_cols+j]=p1.ptr[i*p1.n_cols+j] + p2;
	}
	return tmp;
}
newmat matadd (newmat p1, newmat p2)
{
  newmat tmp(p1.n_rows, p1.n_cols);
  for (int i=0;i<p1.n_rows;i++)
	 {
        for (int j=0;j<p1.n_cols;j++)
		   tmp.ptr[i*p1.n_cols+j]=p1.ptr[i*p1.n_cols+j] + p2.ptr[i*p1.n_cols+j];
	}
	return tmp;
}
int backprop(newmat & tgt, int y0)
{

     newmat final = actuation[OutputLayer];
     final.n_cols--;
     newmat tgt0 = tgt;
     //tgt0.insert_cols(nodes[OutputLayer], 1);
     double err = accu(piecewisemult(diff(tgt , final) , diff(tgt , final))) *0.5;
     if (abs(err) < EPSILON)
     {
          int val = tgt0.index_max_row(0,0,9);
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
     ftick[OutputLayer] = add( mult(actuation[OutputLayer], -1.0)  , 1.0);
     ftick[OutputLayer] =
          piecewisemult(ftick[OutputLayer] , actuation[OutputLayer]);	// element wise multiply
     deltafn[OutputLayer] = piecewisemult(diff(tgt0 , actuation[OutputLayer]) , ftick[OutputLayer]);

     for (int i = OutputLayer - 1; i >= 0; i--)
     {
          weight_updates[i] =  matmult(deltafn[i + 1], actuation[i]);
          new_layer_weights[i]  = matadd( matmult(layer_weights[i], weight_updates[i]),  mult(weight_updates[i], eta) );
          ftick[i] = add(mult(actuation[i] ,(-1.0)) , 1.0);
          ftick[i] = piecewisemult(ftick[i] , actuation[i]);	// element wise multiply
          deltafn[i] = matmult(deltafn[i + 1] ,layer_weights[i]);
          deltafn[i] = piecewisemult(deltafn[i] , ftick[i]);
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
     newmat tgt(1,OUTPUT_LINES+1);
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
                   auto StartCallTime = std::chrono::high_resolution_clock::now();



                   SerialMatrixVectorMultiply(netin[i].ptr, 
                                               actuation[i].ptr, 
                                               layer_weights[i].ptr,  layer_weights[i].n_rows,  layer_weights[i].n_cols);

                   auto EndCallTime = std::chrono::high_resolution_clock::now();
                   auto TotalCallTime = std::chrono::duration_cast<std::chrono::microseconds > 
                                                                       (EndCallTime - StartCallTime);

                   if (TotalCallTime > Call_MaxTime)
                      Call_MaxTime = TotalCallTime;

                   if (TotalCallTime < Call_MinTime)
                      Call_MinTime = TotalCallTime;

                   Avg_Time += TotalCallTime;
                   avgcnt++;

                    for (int j = 0; j < netin[i].n_cols; j++)
                    {
                          netin[i].ptr[j] /= actuation[i].n_cols;
                    }
#ifdef SAMPLEFREQ
                    if ((y + 1) % SAMPLEFREQ == 0)
                         cout << "Netin serial (" << netin[i].n_rows << "," << netin[i].n_cols <<
                         ")= " << netin[i].prtstr("NetIn Serial") << endl << flush;
#endif
#else
                   PreMatMul(actuation[i], layer_weights[i], netin[i]);

#ifdef SAMPLEFREQ
                         cout << "Netin Parallel " << netin[i].n_rows << "," << netin[i].n_cols <<
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
    auto StartChronoTime = std::chrono::high_resolution_clock::now();

     vector<string> strs;
     string bldver = string(__DATE__) + " at time " + string(__TIME__);
     cout << "--------------------------------  Build done on " << bldver << endl <<
          flush;
     init.copyfmt(cout);
     for (int i=0;i<err_summary.n_cols;i++)
        err_summary.ptr[i] = -1.0;
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
/*
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
	 */

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
          newmat rb3a(1, nodes[i] + bias_field);
          actuation.push_back(rb3a);

          newmat drb3(1, nodes[i] + bias_field);
          deltafn.push_back(drb3);

          newmat frb3(1, nodes[i] + bias_field);
          ftick.push_back(frb3);

          if (i < OutputLayer)
          {
               max_mat =
                    max(max_mat, (nodes[i] + bias_field) *(nodes[i + 1] + bias_field));
               // These buffers for the rowvec and mat structures below are done to ensure
               // the Armadillo matrix can be accessed directly and the library doesnt move
               // the memory around
               newmat rb3(1,( nodes[i+1] + bias_field));

               // Create an array of matrices (one element for each layer) for the netin value
               // This holds the sum of weighted signals, for each node, that gets squashed to 
               // produce the nodes output for next layer
               netin.push_back(rb3);
               // Create a buffer of required size for weights, in each layer
               // (plus two more, one for delta updates, and one for holding new weight to be
               // applied after backprop. These maybe consolidated later
               newmat tmpwgt3((nodes[i + 1] + bias_field),( nodes[i] + bias_field));
               for (int p=0;p<(nodes[i + 1] + bias_field)*( nodes[i] + bias_field);p++)
                  tmpwgt3.ptr[p] = rand()/RAND_MAX;
               newmat tmpwgt30((nodes[i + 1] + bias_field),( nodes[i] + bias_field));
               newmat tmpwgt300((nodes[i + 1] + bias_field),( nodes[i] + bias_field));
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
 cout << "CudaMalloc1" << endl << flush;
     checkError(cudaMalloc(&ActuationDevice, max_vec* sizeof(double)));
	 cout << "CudaMalloc2" << endl << flush;
     checkError(cudaMalloc(&NetinDevice, max_vec* sizeof(double)));
	 cout << "CudaMalloc3" << endl << flush;
     checkError(cudaMalloc(&LayerWeightsDevice, max_mat* sizeof(double)));
	 cout << "CudaMalloc4" << endl << flush;
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

     cout << "Min time for " <<  build_type << " call : " << Call_MinTime.count() << " us" << endl;
     cout << "Min time for " <<  build_type << " call : " << Call_MinTime.count()/1000000  << " s" << endl;
     cout << "Min time for " <<  build_type << " call : " << Call_MinTime.count()/60000000  << " min" << endl;
     cout << "Avg time for " <<  build_type << " call : " << (double) Avg_Time.count()/(double) avgcnt << " us" << endl;
     cout << "Avg time for " <<  build_type << " call : " << (double) Avg_Time.count()/(double) (avgcnt *1000000)  << " s" << endl;
     cout << "Avg time for " <<  build_type << " call : " << (double) Avg_Time.count()/(double) (avgcnt * 60000000)  << " min" << endl;
     cout << "Max time for " <<  build_type << " call : " << Call_MaxTime.count() << " us" << endl;
     cout << "Max time for " <<  build_type << " call : " << Call_MaxTime.count()/1000000  << " s" << endl;
     cout << "Max time for " <<  build_type << " call : " << Call_MaxTime.count()/60000000  << " min" << endl;
     auto EndChronoTime = std::chrono::high_resolution_clock::now();
     auto TotalChronoTime = std::chrono::duration_cast<std::chrono::microseconds > (EndChronoTime - StartChronoTime);

     cout << "Time for  Total Program : " << TotalChronoTime.count() << " us " << endl;
     cout << "Used Tile Dimension of " << tile_dimension << endl;

     for (int i=0;i<netin.size();i++)
         netin[i].free_ele();


#ifndef SERIAL_ONLY
     checkError(cudaFree(LayerWeightsDevice));
     checkError(cudaFree(ActuationDevice));
     checkError(cudaFree(NetinDevice));

     checkError(cudaEventDestroy(start));
     checkError(cudaEventDestroy(stop));
#endif


}

