#include <iostream>
#include <iomanip>
#include <cmath>
//#include <nvblas.h>
//#include <cublas.h>
#include <chrono>
#define ARMA_ALLOW_FAKE_GCC
#include <armadillo>
#include <boost/algorithm/string.hpp>
#include<boost/algorithm/string/split.hpp>       
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
#undef SAMPLEFREQ

#define EPOCHS 1
#define EPSILON 1E-04
#define TRAININGSAMPLES 60000
#define TESTINGSAMPLES 10000
#define BLOCK_HEIGHT 1024
#define BLOCK_WIDTH 64
//nvcc --gpu-architecture=sm_35 -Wno-deprecated-gpu-targets -std=c++11 -g -Iarmadillo-10.6.2/include/ -DSERIAL_ONLY  -L armadillo-10.6.2/build/ -larmadillo  -l lapack_static  -o ann_mnist_digits_cuda_ser  ann_mnist_digits.cu
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
float mintime=1000000;
float maxtime=-10;

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

    vector<double *> layer_weights_ptr;
    vector<double *> weight_updates_ptr;
    vector<double *> new_layer_weights_ptr;
    vector<double *> netin_ptr;
    vector<double *> actuation_ptr;
    vector<double *> deltafn_ptr;
    vector<double *> ftick_ptr;

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

// implementation of the matrix-vector multiply function
void MatrixTranspVectorMultiply2(double* Y, const double* X, double* M, int m_nr, int m_nc)
{
int t_r = m_nc; // same as x_nc
int t_c = m_nr; // same as y_nc
// matrix passed in is m_nr x m_nc, need to transpose it to m_nr x m_nc 
// and its stored in as columns so can maniuplate indexes
   for (int i = 0; i < t_r; ++i) 
   {
//cout << "i="<<i<<" m_nc="<<m_nr<<" m_nr="<<m_nr<<endl;
      Y[i] = 0;
      int t=0;
      for (int j = i; j < t_c*t_r; j+=t_c) 
      {
        
         Y[i] += M[i*t_c+j] * X[t++];
         //cout << "j="<<j<<" Y{"<<i<<"] += " << M[j*m_nc+i] << "*" << X[j] << endl;
      }
   }
}
// implementation of the matrix-vector multiply function
void MatrixTranspVectorMultiply(double* Y, const double* X, double* M, int m_nr, int m_nc)
{
   for (int i = 0; i < m_nr; ++i) //m_nc == y_nc
   {
//cout << "i="<<i<<" m_nc="<<m_nr<<" m_nr="<<m_nr<<endl;
      Y[i] = 0;
      for (int j = 0; j < m_nc; ++j) //m_nr == x_nc
      {
         Y[i] += M[i*m_nc+j] * X[j];
         //cout << "j="<<j<<" Y{"<<i<<"] += " << M[j*m_nc+i] << "*" << X[j] << endl;
      }
   }
}
// implementation of the matrix-vector multiply function
void MatrixVectorMultiply(double* Y, double* X, double* M, int m_nr, int m_nc)
{
cout << "X=   ";
for (int i=0;i<m_nr;i++)
 cout << "X="<< X[i] <<endl;
cout <<endl<< "M=   ";
for (int i=0;i<m_nr*m_nc;i+=m_nc)
 cout << "M="<< M[i] <<endl;
 cout << "X="<< X[0] << " "<<X[0] << " "<< X[1] << " "<<X[2] << " "<<X[3] << " "<<X[4]<<endl;
   for (int i = 0; i < m_nc; ++i) //m_nc == y_nc
   {
//cout << "i="<<i<<" m_nc="<<m_nr<<" m_nr="<<m_nr<<endl;
      Y[i] = 0;
      for (int j = 0; j < m_nr; ++j) //m_nr == x_nc
      {
         Y[i] += M[j*m_nc+i] * X[j];
         //cout << "j="<<j<<" Y{"<<i<<"] += " << M[j*m_nc+i] << "*" << X[j] << endl;
      }
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
#ifdef SAMPLEFREQ
    if ((seq+1) % SAMPLEFREQ ==0)
    {
       cout << "For sample :" << seq+1 << endl << flush;
       print_an_image(&mptr[start], img_is_digit);
    }
#endif
    t=zeros<rowvec>(OUTPUT_LINES); // create the target vector (plus one for 'bias' bit)
    if (img_is_digit>9)
    {
       cout << "Error: img_is_digit=" << img_is_digit << "seq=" << seq  << endl;
       exit(1);
    }

    t(img_is_digit)=1;               // set the target 'bit'

//for (int i=0;i<784;i++)
//{   cout << "img=" << i << " / " << img(0) << " & " << *(img.memptr()+i) << " && " << actuation[0](i) << " * " << *(actuation[0].memptr()+i) << *(actuation_ptr[0]+i) << endl;
//}
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
#ifdef SAMPLEFREQ
             if ( (y0+1) % SAMPLEFREQ == 0) 
                cout << "---------------------------------- BACK PROPAGATION  sample=" << y0+1 <<" err=" << err << " < epsilon, for tgt '"<< val <<"' so error is acceptable, returning" << endl << flush;
#endif
             err_summary(val) = err;
             return 1;
        }

#ifdef SAMPLEFREQ
        if ( (y0+1) % SAMPLEFREQ == 0) 
          cout << "------------------------------------ BACK PROPAGATION sample="<< y0+1 << endl << flush;
#endif        
        ftick[OutputLayer] = -actuation[OutputLayer] + 1;
        ftick[OutputLayer] = ftick[OutputLayer] % (actuation[OutputLayer]);  //element wise multiply
        deltafn[OutputLayer]  =  (tgt0 - actuation[OutputLayer])%(ftick[OutputLayer]);

        for (int i=OutputLayer-1;i>=0;i--)
        {
            colvec c=deltafn[i+1].t();
            weight_updates[i]  =  c * actuation[i];
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
#ifdef SAMPLEFREQ
        if ( (y+1) % SAMPLEFREQ == 0)
           cout << "------------------------------------ FORWARD FEED OF "<<intype <<" SAMPLE # "<< y+1 << endl << flush;
#endif
        load_an_image(y, imgdata, actuation[0], tgt, labdata);
//cout << "Act[0]ptr=" << actuation_ptr[0] << endl;
//for (int h=0;h<785;h++)
//cout << "Act["<<h<<"]=" << *(actuation[0].memptr()+h) << endl;
        int tgtval = tgt.subvec(0,9).index_max();

        for (int e=0;e<epochs;e++)
        {
            for (int i=0;i<OutputLayer;i++)  // only n-1 transitions between n layers
            {
               // cout << "------------------------------------ All inputs into L" << i << endl << flush;
                // sum layer 1 weighted input
#ifdef SERIAL_ONLY
#if 0
double d[20];
               mat a(d,1,4,false,true);
a={{10,20,30,40}};
               mat c={{1,2},{3,4},{5,6},{7,8}};
               mat b=c.t();
double * m= b.memptr();

for (int q=0;q<8;q++)
   cout << "b[" << q<< "]="<< *(m+q) << endl;
m=c.memptr();
for (int q=0;q<8;q++)
   cout << "c[" << q<< "]="<< m[q] << endl;
double *p=layer_weights_ptr[i];
                for (int k=0;k< b.n_rows; k++)
                  for (int j=0;j<b.n_cols; j++)
{
                     p[k*b.n_cols+j]=b(k,j);
cout << "p{"<< k*b.n_cols+j << " == " <<  p[k*b.n_cols+j] << endl;
}
                for (int j=0;j< a.n_cols; j++)
                  actuation_ptr[i][j]=a(j);
                MatrixTranspVectorMultiply(d, a.memptr(), b.memptr(), b.n_cols,  b.n_rows);
cout <<"d=" << d[0] << " " << d[1] << endl;
cout <<"a=" << a << endl;
cout <<"b=" << b << endl;
cout <<"c=" << c << endl;
cout <<"a*b=" << d[0] << " " << d[1] << endl;
                MatrixTranspVectorMultiply(d, a.memptr(), c.memptr(), b.n_cols,  b.n_rows);
cout <<"a=" << a << endl;
cout <<"b=" << b << endl;
cout <<"c=" << c << endl;
cout <<"a*b=" << d[0] << " " << d[1] << endl;
exit(1);
///////////////////////////////////////////////////////
                netin[i] =  (actuation[i] * layer_weights[i].t())/actuation[i].n_cols;
///////////////////////////////////////////////////////
//                    cout << "Netin serial ("<<  netin[i].n_rows << "," <<  netin[i].n_cols << ")= "  << netin[i] << endl << flush;
                for (int k=0;k< layer_weights[i].n_rows; k++)
                  for (int j=0;j<layer_weights[i].n_cols; j++)
                     layer_weights_ptr[i][k*layer_weights[i].n_cols+j]=layer_weights[i](k,j);
                for (int j=0;j< actuation[i].n_cols; j++)
                  actuation_ptr[i][j]=actuation[i](j);
#endif
                double dd[50000];
                mat c(dd, layer_weights[i].n_cols, layer_weights[i].n_rows, false, true);
                c=layer_weights[i].t();
                //MatrixVectorMultiply(netin[i].memptr(), actuation[i].memptr(), c.memptr(), c.n_rows,  c.n_cols);
                MatrixVectorMultiply(netin_ptr[i], actuation[i].memptr(), c.memptr(), c.n_rows,  c.n_cols);
             //   MatrixTranspVectorMultiply(netin[i].memptr(), actuation[i].memptr(), layer_weights[i].memptr(), layer_weights[i].n_rows,  layer_weights[i].n_cols);
                for (int j=0;j< netin[i].n_cols; j++)
                {
                  cout <<"net= j"<< j << "  "<<*(netin_ptr[i]+j) <<  "**" << netin[i](j)  << "***" << *(netin[i].memptr()+j) << endl;
            //      netin[i](j) = netin_ptr[i][j]/actuation[i].n_cols;
                }
cout << "act["<<i<<"]=" << actuation[i] << endl;
mat z= layer_weights[i].t();
cout << "lay_wg=" << z.col(0) << endl;
cout << "netin[i]="<< netin[i] << endl;
exit(1);
#ifdef SAMPLEFREQ
                if ( (y+1) % SAMPLEFREQ == 0)
                    cout << "Netin serial ("<<  netin[i].n_rows << "," <<  netin[i].n_cols << ")= "  << netin[i] << endl << flush;
#endif
#else
   //if (t > maxtime)
   //  maxtime=t;
   //if (t < mintime )
   //  mintime=t;

#endif    
//                for (int j=0;j< netin[i].n_cols; j++)
  //              {
            //      cout <<"net= j"<< j << "  "<<*(netin_ptr[i]+j) <<  "**" << netin[i](j)  << "***" << *(netin[i].memptr()+j) << endl;
    //              netin[i](j) = netin_ptr[i][j]/actuation[i].n_cols;
      //          }
                
                    cout << "Netin serial ("<<  netin[i].n_rows << "," <<  netin[i].n_cols << ")= "  << netin[i] << endl << flush;

                actuation[i+1] = sigmoid(netin[i]);
            }
#ifdef SAMPLEFREQ
            if ( (y+1) % SAMPLEFREQ == 0)
            {
               std::cout << "Final output : " << endl <<   std::setw(7) << fixed << showpoint << actuation[OutputLayer].subvec(0,9) << " Sample: " << y+1 <<std::endl << flush;
               std::cout << "Expec output : " << endl  <<  std::setw(7) << fixed << showpoint << tgt.subvec(0,9) << " Sample: " << y+1 << std::endl << flush;
            }
#endif            
                    //////////////////////////// forward feed end
            if (train)
            {
                 // printout intermediate result
                  int outval = actuation[OutputLayer].subvec(0,9).index_max();
#ifdef SAMPLEFREQ
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
#endif
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
        if (!train ) 
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


#if 0
float  randw[10][1000][1000];
void read_weights()
{

    float x;
    int r;
    vector<string> strs;
    std::string type="";
    std::string line;

    while (line.substr(0,6) != "layers")
    {
      getline(wgts, line);
    }
    boost::split(strs,line,boost::is_any_of(":"));
    int nolayers=stoi(strs[1]);

#if 0
#ifndef orig
float flip=-1.0;
#endif
#endif
      getline(wgts, line);
    for (int z=0;z<nolayers;z++)
    {
      boost::split(strs,line,boost::is_any_of(":"));
      int layer=stoi(strs[1]);
      int rows=stoi(strs[3]);
      int cols=stoi(strs[5]);
      cout << layer << " "<< rows << " " << cols<<endl;

        getline(wgts, line);
      for (r=0;r<rows;r++)
      {
    std::istringstream in(line);      //make a stream for the line itself
        for (int c=0;c<cols;c++)
        {
          in >> x;       //now read the whitespace-separated floats
#if 0
#ifndef orig
       x=x*flip;
       flip=-flip;
#endif
#endif
          randw[z][r][c]=x;
//  cout << "randw["<<z<<"]["<<r<<"]["<<c<<"]"<<"="<< randw[z][r][c]<<endl;
        }
        getline(wgts, line);
      }
    }

}
#endif
double l2[10][50000];
int nd[100];
int nd2[100];
int lays;
   int t=0;
 int x=0;
void load_weights(string fname)
{
    ifstream iFile;
    cout << "Loading weights from file : " << fname << endl << flush;
    iFile.open(fname, ios::in);
    string aline;
  
     vector<string> strs;

    if (fname.substr(0,4)=="post")
       stringstream confusion_matrix2;
    getline(iFile, aline);
   boost::split(strs, aline, boost::is_any_of("="));
   if (strs.size() > 1)
      lays=stoi(strs[1]);
   cout<< " Has " << lays << "layers" << endl;
//    cout << aline;
/*
    getline(iFile, aline);
   boost::split(strs, aline, boost::is_any_of("="));
   if (strs.size() > 1)
      nd=stoi(strs[1]);
   cout<< " Has " << nd << " nodes" << endl;
*/
  while (iFile.good()) 
  {
       getline(iFile, aline);
    if (aline.find("NodesInLayer") != std::string::npos)
    {
      nd2[t]=x;
      x=0;
      t++;
      
   boost::split(strs, aline, boost::is_any_of("="));
   if (strs.size() > 1)
      nd[t]=stoi(strs[1]);
   cout<< " Has " << nd[t] << "layers" << endl;
    }
    else if ((aline.find("Error Summary") != std::string::npos) )
    {
       nd2[t]=x;
      x=0;
      t++;
      return;
    }
    else if (aline.find("LayerBiases") == std::string::npos)
    {
       boost::trim(aline);
       boost::split(strs, aline, boost::is_any_of(" "));
       boost::algorithm::split(strs,aline,boost::is_any_of("\t "),boost::token_compress_on);
      for (int y=0;y< strs.size();y++)
      {
        if (strs[y].length() > 0)
        {
           l2[t][x++]=stod(strs[y]);
           cout << y << ":" << l2[t][x-1] << endl;
       }
     }
    }
  }
}
//1:NumberOfLayers=3
//2:NodesInLayer0=784
//35:NodesInLayer1=30
//48:Error Summary
//51:EndFile

#if 0
string n1,n2,n3;
double tmp1,tmp2;
  iFile >> n1 >> endl  >> n2 >> endl;
while (iFile.good()) {
  while (! iFile.eoln())
  {
    iFile >> tmp1;
    cout <<tmp1 << endl;
  }
 iFile >> n1;  
cout  <<n1 <<  endl;
}
	    oFile >> "NumberOfLayers=" >> NumberOfLayers >> endl << flush;
    for (int i=0; i< OutputLayer; i++)
    {

        oFile <<  "NodesInLayer"<<i<<"=" << nodes[i] << endl << flush;
        oFile << layer_weights[i] << endl << flush;
    }
    oFile << "Error Summary" << endl << flush;

    oFile << err_summary << endl << flush;

    oFile << "EndFile" << endl << flush;
    oFile.close();
#endif


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
//string y="initial_random_values_weights_11337071.txt";
string y="initial_random_values_weights_1636167104.txt";

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
         double * rbptr = new double [ nodes[i]+1];
         rowvec rb (rbptr, nodes[i]+bias_field, false, true);
         actuation.push_back(rb); 
         actuation_ptr.push_back(rbptr);
         double * drbptr = new double [ nodes[i]+1];
         rowvec drb (drbptr, nodes[i]+bias_field, false, true);
         deltafn.push_back(drb);
         deltafn_ptr.push_back(drbptr);

         double * frbptr = new double [ nodes[i]+1];
         rowvec frb (frbptr, nodes[i]+bias_field, false, true);
         ftick.push_back(frb);
         ftick_ptr.push_back(frbptr);

         if (i<OutputLayer)
         {
            max_mat=max(max_mat, (nodes[i]+bias_field)*(nodes[i+1]+bias_field));
            double * tmpptrr = new double [nodes[i+1]+bias_field];
            rowvec rb2 (tmpptrr, nodes[i+1]+bias_field, false, true);

            netin.push_back(rb2);   // size=nodes[i],1
            netin_ptr.push_back(tmpptrr);

            double  *tmpptr  = new double [ (nodes[i+1]+1) * (nodes[i]+1) ];
            mat tmpwgt (tmpptr, nodes[i+1]+1 , nodes[i]+1, false, true); // network weights for each node + 1 node bias weight

            double  *tmpptr0  = new double [ (nodes[i+1]+1) * (nodes[i]+1) ];
            mat tmpwgt0 (tmpptr0, nodes[i+1]+1 , nodes[i]+1, false, true); // network weights for each node + 1 node bias weight

            double  *tmpptr00  = new double [ (nodes[i+1]+1) * (nodes[i]+1) ];
            mat tmpwgt00 (tmpptr00, nodes[i+1]+1 ,nodes[i]+1, false, true); // network weights for each node + 1 node bias weight

             mat rmpwgt = randu<mat>( nodes[i+1]+1,nodes[i]+1); // network weights for each node + 1 node bias weight
             mat rmpwgt0 = zeros<mat>( nodes[i+1]+1,nodes[i]+1); // network weights for each node + 1 node bias weight
             mat rmpwgt00 = zeros<mat>( nodes[i+1]+1,nodes[i]+1); // network weights for each node + 1 node bias weight

             tmpwgt = rmpwgt;
             tmpwgt0 = rmpwgt0;
             tmpwgt00 = rmpwgt00;

            layer_weights.push_back( tmpwgt );
            layer_weights_ptr.push_back(tmpptr);

            new_layer_weights.push_back(tmpwgt0);
            new_layer_weights_ptr.push_back(tmpptr0);

            weight_updates.push_back(tmpwgt00);
            weight_updates_ptr.push_back(tmpptr00);
cout << "*********** Y(" << netin[i].n_rows << "x" << netin[i].n_cols << ") = X(" << actuation[i].n_rows << "x" << actuation[i].n_cols << ") TIMES M(" << layer_weights[i].n_rows << "x" <<  layer_weights[i].n_cols << ")" << endl; 

          }
    }
    save_weights("yyyinitial_random_values");
   cout << "Max Matrix size " << max_mat << " Max vector size = " << max_vec << endl << flush;
   cout << "vector lens=" << netin.size() <<"," <<layer_weights.size() << "," <<actuation.size() << endl;
////////////////////*
/*
double l2[10][50000];
int nd[100];
int nd2[100];
Have 3 layers t=3
 Layer 0 has 0 nodes and 0
 Layer 1 has 784 nodes and 24335
 Layer 2 has 30 nodes and 341*/
////////////
load_weights(y);
for (int i=0;i<lays;i++)
{
  for (int j=0;j<nd2[i+1];j++)
  {
if (j==0) cout << (j+1) << "/" << nd[i+1]+2 << " ==0??" << endl;
    int r=(j)/(nd[i+1]+1); 
    int c=(j) % (nd[i+1]+1);
    layer_weights[i](r,c) =  l2[i][j];
  }
}
    save_weights("xxxinitial_random_values");
/*
cout << "Have " << lays << " layers t="<< t << endl;
for (int i=0;i<lays;i++)
  cout << " Layer " << i << " has " << nd[i] << " nodes and " << nd2[i] << endl;
*/
exit(0);
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
