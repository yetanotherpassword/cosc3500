#include <iostream>
#include <iomanip>
#include <cmath>
//#include <nvblas.h>
//#include <cublas.h>
#include <chrono>
#define ARMA_ALLOW_FAKE_GCC
#include <armadillo>
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
#undef SAMPLEFREQ

#define EPOCHS 64
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
        int tgtval = tgt.subvec(0,9).index_max();

        for (int e=0;e<epochs;e++)
        {
            for (int i=0;i<OutputLayer;i++)  // only n-1 transitions between n layers
            {
               // cout << "------------------------------------ All inputs into L" << i << endl << flush;
                // sum layer 1 weighted input
#ifdef SERIAL_ONLY
                netin[i] =  (actuation[i] * layer_weights[i].t())/actuation[i].n_cols;
#ifdef SAMPLEFREQ
                if ( (y+1) % SAMPLEFREQ == 0)
                    cout << "Netin serial ("<<  netin[i].n_rows << "," <<  netin[i].n_cols << ")= "  << netin[i] << endl << flush;
#endif
#else
                
   netin[i] = netin[i]/actuation[i].n_cols; 
   if (t > maxtime)
     maxtime=t;
   if (t < mintime )
     mintime=t;

#endif    

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

            mat tmpwgt = randu<mat>( nodes[i+1]+1,nodes[i]+1); // network weights for each node + 1 node bias weight
            mat tmpwgt0 = zeros<mat>( nodes[i+1]+1,nodes[i]+1); // network weights for each node + 1 node bias weight
            mat tmpwgt00 = zeros<mat>( nodes[i+1]+1,nodes[i]+1); // network weights for each node + 1 node bias weight
            layer_weights.push_back( tmpwgt );
            new_layer_weights.push_back(tmpwgt0);
cout << "i="<< i << " and weight_updates has ("<< tmpwgt00.n_rows <<"x"<<tmpwgt00.n_cols << ")"<<endl;
            weight_updates.push_back(tmpwgt00);
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
