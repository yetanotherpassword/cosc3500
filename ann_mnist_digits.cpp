#include <iostream>
#include <iomanip>
#include <cmath>
#include <armadillo>
#include <boost/algorithm/string.hpp>
#include <chrono>

//#include <stdlib.h>
//#include <stdio.h>
#include <ctime>
 
#undef EXTRA_OUTPUT
#define ARMA_64BIT_WORD

#define INPUT_LINES 784
#define DEFAULT_HIDDEN 30
#define OUTPUT_LINES 10

#define MATRIX_SIDE 28
#define MAX_PIXEL_VAL 255.0f
#define IMAGE_OFFSET 16

#define ETA_DEFAULT -1e-3
#define MOMENTUM 0.9f
#define EPSILON 1e-3
#define EPOCHS 512
#undef STOP_AT_EPSILON

#define USE_BIASES
#define EXTRA_MESSAGE ""


/*
 * ALLAN CAMPTON
 * COSC3500 Milestone 1 Serial Version
 *
 * To perform a full build and run from scratch, do the following
 *
 *    git clone git@github.com:yetanotherpassword/cosc3500 --branch="Milestone1"
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

 
using namespace arma;
using namespace std;

const double epsilon = 1e-3;
unsigned int NumberOfLayers;
unsigned int * nodes;
unsigned int OLayer;         // Output Layer as index to NumberOfLayers
double eta;               // Learning factor
vector<rowvec> netin;
#ifdef USE_BIASES
vector<rowvec> layer_biases;
#endif
vector<rowvec> actuation;
vector<rowvec> deltafn;
vector<rowvec> theta;
vector<rowvec> ftick;
vector<mat> layer_weights;
vector<mat> weight_updates;
vector<mat> new_layer_weights;
rowvec input; 

std::time_t result = std::time(nullptr);
string fid = to_string(result);

//rowvec err_summary=ones<rowvec>(OUTPUT_LINES) * (-1);
rowvec sigmoid( rowvec  & net)
{
   rowvec out = 1/(1+exp(-net));
   return out;
}

/////////////////////////////////////////////
//
// DEBUGGING ROUTINES
//
void print_an_image(unsigned char * c, int i)
{
     cout << "This is a : " << i << endl << flush;
     for (int i=0;i<INPUT_LINES;i++)
     {
       if (i%MATRIX_SIDE==0)
         cout << endl << flush;
       cout  << hex << std::setfill('0') << std::setw(2) << (unsigned int)c[i] << dec << " " << flush;
     }
     cout << setfill(' ') << endl << flush;
}
   

void print_images(unsigned char * c,  int size)
{
    for (int i=IMAGE_OFFSET;i<size;i++)
    {
       if (((i-IMAGE_OFFSET)%MATRIX_SIDE)==0)
           cout << endl << flush;
       if (((i-IMAGE_OFFSET)%INPUT_LINES)==0)
           cout << endl << "Image : " << dec << ((i-IMAGE_OFFSET)/INPUT_LINES)+1 << endl << flush;
       cout << hex << std::setfill('0') << std::setw(2) << (unsigned int)c[i] << " " << flush;
    }
    cout << setfill(' ') << flush;
}

void outp(mat m, string s)
{
   cout << "matrix:" << s << " rows=" << m.n_rows <<   " cols=" << m.n_cols << endl << flush;

}

void outp(rowvec v, string s)
{
   cout << "vector:" << s << " rows=" << v.n_rows <<   " cols=" << v.n_cols << endl << flush;

}
//
//
/////////////////////////////////////////////

void load_file(string filename, string labels, unsigned char * * labs, unsigned char * * data)
{
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
        *data = new unsigned char [size];
        inFile.seekg (0, ios::beg);
        inFile.read ((char *)*data , size);
        inFile.close();

        cout << "the entire file content is in memory, all " << size << " bytes of it" << endl << flush;
         //print_images(memblock, size);
 
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
        // print_images(memblock, size);
 
    }
    inFile.close();

}

void load_an_image(int seq, unsigned char * &mptr, rowvec & img, rowvec & t, unsigned char * &lp)
{
    int start=(INPUT_LINES*seq)+IMAGE_OFFSET;
    double greyval=MAX_PIXEL_VAL;
    //img.set_size(INPUT_LINES+1);
    //img(INPUT_LINES)=1;          // set bias signal, so can multiply with [node weights | bias weights] augmented matrix
    img.set_size(INPUT_LINES);
    for (int i=0;i<INPUT_LINES;i++)
    {
        img(i) = ((double ) mptr[start+i])/greyval;
      /* 
        if (mptr[start+i] == 0)
           img(i)=0.0;
        else
           img(i)=1.0;
     */ 
    }
     //   cout << img << endl << "an image ************************" << endl << flush;

    int img_is_digit=(int) lp[8+seq];

#ifdef EXTRA_OUTPUT 
    print_an_image(&mptr[start], img_is_digit);
#endif

//    t=zeros<rowvec>(OUTPUT_LINES+1); // create the target vector (plus one for 'bias' bit)
//    t(t.n_cols-1)=1;                 // set bias signal (redundant for target, but keeps vectors same size)

    t=zeros<rowvec>(OUTPUT_LINES); // create the target vector (plus one for 'bias' bit)
    t(img_is_digit)=1;               // set the target 'bit'

}

int backprop(rowvec tgt, int s, int e)
{
        double err = accu((tgt - actuation[OLayer]) %  (tgt - actuation[OLayer]))*0.5;
#ifdef STOP_AT_EPSILON
        if (err < epsilon)
        {
             int val=tgt.index_max();
             cout << "---------------------------------- BACK PROPAGATION  err=" << err << " < epsilon (" << epsilon <<"), for tgt '"<< val <<"' so error is acceptable (for epoch "<< e << " of sample " << s << "), returning" << endl << flush;
             return 1;
        }
#endif
#ifdef EXTRA_OUTPUT 
        cout << "------------------------------------ BACK PROPAGATION  err=" << err << endl << flush;
#endif
     
        ftick[OLayer] = (-actuation[OLayer] + 1) % (actuation[OLayer]);  //element wise multiply

        deltafn[OLayer]  =  (tgt - actuation[OLayer])%(ftick[OLayer]);

        for (int i = OLayer - 1; i >= 0; i--)
        {
            weight_updates[i] = actuation[i].t() * deltafn[i+1];
            new_layer_weights[i]  =  layer_weights[i] + (eta *  weight_updates[i]) ;
             
            ftick[i] = (-actuation[i] + 1) % actuation[i];

            deltafn[i] = ( layer_weights[i] * deltafn[i+1].t() ).t() % ftick[i];
        }
        for (int i=0;i<OLayer;i++)
        {
           layer_weights[i] =  new_layer_weights[i];
#ifdef USE_BIASES
           layer_biases[i] += deltafn[i+1];
#endif
        }
        return 0;
}

void forward_feed(unsigned char * &imgdata, unsigned char * &labdata, bool train, int samples)
{
    rowvec tgt;
    int tot_correct=0;
    int tot_wrong=0;
    int correct_num=-1;
    int best_guess=-1;
    int num_correct[10]={0,0,0,0,0,0,0,0,0,0};
    int num_wrong[10]={0,0,0,0,0,0,0,0,0,0};
    int chosen_wrongly[10][10]={{ 0,0,0,0,0,0,0,0,0,0},
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
    bool fin_this_epoch = false;
    string intype;
    int epochs;

    if (train)
    {
       intype = "TRAINING";
       // Training ANN so process and backpropagate "epoch" times
       epochs = EPOCHS;
    }
    else
    {
       intype="TEST    ";
       // Testing, so do only once and get results
       epochs = 1;
    }
    for (int y=0;y<samples;y++)
    {
        fin_this_epoch = false;
        cout << "------------------------------------ FORWARD FEED OF "<<intype <<" SAMPLE # "<< y+1 << " for "<<epochs << " epochs"<< endl << flush;
        load_an_image(y, imgdata, actuation[0], tgt, labdata);
        int tgtval = tgt.index_max();
        for (int z = 0; z < epochs && !fin_this_epoch; z++)
        {
#ifdef  EXTRA_OUTPUT
            cout << "----------- Epoch # " << z+1 << " on Sample # " << y+1 << endl << flush;
#endif
            for (int i=0;i<OLayer;i++)  // only n-1 transitions between n layers
            {
               // cout << "------------------------------------ All inputs into L" << i << endl << flush;
                // sum layer 1 weighted input
                         //netin[i] =  (actuation[i] * layer_weights[i])/((double) actuation[i].n_cols);
#ifdef USE_BIASES
                netin[i] =  (actuation[i] * layer_weights[i]) + (layer_biases[i]);
#else
                netin[i] =  (actuation[i] * layer_weights[i]);
#endif
                //cout << "------------------------------------ Net weighted sum into L" << i << endl << flush;
                //cout << "------------------------------------ Activation out of L" << i << endl << flush;
                actuation[i+1] = sigmoid(netin[i]);
            }
            if (train)
            {
                // printout intermediate result
                int outval = actuation[OLayer].index_max();
#ifdef EXTRA_OUTPUT
                std::cout << "Train output : " << endl << actuation[OLayer] << std::endl << flush;
                int minval= tgtval<outval?tgtval:outval;
                int maxval= tgtval>outval?tgtval:outval;
                string minc= tgtval == minval ? to_string(minval)+string("A"):to_string(minval)+string("O");
                string maxc= tgtval == maxval ? to_string(maxval)+"A":to_string(maxval)+"O";
                if (minval==maxval)
                   minc="*"+to_string(minval); // correct
                for (int x = 0; x < minval; x++)
                    cout << "         " << flush;
                cout << "       " << minc << flush;  
                for (int x = 0; x < maxval - minval-1; x++)
                    cout << "         " << flush;
                if (minval != maxval)
                    cout << "       " << maxc << flush;  // expected
                cout << endl << flush;
#endif
                if (backprop(tgt,y,z)==1)
                         fin_this_epoch=true;
            }
        }
        if (!train)
        {
            correct_num = tgt.index_max();
            best_guess = actuation[OLayer].index_max();

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
#ifdef EXTRA_OUTPUT
        std::cout << "Final output : " << endl << actuation[OLayer] << std::endl << flush;
        for (int t=0;t<actuation[OLayer].index_max();t++)
             cout << "         " << flush;
        cout << "       ^" << endl << flush;
        std::cout << "Expec output : " << endl << tgt << std::endl << flush;
#endif 
                //////////////////////////// forward feed end
    }
    if (!train)
    {
         cout << endl << endl << endl << "CONFUSION MATRIX" << endl << "****************" << endl << flush;
         cout << "Tested " << num_tested << " samples"<<endl << flush;
         for (int i=0;i<10;i++)
             cout  <<  "      "<< dec << std::setw(7) << i  << flush;
         cout << "      Guessed"  << flush;
         double colsum[10]={0,0,0,0,0,0,0,0,0,0};
         double rowsum[10]={0,0,0,0,0,0,0,0,0,0};
         for (int i=0;i<10;i++)
         {
            cout << endl  << "---------------------------------------------------------------------------------------------------------------------------------------" << endl << i << " |  " << flush;
            for (int j=0;j<10;j++)
            {
                rowsum[i] +=  chosen_wrongly[i][j];
                colsum[j] +=  chosen_wrongly[i][j];
                cout  << std::setw(7) << chosen_wrongly[i][j] <<  "      " << flush;
            }
            float pctg=(float)(rowsum[i])/ (float) (tot_wrong) * 100.0f;
            cout << "| " <<  setw(7)  <<rowsum[i]  << flush;
            cout <<  setw(7)   <<"         " << pctg  <<  resetiosflags( ios::fixed  |ios::showpoint )<< "%" << flush;

         }
         cout << endl << flush;
         cout << "---------------------------------------------------------------------------------------------------------------------------------------" << endl << "     " << flush;
         for (int i=0;i<10;i++)
             cout  << dec << std::setw(7) << colsum[i] << "      " << flush;
         cout << endl << "     " << flush;
         for (int i=0;i<10;i++)
         {
             float pctg=(float)(colsum[i])/ (float) (tot_wrong) * 100.0f;
            cout << dec <<  setw(7) << fixed << showpoint << setprecision(2) << pctg  << resetiosflags( ios::fixed | ios::showpoint )<< "%     " << flush;
         }
         cout << endl << flush;
         float totpctg=(float)(tot_correct)/ (float) (tot_correct+tot_wrong) * 100.0f;
         cout << "Target " << endl << "Above percentages are of number total wrong (" << tot_wrong << ") out of total " << tot_correct+tot_wrong << " (ie ~" << 100- totpctg << "% of total tests)" << endl << endl << endl << endl << "Correct selections:" << endl << flush;
         for (int i=0;i<10;i++)
             cout  << dec << std::setw(7) << i << "      " << flush;
         cout << endl << flush;
         for (int i=0;i<10;i++)
         {
                cout  << std::setw(7) << num_correct[i] <<  "      " << flush;
         }
         cout << endl << endl << "Incorrect selections:" << endl << flush;
         for (int i=0;i<10;i++)
             cout  << dec << std::setw(7) << i << "      " << flush;
         cout << endl << flush;
         for (int i=0;i<10;i++)
         {
                cout  << std::setw(7) << num_wrong[i] <<  "      " << flush;
         }
         cout << endl << endl << flush; 
         cout << "Total Correct : " <<  std::setw(7) << fixed << showpoint <<std::setprecision(2) <<totpctg << "%     " << resetiosflags( ios::fixed | ios::showpoint ) <<endl << endl << flush;
    }
                
}

void save_weights(string hdr)
{
    ofstream oFile;
    string fname = hdr+string("_weights_") + fid +string(".txt");
    cout << "Saving weights to file : " << fname << endl << flush;
    oFile.open(fname, ios::out);
    oFile << "NumberOfLayers=" << NumberOfLayers << endl << flush;
    for (int i=0; i< OLayer; i++)
    {
        
        oFile <<  "NodesInLayer"<<i<<"=" << nodes[i] << endl << flush;
        oFile << layer_weights[i] << endl << flush;
        oFile <<  "LayerBiases"<<i<< endl << flush;
#ifdef USE_BIASES
        oFile << layer_biases[i] << endl << flush;
#else
        oFile << "No layer biases are used" << endl << flush;
#endif
    }
//    oFile << "Error Summary" << endl << flush;
//    oFile << err_summary << endl << flush;
    oFile << "EndFile" << endl << flush;
    oFile.close();

}

int main (int argc, char *argv[])
{


    extern char **environ;

    vector<string> strs;

#ifdef EXTRA_MESSAGE
    cout << EXTRA_MESSAGE << endl;
#endif
    // Use slurm job number if avaiable (else defaults to epoch time) for file ids created
    for(char **current = environ; *current; current++) {
        string tmp = *current;
        boost::split(strs, tmp, boost::is_any_of("="));
        if ((strs[0] == "SLURM_JOBID") || (strs[0] == "SLURM_JOB_ID"))
           if (strs[1].length() > 0)
           {
             fid = strs[1];
             break;
           }
    }

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
        else if (argc < 5)
        {
             cout << "Usage: " << argv[0] << " ETA IN H1 [H2 H3 ...] OUT" << endl << flush;
             cout << "       Where ETA is the learning factor, &" << endl << flush;
             cout << "       Where number of parameters after ETA is the number of layers" << endl << flush;
             cout << "       Must have a minimum of 3, i.e. IN H1 OUT" << endl << flush;
             cout << "       And the parameters themselves are numbers, "<< endl << flush;
             cout << "       indicating the number of nodes in that layer." << endl << flush;
             cout << "       e.g. \"" << argv[0] <<  " "<< ETA_DEFAULT << " " << INPUT_LINES << " " << DEFAULT_HIDDEN << " " << OUTPUT_LINES << "\" " << endl << flush;
             cout << "       and is the default, if no params supplied." << endl << flush;
             exit (1);
        }
        else
        {
             NumberOfLayers = argc-2;
             nodes = new unsigned int [NumberOfLayers];
             eta = stod(string(argv[1]));
             if (eta <= 0)
             {
                   cout << "Error: ETA must be positive, usually less than 1" << endl << flush;
                   exit(1);
             }
             cout << "And ETA=" << eta << endl << flush;;
             nodes[0] = stoi(string(argv[2]));
             if (nodes[0] != 784)
             {
                  cout << "Error: For this application, number of input nodes MUST be 784 (as flattening a 28x28 pixel image)" << endl;
                  exit(1);
             }
             nodes[argc-3] = stoi(string(argv[argc-1]));
             if (nodes[argc-3] != 10)
             {
                  cout << "Error: For this application, number of output nodes MUST be 10 (as categorising this many classes)" << endl;
                  exit(1);
             }
             string optns="";
             for (int i=2;i<argc;i++)
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
                optns = optns + " " + string(argv[i]);
             }
             cout << "Using 1 input and " << NumberOfLayers-2 << " hiddenlayers and 1 outputlayer" << endl;
             cout << "That contain number of nodes " << optns << " respectively" << endl;
        }
        OLayer = NumberOfLayers - 1;
        
    unsigned char * trainlabels; 
    unsigned char * testlabels; 
    unsigned char * traindata;
    unsigned char * testdata;
    auto StartTime = std::chrono::high_resolution_clock::now();



    load_file("train-images-idx3-ubyte", "train-labels-idx1-ubyte", &trainlabels, &traindata);
    load_file("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", &testlabels, &testdata);
    cout << "Training epochs per test data is : " << EPOCHS << endl << flush;
#ifdef USE_BIASES
    cout << "Bias weighting is used " << endl << flush;
#else
    cout << "Bias weighting is NOT used " << endl << flush;
#endif

///////////////////////////////////////////////
//
//  CREATE ARRAY OF MATRICES AND VECTORS
//  AND SET WEIGHTS TO RANDOM (0 < w < 1)
//
    input =  zeros< rowvec > ( nodes[0] );
    actuation.push_back( input );
    deltafn.push_back( input );
    ftick.push_back( input );
    for (int i = 0; i < OLayer; i++)
    {
         rowvec t =  ones< rowvec >( nodes[i+1] ); // network weights for each node + 1 node bias weight
         rowvec r =  randu< rowvec >( nodes[i+1] ); // network weights for each node + 1 node bias weight
          // initialise weights randomly between -0.5 and 0.5
         mat tmpwgt1 = (2 * randu< mat >( nodes[i], nodes[i+1] ) - 1)/2; // network weights for each node + 1 node bias weight
         mat zzzwgt2 =  zeros< mat >( nodes[i], nodes[i+1] ); // network weights for each node + 1 node bias weight
         netin.push_back( t );   // size=nodes[i],1
#ifdef USE_BIASES
         layer_biases.push_back( r );   // size=nodes[i],1
#endif
         actuation.push_back( t ); // size= nodes[i],1
         deltafn.push_back( t );
//         theta.push_back( tmpwgt1  );
         ftick.push_back( t );

         layer_weights.push_back( tmpwgt1 );
         new_layer_weights.push_back( zzzwgt2 );
         weight_updates.push_back( zzzwgt2 );
    }
   save_weights("initial_random_values"); 
/////////////////////////////////////////////// 
//
// TRAIN THE DATA
//
    cout << "Training on data started...." << endl;
    auto StartTrainTime = std::chrono::high_resolution_clock::now();
    forward_feed(traindata, trainlabels, true, 60000);
    auto EndTrainTime = std::chrono::high_resolution_clock::now();
    save_weights("post_training_weights");   
    cout << "Training complete" << endl;
/////////////////////////////////////////////// 
//
// TEST THE DATA
//
    cout << "Testing of data started...." << endl;
    auto StartTestTime = std::chrono::high_resolution_clock::now();
    forward_feed(testdata, testlabels, false, 10000);
    auto EndTestTime = std::chrono::high_resolution_clock::now();
    cout << "Testing complete" << endl;

    cout << "Total Time       : " <<    std::setw(12) << (EndTestTime-StartTime).count() <<" us"<< endl;
    cout << "Total Train Time : " << std::setw(12) <<    (EndTrainTime-StartTrainTime).count() <<" us"<< endl;
    cout << "Total Test Time  : " <<  std::setw(12) <<   (EndTestTime-StartTestTime).count() <<" us"<< endl;

 
//        delete[] traindata;
//        delete[] trainlabels;
//        delete[] testdata;
//        delete[] testlabels;
}





/*
below is based on final result from post_training_weights_weights_1631975398.txt
CONFUSION MATRIX
****************
Tested 10000 samples
            0            1            2            3            4            5            6            7            8            9      Guessed
---------------------------------------------------------------------------------------------------------------------------------------
0 |        0            0            0            1            0            1            1            0            8            1      |      12         1.60643%
---------------------------------------------------------------------------------------------------------------------------------------
1 |        0            0            1            7            0            0            1            1           13            0      |      23         3.07898%
---------------------------------------------------------------------------------------------------------------------------------------
2 |       12            2            0           33            5            0           11            6           21            3      |      93         12.4498%
---------------------------------------------------------------------------------------------------------------------------------------
3 |        5            2            5            0            1            8            1            4           12            7      |      45         6.0241%
---------------------------------------------------------------------------------------------------------------------------------------
4 |        4            1            4            1            0            1           11            1            2           66      |      91         12.1821%
---------------------------------------------------------------------------------------------------------------------------------------
5 |       13            4            0           40           12            0           12            3           48           22      |     154         20.6158%
---------------------------------------------------------------------------------------------------------------------------------------
6 |       16            3            2            1            9            6            0            1           12            0      |      50         6.69344%
---------------------------------------------------------------------------------------------------------------------------------------
7 |       11           18           19           19            8            1            0            0            3           39      |     118         15.7965%
---------------------------------------------------------------------------------------------------------------------------------------
8 |        8            5            6           31            9            3            6            4            0           38      |     110         14.7256%
---------------------------------------------------------------------------------------------------------------------------------------
9 |        7            6            2            6           12            2            2            6            8            0      |      51         6.82731%
---------------------------------------------------------------------------------------------------------------------------------------
          76           41           39          139           56           22           45           26          127          176      
       10.17%        5.49%        5.22%       18.61%        7.50%        2.95%        6.02%        3.48%       17.00%       23.56%     
Target 
Above percentages are of number total wrong (747) out of total 10000 (ie 7.5% of total tests)



Correct selections:
      0            1            2            3            4            5            6            7            8            9      
    968         1112          939          965          891          738          908          910          864          958      

Incorrect selections:
      0            1            2            3            4            5            6            7            8            9      
     12           23           93           45           91          154           50          118          110           51      

Total Correct :   92.53%     
*/
