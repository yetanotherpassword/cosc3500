#include <iostream>
#include <iomanip>
#include <cmath>
#include <armadillo>

#include <ctime>
 

#define ARMA_64BIT_WORD

#define INPUT_LINES 784
#define DEFAULT_HIDDEN 30
#define OUTPUT_LINES 10


#define MATRIX_SIDE 28
#define MAX_PIXEL_VAL 255.0f
#define IMAGE_OFFSET 16

#define ETA_DEFAULT 1e-3
#define MOMENTUM 0.9f
#define EPSILON 1e-3

#undef USE_BIASES


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
stringstream confusion_matrix;
rowvec err_summary=ones<rowvec>(OUTPUT_LINES) * (-1);
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
     cout << "This is a : " << i << endl;
     for (int i=0;i<INPUT_LINES;i++)
     {
       if (i%MATRIX_SIDE==0)
         cout << endl;
       cout  << hex << std::setfill('0') << std::setw(2) << (unsigned int)c[i] << dec << " ";
     }
     cout << setfill(' ') << endl;
}
   

void print_images(unsigned char * c,  int size)
{
    for (int i=IMAGE_OFFSET;i<size;i++)
    {
       if (((i-IMAGE_OFFSET)%MATRIX_SIDE)==0)
           cout << endl;
       if (((i-IMAGE_OFFSET)%INPUT_LINES)==0)
           cout << endl << "Image : " << dec << ((i-IMAGE_OFFSET)/INPUT_LINES)+1 << endl;
       cout << hex << std::setfill('0') << std::setw(2) << (unsigned int)c[i] << " ";
    }
    cout << setfill(' ');
}

void outp(mat m, string s)
{
   cout << "matrix:" << s << " rows=" << m.n_rows <<   " cols=" << m.n_cols << endl;

}

void outp(rowvec v, string s)
{
   cout << "vector:" << s << " rows=" << v.n_rows <<   " cols=" << v.n_cols << endl;

}
//
//
/////////////////////////////////////////////

unsigned char * load_file(string filename, string labels, unsigned char * * labs)
{
    unsigned char * memblock;
    ifstream inFile;
    streampos size;

    cout << "Using file '" << filename << "'" << endl;
//
// Load MNIST DIGIT IMAGES
//
    inFile.open(filename, ios::in|ios::binary|ios::ate);
    if (!inFile) {
        cout << "Unable to open file '" << filename << "'" << endl;
        exit(1); // terminate with error
    }
    else
    {
       cout << "OK opened '" << filename << "' Successfully" << endl;
    }

    if (inFile.is_open())
    {
        size = inFile.tellg();
        memblock = new unsigned char [size];
        inFile.seekg (0, ios::beg);
        inFile.read ((char *)memblock, size);
        inFile.close();

        cout << "the entire file content is in memory, all " << size << " bytes of it" << endl;
         //print_images(memblock, size);
 
    }
    inFile.close();
//
// Load MNIST DIGIT LABELS
//
    inFile.open(labels, ios::in|ios::binary|ios::ate);
    if (!inFile) {
        cout << "Unable to open file '" << labels << "'" << endl;
        exit(1); // terminate with error
    }
    else
    {
       cout << "OK opened '" << labels << "' Successfully" << endl;
    }

    if (inFile.is_open())
    {
        size = inFile.tellg();
        *labs = new unsigned char [size];
        inFile.seekg (0, ios::beg);
        inFile.read ((char *) *labs, size);
        inFile.close();

        cout << "the entire file content is in memory, all " << size << " bytes of it" << endl;
        // print_images(memblock, size);
 
    }
    inFile.close();
    return memblock;

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
     //   cout << img << endl << "an image ************************" << endl;

    int img_is_digit=(int) lp[8+seq];

    print_an_image(&mptr[start], img_is_digit);

//    t=zeros<rowvec>(OUTPUT_LINES+1); // create the target vector (plus one for 'bias' bit)
//    t(t.n_cols-1)=1;                 // set bias signal (redundant for target, but keeps vectors same size)

    t=zeros<rowvec>(OUTPUT_LINES); // create the target vector (plus one for 'bias' bit)
    t(img_is_digit)=1;               // set the target 'bit'

}

int backprop(rowvec tgt)
{
        double err = accu((tgt - actuation[OLayer]) %  (tgt - actuation[OLayer]))*0.5;
        if (err < epsilon)
        {
             int val=tgt.index_max();
             cout << "---------------------------------- BACK PROPAGATION  err=" << err << " < epsilon, for tgt '"<< val <<"' so error is acceptable, returning" << endl;
             err_summary(val) = err;
             return 1;
        }
        cout << "------------------------------------ BACK PROPAGATION  err=" << err << endl;
     
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
    int epochs=1;
    string intype="TEST    ";

    if (train)
    {
       intype="TRAINING";
       epochs=512;
    }
    for (int y=0;y<samples;y++)
    {
        cout << "------------------------------------ FORWARD FEED OF "<<intype <<" SAMPLE # "<< y+1 << endl;
        load_an_image(y, imgdata, actuation[0], tgt, labdata);
        int tgtval = tgt.index_max();
        for (int z=0;z<epochs;z++)
        {
            cout << "----------- Epoch # " << z+1 << " on Sample # " << y+1 << endl;
            for (int i=0;i<OLayer;i++)  // only n-1 transitions between n layers
            {
               // cout << "------------------------------------ All inputs into L" << i << endl;
                // sum layer 1 weighted input
                         //netin[i] =  (actuation[i] * layer_weights[i])/((double) actuation[i].n_cols);
#ifdef USE_BIASES
                netin[i] =  (actuation[i] * layer_weights[i]) + (layer_biases[i]);
#else
                netin[i] =  (actuation[i] * layer_weights[i]);
#endif
                //cout << "------------------------------------ Net weighted sum into L" << i << endl;
                //cout << "------------------------------------ Activation out of L" << i << endl;
                actuation[i+1] = sigmoid(netin[i]);
            }
            if (train)
            {
                // printout intermediate result
                int outval = actuation[OLayer].index_max();
                std::cout << "Train output : " << endl << actuation[OLayer] << std::endl;
                int minval= tgtval<outval?tgtval:outval;
                int maxval= tgtval>outval?tgtval:outval;
                string minc= tgtval == minval ? to_string(minval)+string("A"):to_string(minval)+string("O");
                string maxc= tgtval == maxval ? to_string(maxval)+"A":to_string(maxval)+"O";
                if (minval==maxval)
                   minc="*"+to_string(minval); // correct
                for (int z1 = 0; z1 < minval; z1++)
                    cout << "         ";
                cout << "       " << minc;  
                for (int z1 = 0; z1 < maxval - minval-1; z1++)
                    cout << "         ";
                if (minval != maxval)
                    cout << "       " << maxc;  // expected
                cout << endl;
                if (backprop(tgt))
                         break;
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
        std::cout << "Final output : " << endl << actuation[OLayer] << std::endl;
        for (int z=0;z<actuation[OLayer].index_max();z++)
             cout << "         ";
        cout << "       ^" << endl;
        std::cout << "Expec output : " << endl << tgt << std::endl;
        
                //////////////////////////// forward feed end
    }
    if (!train)
    {
         confusion_matrix << endl << endl << endl << "CONFUSION MATRIX" << endl << "****************" << endl;
         confusion_matrix << "Tested " << num_tested << " samples"<<endl;
         cout << "Tested Correct " << tot_correct << " samples"<<endl;
         cout << "Tested Wrong   " << tot_wrong<< " samples"<<endl;
         for (int i=0;i<10;i++)
             confusion_matrix  <<  "      "<< dec << std::setw(7) << i ;
         confusion_matrix << "      Guessed" ;
         double colsum[10]={0,0,0,0,0,0,0,0,0,0};
         double rowsum[10]={0,0,0,0,0,0,0,0,0,0};
         for (int i=0;i<10;i++)
         {
            confusion_matrix << endl  << "---------------------------------------------------------------------------------------------------------------------------------------" << endl << i << " |  ";
            for (int j=0;j<10;j++)
            {
                rowsum[i] +=  chosen_wrongly[i][j];
                colsum[j] +=  chosen_wrongly[i][j];
                confusion_matrix  << std::setw(7) << chosen_wrongly[i][j] <<  "      ";
            }
            float pctg=(float)(rowsum[i])/ (float) (tot_wrong) * 100.0f;
            confusion_matrix << "| " <<  setw(7)  <<rowsum[i] ;
            confusion_matrix <<  setw(7)   <<"         " << pctg  <<  resetiosflags( ios::fixed  |ios::showpoint )<< "%";

         }
         confusion_matrix << endl;
         confusion_matrix << "---------------------------------------------------------------------------------------------------------------------------------------" << endl << "     ";
         for (int i=0;i<10;i++)
             confusion_matrix  << dec << std::setw(7) << colsum[i] << "      ";
         confusion_matrix << endl << "     ";
         for (int i=0;i<10;i++)
         {
             float pctg=(float)(colsum[i])/ (float) (tot_wrong) * 100.0f;
            confusion_matrix << dec <<  setw(7) << fixed << showpoint << setprecision(2) << pctg  << resetiosflags( ios::fixed | ios::showpoint )<< "%     ";
         }
         confusion_matrix << endl;
         float totpctg=(float)(tot_correct)/ (float) (tot_correct+tot_wrong) * 100.0f;
         confusion_matrix << "Target " << endl << "Above percentages are of number total wrong (" << tot_wrong << ") out of total " << tot_correct+tot_wrong << " (ie " << 100- totpctg << "% of total tests)" << endl << endl << endl << endl << "Correct selections:" << endl;
         for (int i=0;i<10;i++)
             confusion_matrix  << dec << std::setw(7) << i << "      ";
         confusion_matrix << endl;
         for (int i=0;i<10;i++)
         {
                confusion_matrix  << std::setw(7) << num_correct[i] <<  "      ";
         }
         confusion_matrix << endl << endl << "Incorrect selections:" << endl;
         for (int i=0;i<10;i++)
             confusion_matrix  << dec << std::setw(7) << i << "      ";
         confusion_matrix << endl;
         for (int i=0;i<10;i++)
         {
                confusion_matrix  << std::setw(7) << num_wrong[i] <<  "      ";
         }
         confusion_matrix << endl << endl; 
         confusion_matrix << "Total Correct : " <<  std::setw(7) << fixed << showpoint <<std::setprecision(2) <<totpctg << "%     " << resetiosflags( ios::fixed | ios::showpoint ) <<endl << endl;
         cout << confusion_matrix;
    }
                
}

void save_weights(string hdr)
{
    ofstream oFile;
    std::time_t result = std::time(nullptr);
    string fname = hdr+string("_weights_") + to_string(result)+string(".txt");
    cout << "Saving weights to file : " << fname << endl;
    oFile.open(fname, ios::out);
    if (hdr.substr(0,4)=="post")
       oFile << confusion_matrix;      
    oFile << "NumberOfLayers=" << NumberOfLayers << endl;
    for (int i=0; i< OLayer; i++)
    {
        
        oFile <<  "NodesInLayer"<<i<<"=" << nodes[i] << endl;
        oFile << layer_weights[i] << endl;
        oFile <<  "LayerBiases"<<i<< endl;
#ifdef USE_BIASES
        oFile << layer_biases[i] << endl;
#else
        oFile << "No layer biases are used" << endl;
#endif
    }
    oFile << "Error Summary" << endl;
    oFile << err_summary << endl;
    oFile << "EndFile" << endl;
    oFile.close();

}

int main (int argc, char *argv[])
{
    
        if (argc < 2)
        {
            NumberOfLayers=3;
            nodes = new unsigned int [NumberOfLayers];
            nodes[0]=INPUT_LINES;
            nodes[1]=DEFAULT_HIDDEN;
            nodes[2]=OUTPUT_LINES;
            eta = ETA_DEFAULT;
            cout << "Using default setting of \"" << nodes[0] << " " << nodes[1] << " " << nodes[2]<<  "\" " << endl;
            cout << "And ETA=" << eta << endl;;
        }
        else if (argc < 5)
        {
             cout << "Usage: " << argv[0] << " ETA IN H1 [H2 H3 ...] OUT" << endl;
             cout << "       Where ETA is the learning factor, &" << endl;
             cout << "       Where number of parameters after ETA is the number of layers" << endl;
             cout << "       Must have a minimum of 3, i.e. IN H1 OUT" << endl;
             cout << "       And the parameters themselves are numbers, "<< endl;
             cout << "       indicating the number of nodes in that layer." << endl;
             cout << "       e.g. \"" << argv[0] <<  " "<< ETA_DEFAULT << " " << INPUT_LINES << " " << DEFAULT_HIDDEN << " " << OUTPUT_LINES << "\" " << endl;
             cout << "       and is the default, if no params supplied." << endl;
             exit (1);
        }
        else
        {
             NumberOfLayers = argc-2;
             nodes = new unsigned int [NumberOfLayers];
             eta = stod(string(argv[1]));
             if (eta <= 0)
             {
                   cout << "Error: ETA must be positive, usually less than 1" << endl;
                   exit(1);
             }
             for (int i=2;i<argc;i++)
             {
                int p = stoi(string(argv[i]));
                if (p > 0)
                {
                   nodes[i-2] = stoi(string(argv[i]));
                }
                else
                {
                   cout << "Error in parameter " << i << " - must be positive" << endl;
                   exit (1);
                }
             }
        }
        OLayer = NumberOfLayers - 1;
    unsigned char * trainlabels; 
    unsigned char * testlabels; 
    unsigned char * traindata = load_file("train-images-idx3-ubyte", "train-labels-idx1-ubyte", &trainlabels);
    unsigned char * testdata = load_file("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", &testlabels);

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
    forward_feed(traindata, trainlabels, true, 60000);
    save_weights("post_training_weights");   
/////////////////////////////////////////////// 
//
// TEST THE DATA
//
    forward_feed(testdata, testlabels, false, 10000);
    
        delete[] traindata;
        delete[] trainlabels;
        delete[] testdata;
        delete[] testlabels;
}
