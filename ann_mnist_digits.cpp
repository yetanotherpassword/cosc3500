#include <iostream>
#include <iomanip>
#include <cmath>
#include <armadillo>

#undef DEBUGON

#define ARMA_64BIT_WORD
#define INPUT_LINES 784
#define OUTPUT_LINES 10
#define MATRIX_SIDE 28
#define MAX_PIXEL_VAL 255.0f
#define IMAGE_OFFSET 16
#define DEFAULT_HIDDEN 30
#define ETA_DEFAULT 0.5f



/*
 * ALLAN CAMPTON
 * COSC3500 Milestone 1 Serial Version
 *
 * To perform a full build and run from scratch, to the following
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

 
using namespace arma;
using namespace std;


    unsigned int NumberOfLayers;
    unsigned int * nodes;
    double eta;               // Learning factor
    vector<rowvec> netin;
    vector<rowvec> actuation;
    vector<rowvec> deltafn;
    vector<rowvec> ftick;
    vector<mat> layer_weights;
    vector<mat> weight_updates;
    vector<mat> new_layer_weights;
    rowvec tmpbias; 
    mat tmpwgt; 
    colvec vbias;

rowvec sigmoid( rowvec  & net)
{
   rowvec out = 1/(1+exp(-net));
   out.insert_cols(out.n_cols, 1); // add bias signal (1) column
   out(out.n_cols-1)=1.0;          // add bias signal value
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
     cout << endl;
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
    img.set_size(INPUT_LINES+1);
    for (int i=0;i<INPUT_LINES;i++)
    {
        img(i) = ((double ) mptr[start+i])/greyval;
    }
     //   cout << img << endl << "an image ************************" << endl;
    img(INPUT_LINES)=1;          // set bias signal, so can multiply with [node weights | bias weights] augmented matrix

    int img_is_digit=(int) lp[8+seq];

//    print_an_image(&mptr[start], img_is_digit);

    t=zeros<rowvec>(OUTPUT_LINES+1); // create the target vector (plus one for 'bias' bit)
    t(img_is_digit)=1;               // set the target 'bit'
    t(t.n_cols-1)=1;                 // set bias signal (redundant for target, but keeps vectors same size)

}

void backprop(rowvec tgt)
{
        cout << "------------------------------------ BACK PROPAGATION" << endl;
     
        ftick[NumberOfLayers-1] = -actuation[NumberOfLayers-1] + 1;
        ftick[NumberOfLayers-1] = ftick[NumberOfLayers-1] % (actuation[NumberOfLayers-1]);  //element wise multiply
        deltafn[NumberOfLayers-1]  =  (tgt - actuation[NumberOfLayers-1])%(ftick[NumberOfLayers-1]);
        deltafn[NumberOfLayers-1].shed_col(deltafn[NumberOfLayers-1].n_cols-1);
        for (int i=NumberOfLayers-2;i>=0;i--)
        {
            weight_updates[i]  =  deltafn[i+1].t() * actuation[i];
            new_layer_weights[i]  =  layer_weights[i] + (eta *  weight_updates[i]) ;
             
            ftick[i] = -actuation[i] + 1;
            ftick[i] = ftick[i] % (actuation[i]);  //element wise multiply
            deltafn[i] = deltafn[i+1]*layer_weights[i];
            deltafn[i] = deltafn[i] % ftick[i];
            deltafn[i].shed_col(deltafn[i].n_cols-1);
        }
        for (int i=0;i<NumberOfLayers;i++)
        {
           layer_weights[i] =  new_layer_weights[i];
        }
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
    string intype="TEST    ";
    if (train)
       intype="TRAINING";
    for (int y=0;y<samples;y++)
    {
        cout << "------------------------------------ FORWARD FEED OF "<<intype <<" SAMPLE # "<< y+1 << endl;
        load_an_image(y, imgdata, actuation[0], tgt, labdata);
        for (int i=0;i<NumberOfLayers-1;i++)  // only n-1 transitions between n layers
        {
           // cout << "------------------------------------ All inputs into L" << i << endl;
            // sum layer 1 weighted input
            netin[i] =  (actuation[i] * layer_weights[i].t())/actuation[i].n_cols;
            //cout << "------------------------------------ Net weighted sum into L" << i << endl;
            //cout << "------------------------------------ Activation out of L" << i << endl;

            actuation[i+1] = sigmoid(netin[i]);
        }
        std::cout << "Final output : " << endl << actuation[NumberOfLayers-1] << std::endl;
        std::cout << "Expec output : " << endl << tgt << std::endl;
        
                //////////////////////////// forward feed end
        if (train)
        {
            backprop(tgt);
        }
        else
        {
            double max_guess=-100.0;
            for (int i=0;i< actuation[NumberOfLayers-1].n_cols-1;i++)
            {
                   if (tgt(i)==1)
                       correct_num = i;
                   if ( actuation[NumberOfLayers-1](i) > max_guess)
                   {
                       best_guess = i;
                       max_guess = actuation[NumberOfLayers-1](i);
                   }
                   cout << " Guessed " << best_guess << " and it was " << correct_num << endl;
                   std::cout << "Test Final output : " << endl << actuation[NumberOfLayers-1] << std::endl;
                   std::cout << "Test Expec output : " << endl << tgt << std::endl;
  	    }
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
    }
    if (!train)
    {
         cout << "Tested " << num_tested << " samples"<<endl;
         for (int i=0;i<10;i++)
             cout  <<  "      "<< dec << std::setw(7) << i ;
         cout << "      Guessed" << endl;;
         cout << "---------------------------------------------------------------------------------------------------------------------------------------" << endl;
         double colsum[10]={0,0,0,0,0,0,0,0,0,0};
         double rowsum[10]={0,0,0,0,0,0,0,0,0,0};
         for (int i=0;i<10;i++)
         {
            cout << endl << i << " |  ";
            for (int j=0;j<10;j++)
            {
                rowsum[i] +=  chosen_wrongly[i][j];
                colsum[j] +=  chosen_wrongly[i][j];
                cout  << std::setw(7) << chosen_wrongly[i][j] <<  "      ";
            }
            float pctg=(float)(rowsum[i])/ (float) (tot_wrong) * 100.0f;
            cout << "| " <<  setw(7)  <<rowsum[i] ;
            cout <<  setw(7)   <<"         " << pctg  <<  resetiosflags( ios::fixed  |ios::showpoint )<< "%";

         }
         cout << endl;
         cout << "---------------------------------------------------------------------------------------------------------------------------------------" << endl << "     ";
         for (int i=0;i<10;i++)
             cout  << dec << std::setw(7) << colsum[i] << "      ";
         cout << endl << "     ";
         for (int i=0;i<10;i++)
         {
             float pctg=(float)(colsum[i])/ (float) (tot_wrong) * 100.0f;
            cout << dec <<  setw(7) << fixed << showpoint << setprecision(2) << pctg  << resetiosflags( ios::fixed | ios::showpoint )<< "%     ";
         }
         cout << endl;
         cout << "Target " << endl << endl << endl << endl << endl << "Correct selections:" << endl;
         for (int i=0;i<10;i++)
             cout  << dec << std::setw(7) << i << "      ";
         cout << endl;
         for (int i=0;i<10;i++)
         {
                cout  << std::setw(7) << num_correct[i] <<  "      ";
         }
         cout << endl << endl << "Incorrect selections:" << endl;
         for (int i=0;i<10;i++)
             cout  << dec << std::setw(7) << i << "      ";
         cout << endl;
         for (int i=0;i<10;i++)
         {
                cout  << std::setw(7) << num_wrong[i] <<  "      ";
         }
         cout << endl << endl; 
         float pctg=(float)(tot_correct)/ (float) (tot_correct+tot_wrong) * 100.0f;
         cout << "Total Correct : " <<  std::setw(7) << fixed << showpoint <<std::setprecision(2) <<pctg << "%     " << resetiosflags( ios::fixed | ios::showpoint ) <<endl << endl;
    }
                
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
    unsigned char * trainlabels; 
    unsigned char * testlabels; 
    unsigned char * traindata = load_file("train-images-idx3-ubyte", "train-labels-idx1-ubyte", &trainlabels);
    unsigned char * testdata = load_file("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", &testlabels);

///////////////////////////////////////////////
//
//  CREATE ARRAY OF MATRICES AND VECTORS
//  AND SET WEIGHTS TO RANDOM (0 < w < 1)
//
    for (int i=0;i <= NumberOfLayers-1; i++)
    {
         netin.push_back({});   // size=nodes[i],1
         actuation.push_back({}); // size= nodes[i],1
         deltafn.push_back({});
         ftick.push_back({});
          if (i<NumberOfLayers-1)
            tmpwgt = randu<mat>(nodes[i+1],nodes[i]+1); // network weights for each node + 1 node bias weight
         layer_weights.push_back( tmpwgt );
         new_layer_weights.push_back({});
         weight_updates.push_back({});
    }
   
/////////////////////////////////////////////// 
//
// TRAIN THE DATA
//
    forward_feed(traindata, trainlabels, true, 60000);
   
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
