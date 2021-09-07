#include <iostream>
#include <iomanip>
#include <cmath>
#include <armadillo>
#define TESTING
#define ARMA_64BIT_WORD
#define INPUT_LINES 784
#define OUTPUT_LINES 10
#define MATRIX_SIDE 28
#define MAX_PIXEL_VAL 255.0
#define IMAGE_OFFSET 16
// g++ armo.cpp -g -o armo -std=c++11 -O2 -larmadillo

// requires armodillo
// http://arma.sourceforge.net/docs.html#element_access
// (For testing data) https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

 
using namespace arma;
using namespace std;


rowvec sigmoid( rowvec  & net)
{
   rowvec norml = net;  // normalise
   rowvec out = 1/(1+exp(-norml));
   out.insert_cols(out.n_cols, 1); // add bias signal (1)
   out(out.n_cols-1)=1.0;
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

void load_an_image(int seq, unsigned char * mptr, rowvec & img, rowvec & t, unsigned char * lp)
{
    int start=(INPUT_LINES*seq)+IMAGE_OFFSET;
    img.set_size(INPUT_LINES+1);
    for (int i=0;i<INPUT_LINES;i++)
    {
        img(i) = ((float) mptr[start+i])/MAX_PIXEL_VAL;
    }
    img(INPUT_LINES)=1;          // set bias signal, so can multiply with [node weights | bias weights] augmented matrix

    int img_is_digit=(int) lp[8+seq];

    //print_an_image(&mptr[start], i);

    t=zeros<rowvec>(OUTPUT_LINES+1); // create the target vector (plus one for 'bias' bit)
    t(img_is_digit)=1;               // set the target 'bit'
    t(t.n_cols-1)=1;                 // set bias signal (redundant for target, but keeps vectors same size)

}


int main (int argc, char *argv[])
{
    /*
        if (argc < 2)
        {
            cout << "Usage: entropy FILENAME" << endl;
            return 1;
        }
        filename = string(argv[1]);
    */
    unsigned char * labptr; 
#ifdef TESTING
    unsigned int NumberOfLayers=3;
    int nodes[NumberOfLayers]={2,2,2};
#else
    unsigned int NumberOfLayers=4;
    int nodes[NumberOfLayers]={INPUT_LINES, 1000, 1000, OUTPUT_LINES};  
    unsigned char * memptr = load_file("train-images-idx3-ubyte", "train-labels-idx1-ubyte", &labptr);
#endif
    double eta = 0.5;               // Learning factor

    rowvec endtestcase; 
    rowvec tgt; 
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

    for (int i=0;i < NumberOfLayers; i++)
    {
         netin.push_back({});   // size=nodes[i],1
         actuation.push_back({}); // size= nodes[i],1
         deltafn.push_back({});
         ftick.push_back({});
         if (i==0)
         {
#ifdef TESTING
            tmpwgt = {{ 0.15, 0.2, 0.35 }, {0.25, 0.3, 0.35}};   // TEST CASE
#else
            tmpwgt = ones<mat>(nodes[i],2); // (not used) but no weight change to inputs, but needs an entry to match indices
#endif
         }
         else
         {
#ifdef TESTING
            tmpwgt = { { 0.4, 0.45, 0.6 }, {0.5, 0.55, 0.6}};   // TEST CASE
#else
            tmpwgt = randu<mat>(nodes[i],nodes[i-1]+1); // network weights for each node + 1 node bias weight
#endif
         }
         layer_weights.push_back( tmpwgt );
         new_layer_weights.push_back({});
         weight_updates.push_back({});
    }
    
#ifndef TESTING
            load_an_image(0, memptr, actuation[0], tgt, labptr);
            endtestcase=actuation[0];
            for (int y=1;y<60000;y++)
#endif
            {
#ifdef TESTING
                actuation[0]={0.05, 0.1, 1};
                tgt={0.01, 0.99, 1};
#else
                load_an_image(y, memptr, actuation[0], tgt, labptr);
#endif
                for (int i=0;i<NumberOfLayers-1;i++)  // only n-1 transitions between n layers
                {
                    // sum layer 1 weighted input
                    cout << "------------------------------------ Net Input into L" << i << endl;
#ifdef TESTING
                    netin[i] =  (actuation[i] * layer_weights[i].t());
#else
                    netin[i] =  (actuation[i] * layer_weights[i].t())/actuation[i].n_cols;
#endif
cout << "----- Actuation In" << endl;
                    std::cout << actuation[i] << std::endl;
cout << "----- Weights" << endl;
                    std::cout << layer_weights[i].t() << std::endl;
cout << "----- Netin" << endl;
                    std::cout << netin[i] << std::endl;
                
                    cout << "------------------------------------ Activation out of L" << i << endl;

                    actuation[i+1] = sigmoid(netin[i]);
cout << "----- Actuation Out" << endl;
                    std::cout << actuation[i+1] << std::endl;
     outp(actuation[i+1], "actuation[i+1]");           
                }
                    std::cout << "Final output : " << endl << actuation[NumberOfLayers-1] << std::endl;
                    std::cout << "Expec output : " << endl << tgt << std::endl;
        
                //////////////////////////// forward feed end
                
                cout << "------------------------------------ BACK PROPAGATION" << endl;
     
                ftick[NumberOfLayers-1] = -actuation[NumberOfLayers-1] + 1;
     outp(ftick[NumberOfLayers-1], "ftick[NumberOfLayers-1]");
                ftick[NumberOfLayers-1] = ftick[NumberOfLayers-1] % (actuation[NumberOfLayers-1]);  //element wise multiply
     outp(ftick[NumberOfLayers-1], "ftick[NumberOfLayers-1]*2");
                deltafn[NumberOfLayers-1]  =  (tgt - actuation[NumberOfLayers-1])%(ftick[NumberOfLayers-1]);
                deltafn[NumberOfLayers-1].shed_col(deltafn[NumberOfLayers-1].n_cols-1);
     outp(deltafn[NumberOfLayers-1], "deltafn[NumberOfLayers-1]");
                 
                for (int i=NumberOfLayers-2;i>=0;i--)
                {
                    cout << "------------------------------------ Delta of Layer"<< i << endl;
                   std::cout << deltafn[i+1]<< std::endl;
                    
                    cout << "------------------------------------ Updates to weights at Layer" << i << endl;
                    weight_updates[i]  =  deltafn[i+1].t() * actuation[i];
                    std::cout << weight_updates[i]<< std::endl;
                    
                    cout << "------------------------------------ New weights are at Layer"<<i << endl;
                    new_layer_weights[i]  =  layer_weights[i] + (eta *  weight_updates[i]) ;
                    std::cout << new_layer_weights[i] << std::endl;
                    
                    here:
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

#ifndef TESTING        
// test
            actuation[0]=endtestcase;
            for (int i=0;i<NumberOfLayers-1;i++)  // only n-1 transitions between n layers
            {
                cout << "------------------------------------ Net Input into L" << i << endl;
                netin[i] =  ((actuation[i] * layer_weights[i].t()) )/(actuation[i].n_cols+1);
 outp(netin[i], "netin[i]");           
                std::cout << netin[i] << std::endl;
            
                cout << "------------------------------------ Activation out of L" << i << endl;
                actuation[i+1] = sigmoid(netin[i]);
 outp(actuation[i+1], "actuation[i+1]");           
                std::cout << actuation[i+1] << std::endl;
            }
            std::cout << "Test Final output : " << endl << actuation[NumberOfLayers-1] << std::endl;
            std::cout << "Test Expec output : " << endl << tgt << std::endl;
    
        delete[] memptr;
#endif
}
