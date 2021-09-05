#include <iostream>
#include <cmath>
#define ARMA_64BIT_WORD
#include <armadillo>
// g++ armo.cpp -g -o armo -std=c++11 -O2 -larmadillo

// requires armodillo
// http://arma.sourceforge.net/docs.html#element_access

 
using namespace arma;
using namespace std;

        // https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

rowvec sigmoid( rowvec  net)
{
   rowvec tmp = net;
   for (int i = 0;i<net.n_cols;i++)
      tmp(i)  =  1/(1+exp(-net(i)));
   return tmp;
}

/*
void print_images(char * c,  size)
{
    for (int i=16;i<size;i++)
    {
       if (((i-16)%28)==0)
           cout << endl;
       if (((i-16)%784)==0)
           cout << endl << "Image : " << dec << ((i-16)/784)+1 << endl;
       cout << hex << std::setfill('0') << std::setw(2) << (unsigned int)c[i] << " ";
    }
}

*/
unsigned char * load_file(string filename, string labels, unsigned char * labs)
{
    unsigned char * memblock;
    ifstream inFile;
    streampos size;

    cout << "Using file '" << filename << "'" << endl;


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
        // print_images(memblock, size);
 
    }
    inFile.close();
///////////////////////////////////////////////
    inFile.open(labels, ios::in|ios::binary|ios::ate);
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
        labs = new unsigned char [size];
        inFile.seekg (0, ios::beg);
        inFile.read ((char *)labs, size);
        inFile.close();

        cout << "the entire file content is in memory, all " << size << " bytes of it" << endl;
        // print_images(memblock, size);
 
    }
    inFile.close();
    return memblock;
}

void load_an_image(int seq, unsigned char * mptr, rowvec & rv, rowvec & t, unsigned char * lp)
{
    int offset=16;
    int start=(748*seq)+offset;
    for (int i=0;i<748;i++)
        rv(i) = ((float) mptr[start+i])/255.0;
    if (lp[offset+seq]==0)
       t={0,0,0,0,0,0,0,0,0,0};
    else if (lp[offset+seq]==1)
       t={0,0,0,0,0,0,0,0,0,1};
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
    unsigned char * memptr = load_file("./tmp/cosc3500/data/neural-networks-and-deep-learning/src/mnist-neural-network-plain-c/data/train-images-idx3-ubyte", "./tmp/cosc3500/data/neural-networks-and-deep-learning/src/mnist-neural-network-plain-c/data/train-labels-idx1-ubyte", labptr);
    
    unsigned int NumberOfLayers=4;
    int nodes[NumberOfLayers]={784, 100, 100, 10};  

    double eta = 0.5;

    
    /*
    rowvec L0_Input =     { 0.05, 0.1 };  // Inputs to forward feed
    
    
    rowvec L0Bias = { 0, 0 };
    
    rowvec L1Bias = { 0.35, 0.35 };
    rowvec L2Bias = { 0.6,  0.6 };
    
    rowvec L2_target  =  { 0.01, 0.99 };
    mat L1_Weights  =  {   { 0.15, 0.25 },   // L1N1 weights for I1 (col 1), I2 (col 2)and Bias
                           { 0.2,  0.3 }    // to/from == row/col 1000x784
               };   
                       {
    
    mat L2_Weights  = {    {  0.4,  0.5  }, 
                           {  0.45, 0.55 }
              };
    
    
    rowvec n1, n2;
    rowvec a1, a2;
    rowvec delta1, delta2;
    rowvec ftick1, ftick2;
    
    mat updates1;
    mat updates2;
    mat new_weights1;
    mat new_weights2;
    */
    
    rowvec tgt; 
    vector<rowvec> netin;
    vector<rowvec> actuation;
    vector<rowvec> deltafn;
    vector<rowvec> ftick;
    vector<rowvec> layer_bias;
    vector<mat> layer_weights;
    vector<mat> weight_updates;
    vector<mat> new_layer_weights;
    
    //int nodes[]={784, 1000, 1000, 10};  
    for (int i=1;i <= NumberOfLayers; i++)
    {
         netin.push_back({});
         actuation.push_back({});
         deltafn.push_back({});
         ftick.push_back({});
         layer_weights.push_back( randu(nodes[i],nodes[i-1]) );
         new_layer_weights.push_back({});
         weight_updates.push_back({});
         rowvec tmp=randu<rowvec>(nodes[i-1]);
         layer_bias.push_back( tmp );
    }
    
   /* 
    layer_weights[1]=L1_Weights;
    layer_weights[2]=L2_Weights;
    layer_bias[1] = L1Bias;
    layer_bias[2] = L2Bias;
   */ 
    forward_feed:
        for (int y=0;y < 10; y++)
        {
            load_an_image(y, memptr, actuation[0], tgt, labptr);
            cout << "*********** For iteration # " << y << " *******************" << endl;
            cout << "------------------------------------ FORWARD FEED" << endl;
            cout << "------------------------------------ INPUT Signals" << endl;
            
            std::cout << actuation[0]<< std::endl;
           // actuation[0] = L0_Input;
            
            
            for (int i=1;i<=NumberOfLayers;i++)
            {
                // sum layer 1 weighted input
                cout << "------------------------------------ Net Input into L" << i << endl;
                netin[i] =  (actuation[i-1] * layer_weights[i]) + layer_bias[i];
                std::cout << netin[i] << std::endl;
            
                cout << "------------------------------------ Activation out of L" << i << endl;
                actuation[i] = sigmoid(netin[i]);
                std::cout << actuation[i] << std::endl;
            }
    
            //////////////////////////// forward feed end
            
            cout << "------------------------------------ BACK PROPAGATION" << endl;
            
            ftick[NumberOfLayers] = -actuation[NumberOfLayers] + 1;
            ftick[NumberOfLayers] = ftick[NumberOfLayers] % (actuation[NumberOfLayers]);  //element wise multiply
            deltafn[NumberOfLayers]  =  (tgt - actuation[NumberOfLayers])%(ftick[NumberOfLayers]);
             
            for (int i=NumberOfLayers;i>0;i--)
            {
                cout << "------------------------------------ Delta of Layer"<< i << endl;
                std::cout << deltafn[i]<< std::endl;
                
                cout << "------------------------------------ Updates to weights at Layer" << i << endl;
                weight_updates[i]  =  deltafn[i].t() * actuation[i];
                std::cout << weight_updates[i]<< std::endl;
                
                cout << "------------------------------------ New weights are at Layer"<<i << endl;
                new_layer_weights[i]  =  layer_weights[i] + (eta *  weight_updates[i].t()) ;
                std::cout << new_layer_weights[i] << std::endl;
                
                here:
                if (i > 1)
                {
                    cout << "------------------------------------ Delta of Hidden Layer" << endl;
                    ftick[i-1] = -actuation[i-1] + 1;
                    ftick[i-1] = ftick[i-1] % (actuation[i-1]);  //element wise multiply
                    deltafn[i-1] = deltafn[i]*layer_weights[i-1];
                    deltafn[i-1] = deltafn[i-1] % ftick[i-1];
                }        
            }
            for (int i=1;i<=NumberOfLayers;i++)
            {
               layer_weights[i] =  new_layer_weights[i];
            }
            
        }
        delete[] memptr;
}
