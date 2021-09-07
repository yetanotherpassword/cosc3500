#include <iostream>
#include <iomanip>
#include <cmath>
#define ARMA_64BIT_WORD
#include <armadillo>
#undef TESTING
// g++ armo.cpp -g -o armo -std=c++11 -O2 -larmadillo

// requires armodillo
// http://arma.sourceforge.net/docs.html#element_access

 
using namespace arma;
using namespace std;

        // https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

rowvec sigmoid( rowvec  & net)
{

   rowvec norml = net; // / net.n_cols;  // normalise
   rowvec out = 1/(1+exp(-norml));
   out.insert_cols(out.n_cols, ones<colvec>(1)); // add bias signal (1)
   return out;
}

void print_an_image(unsigned char * c, int i)
{
     cout << "This is a : " << i << endl;
     for (int i=0;i<784;i++)
     {
       if (i%28==0)
         cout << endl;
       cout  << hex << std::setfill('0') << std::setw(2) << (unsigned int)c[i] << dec << " ";
     }
     cout << endl;
}
   

void print_images(unsigned char * c,  int size)
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


unsigned char * load_file(string filename, string labels, unsigned char * * labs)
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
         //print_images(memblock, size);
 
    }
    inFile.close();
///////////////////////////////////////////////
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

void load_an_image(int seq, unsigned char * mptr, rowvec & rv, rowvec & t, unsigned char * lp)
{
    int offset=16;
    int start=(784*seq)+offset;
    rv.set_size(784);
    for (int i=0;i<784;i++)
    {
        rv(i) = ((float) mptr[start+i])/255.0;
    }

    int i=(int) lp[8+seq];

    //print_an_image(&mptr[start], i);

    t=zeros<rowvec>(10);
    t(i)=1;

 //   if (i==0) t={0,0,0,0,0,0,0,0,0,1}; else if (i==1) t={0,0,0,0,0,0,0,0,1,0}; else if (i==2) t={0,0,0,0,0,0,0,1,0,0};
//    else if (i==3) t={0,0,0,0,0,0,1,0,0,0}; else if (i==4) t={0,0,0,0,0,1,0,0,0,0}; else if (i==5) t={0,0,0,0,1,0,0,0,0,0};
//    else if (i==6) t={0,0,0,1,0,0,0,0,0,0}; else if (i==7) t={0,0,1,0,0,0,0,0,0,0}; else if (i==8) t={0,1,0,0,0,0,0,0,0,0};
//    else if (i==9) t={1,0,0,0,0,0,0,0,0,0}; else { cout << "Error expected between 0-9 but got " << i << endl; exit; 
 //   }
}
void outp(mat m, string s)
{
   cout << "vector:" << s << " cols=" << m.n_cols <<   " rows=" << m.n_rows << endl;

}
void outp(rowvec v, string s)
{
   cout << "vector:" << s << " cols=" << v.n_cols <<   " rows=" << v.n_rows << endl;

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
    unsigned char * memptr = load_file("./tmp/cosc3500/data/neural-networks-and-deep-learning/src/mnist-neural-network-plain-c/data/train-images-idx3-ubyte", "./tmp/cosc3500/data/neural-networks-and-deep-learning/src/mnist-neural-network-plain-c/data/train-labels-idx1-ubyte", &labptr);
   #ifdef TESTING
    unsigned int NumberOfLayers=3;
    int nodes[NumberOfLayers]={2,2,2};
#else
    unsigned int NumberOfLayers=4;
    int nodes[NumberOfLayers]={784, 1000, 1000, 10};  
#endif
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
    */
   rowvec tmprw; 
    rowvec tgt; 
    vector<rowvec> netin;
    vector<rowvec> actuation;
    vector<rowvec> deltafn;
    vector<rowvec> ftick;
    vector<rowvec> layer_bias;
    vector<mat> layer_weights;
    vector<mat> weight_updates;
    vector<mat> new_layer_weights;
    rowvec tmpbias; 
    mat tmpwgt; 
    colvec v;

    for (int i=0;i < NumberOfLayers; i++)
    {
         netin.push_back({});   // size=nodes[i],1
         actuation.push_back({}); // size= nodes[i],1
         deltafn.push_back({});
         ftick.push_back({});
#ifdef TESTING
         if (i==0)
         {
            tmpwgt = ones<mat>(nodes[i],1); // (not used) but no weight change to inputs
            tmpbias=zeros<rowvec>(nodes[i]);  // (not used) but no bias to inputs
         }
         else
         {
            if (i==1)
            {
            tmpwgt = {{ 0.15, 0.2 }, {0.25, 0.3}};
            tmpbias=0.35;
            colvec v = ones<vec>(tmpwgt.n_rows)*tmpbias;
            tmpwgt.insert_cols(tmpwgt.n_cols, v);
            }
            else if (i==2)
            {
            tmpwgt = { { 0.4, 0.45 }, {0.5, 0.55}};
            tmpbias=0.6;
            colvec v = ones<vec>(tmpwgt.n_rows)*tmpbias;
            tmpwgt.insert_cols(tmpwgt.n_cols, v);
            }
         }
#else
         if (i==0)
         {
            tmpwgt = ones<mat>(nodes[i],1); // (not used) but no weight change to inputs
            //tmpbias=zeros<rowvec>(nodes[i]);  // (not used) but no bias to inputs
            tmpbias=rand();
            v = ones<vec>(tmpwgt.n_rows)*tmpbias;
         }
         else
         {
            tmpwgt = randu<mat>(nodes[i],nodes[i-1]);
            tmpbias=rand();
            v = ones<vec>(tmpwgt.n_rows)*tmpbias;
         }
#endif
            tmpwgt.insert_cols(tmpwgt.n_cols, v);
         layer_weights.push_back( tmpwgt );
cout << "Layer_weights[ "<<i <<"]: has rows="<< layer_weights[i].n_rows << " and cols=" << layer_weights[i].n_cols << endl;
//cout << layer_weights[i-1] << endl;
         new_layer_weights.push_back({});
         weight_updates.push_back({});
         layer_bias.push_back( tmpbias );
cout << "Layer_bias[ "<<i <<"]: has rows="<< layer_bias[i].n_rows << " and cols=" << layer_bias[i].n_cols << endl;
//cout << layer_bias[i-1] << endl;
    }
    
   /* 
    layer_weights[1]=L1_Weights;
    layer_weights[2]=L2_Weights;
    layer_bias[1] = L1Bias;
    layer_bias[2] = L2Bias;
   */ 
#ifdef TESTING
            actuation[0]={0.05, 0.1};
            tgt={0.01, 0.99};
#else
            load_an_image(0, memptr, actuation[0], tgt, labptr);
#endif
            tgt.insert_cols(tgt.n_cols, ones<colvec>(1));
            actuation[0].insert_cols(actuation[0].n_cols, ones<colvec>(1));
tmprw=actuation[0];
for (int y=1;y<10000;y++)
{
            load_an_image(y, memptr, actuation[0], tgt, labptr);
            tgt.insert_cols(tgt.n_cols, ones<colvec>(1));
            actuation[0].insert_cols(actuation[0].n_cols, ones<colvec>(1));
            for (int i=1;i<NumberOfLayers;i++)  // only n-1 transitions between n layers
            {
                // sum layer 1 weighted input
                cout << "------------------------------------ Net Input into L" << i << endl;
//cout << actuation[i-1] << " * " << layer_weights[i].t() <<  " == " << endl;
//cout << " == " << (actuation[i-1] * layer_weights[i].t()) << endl;
                netin[i] =  (actuation[i-1] * layer_weights[i].t());
 outp(netin[i], "netin[i]");           
//                std::cout << netin[i] << std::endl;
            
                cout << "------------------------------------ Activation out of L" << i << endl;
                actuation[i] = sigmoid(netin[i]);
 outp(actuation[i], "actuation[i]");           
//                std::cout << actuation[i] << std::endl;
             //   actuation[i-1].shed_col(actuation[i-1].n_cols-1);
            }
        //    actuation[NumberOfLayers-1].shed_col(actuation[NumberOfLayers-1].n_cols-1);
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
//               std::cout << deltafn[i+1]<< std::endl;
                
                cout << "------------------------------------ Updates to weights at Layer" << i << endl;
                weight_updates[i]  =  deltafn[i+1].t() * actuation[i];
//weight_updates[i].shed_row(weight_updates[i].n_rows-1);
//                std::cout << weight_updates[i]<< std::endl;
                
                cout << "------------------------------------ New weights are at Layer"<<i << endl;
                new_layer_weights[i+1]  =  layer_weights[i+1] + (eta *  weight_updates[i]) ;
//                std::cout << new_layer_weights[i+1] << std::endl;
                
                here:
                ftick[i] = -actuation[i] + 1;
                ftick[i] = ftick[i] % (actuation[i]);  //element wise multiply
                deltafn[i] = deltafn[i+1]*layer_weights[i+1];
                deltafn[i] = deltafn[i] % ftick[i];
                 deltafn[i].shed_col(deltafn[i].n_cols-1);
            }
            for (int i=1;i<=NumberOfLayers;i++)
            {
               layer_weights[i] =  new_layer_weights[i];
            }
}            

        
#if 0
    forward_feed:
        for (int y=0;y < 60000; y++)
        {
            load_an_image(y, memptr, actuation[0], tgt, labptr);

            cout << "*********** For iteration # " << y << " *******************" << endl;
            cout << "------------------------------------ FORWARD FEED" << endl;
            cout << "------------------------------------ INPUT Signals" << endl;
            
//            std::cout << actuation[0]<< std::endl;
           // actuation[0] = L0_Input;
 outp(actuation[0], "Actuation[0]");           
            
            for (int i=1;i<NumberOfLayers;i++)  // only n-1 transitions between n layers
            {
                // sum layer 1 weighted input
                cout << "------------------------------------ Net Input into L" << i << endl;
                netin[i] =  ((actuation[i-1] * layer_weights[i].t()) + layer_bias[i])/(actuation[i-1].n_cols+1);
 outp(netin[i], "netin[i]");           
//                std::cout << netin[i] << std::endl;
            
                cout << "------------------------------------ Activation out of L" << i << endl;
                actuation[i] = sigmoid(netin[i]);
 outp(actuation[i], "actuation[i]");           
//                std::cout << actuation[i] << std::endl;
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
 outp(deltafn[NumberOfLayers-1], "deltafn[NumberOfLayers-1]");
             
            for (int i=NumberOfLayers-2;i>=0;i--)
            {
                cout << "------------------------------------ Delta of Layer"<< i << endl;
 //               std::cout << deltafn[i+1]<< std::endl;
                
                cout << "------------------------------------ Updates to weights at Layer" << i << endl;
                weight_updates[i]  =  deltafn[i+1].t() * actuation[i];
//                std::cout << weight_updates[i]<< std::endl;
                
                cout << "------------------------------------ New weights are at Layer"<<i << endl;
                new_layer_weights[i+1]  =  layer_weights[i+1] + (eta *  weight_updates[i]) ;
//                std::cout << new_layer_weights[i+1] << std::endl;
                
                here:
                ftick[i] = -actuation[i] + 1;
                ftick[i] = ftick[i] % (actuation[i]);  //element wise multiply
                deltafn[i] = deltafn[i+1]*layer_weights[i+1];
                deltafn[i] = deltafn[i] % ftick[i];
            }
            for (int i=1;i<=NumberOfLayers;i++)
            {
               layer_weights[i] =  new_layer_weights[i];
            }
            
        }
#endif
// test
        actuation[0]=tmprw;
            for (int i=1;i<NumberOfLayers;i++)  // only n-1 transitions between n layers
            {
                // sum layer 1 weighted input
                cout << "------------------------------------ Net Input into L" << i << endl;
                netin[i] =  ((actuation[i-1] * layer_weights[i].t()) + layer_bias[i])/(actuation[i-1].n_cols+1);
 outp(netin[i], "netin[i]");           
//                std::cout << netin[i] << std::endl;
            
                cout << "------------------------------------ Activation out of L" << i << endl;
                actuation[i] = sigmoid(netin[i]);
 outp(actuation[i], "actuation[i]");           
//                std::cout << actuation[i] << std::endl;
            }
                std::cout << "Final output : " << endl << actuation[NumberOfLayers-1] << std::endl;
                std::cout << "Expec output : " << endl << tgt << std::endl;
    
        delete[] memptr;
}
