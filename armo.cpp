#include <iostream>
#include <cmath>
#include <armadillo>
// g++ armo.cpp -g -o armo -std=c++11 -O2 -larmadillo

// requires armodillo
// http://arma.sourceforge.net/docs.html#element_access

 
using namespace arma;
using namespace std;

/*
int main()
{
        Signals result;
        Signals input_train = { 0.05, 0.1 };
        Signals train_answer = { 0.01, 0.99};
        //  perceptron.show_net();
        result = perceptron.forward_feed(input_train);
        perceptron.back_prop(result, train_answer);

        for (int i = 0; i < result.size(); i++)
                cout << result[i] << " is Output " << i << endl;
        for (int layer=0;layer <perceptron.graph.size(); layer++)
        {
              cout << " layer== " << layer << endl;
           for (int d=0; d<perceptron.graph[layer].delta.size(); d++)
        {
              cout  <<" d== " << d << " delta=" << perceptron.graph[layer].delta[d] << endl;
        }
        }
        // reference for verify
        // https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
}
*/ 
rowvec sigmoid( rowvec  net)
{
   rowvec tmp=net;
   for (int i=0;i<net.n_cols;i++)
      tmp(i) = 1/(1+exp(-net(i)));
   return tmp;
}
int main()
{
float eta=0.5;
float L0_Bias_weight = 0;
float L1_Bias_weight = 0.35;
float L2_Bias_weight = 0.60;
float L3_Bias_weight = 0.60;
float Bias = 1.0;

rowvec Linput={0.05, 0.1};
rowvec LinputBias={0, 0};
rowvec L1Bias={0.35, 0.35};
rowvec LoutputBias={0.6, 0.6};
rowvec expected_answer = { 0.01, 0.99};

rowvec AOnes={1,1};
rowvec n1, n2;
rowvec a1, a2;
rowvec delta1, delta2;
rowvec ftick1, ftick2;

mat updates1(2,2);
mat updates2(2,2);
mat new_weights1(2,2);
mat new_weights2(2,2);
mat L1 = {{   0.15, 0.25},   // L1N1 weights for I1 (col 1), I2 (col 2)and Bias
       {0.2, 0.3}};   

mat Loutput ={{  0.4, 0.5}, 
               {0.45, 0.55}};
forward_feed:
cout << "------------------------------------ Layer1 weights" << endl;
std::cout << L1 << std::endl;
cout << "------------------------------------ Output weights" << endl;
std::cout << Loutput << std::endl;
cout << "------------------------------------" << endl;
cout << "------------------------------------ INPUT Signals" << endl;

std::cout << Linput << std::endl;



// sum layer 1 weighted input
cout << "------------------------------------ Net Input into L1" << endl;
n1= (Linput * L1) +L1Bias;
std::cout << n1<< std::endl;

cout << "------------------------------------ Activation out of L1" << endl;
a1=sigmoid(n1);
std::cout << a1<< std::endl;

cout << "------------------------------------ Net Input into Output Layer" << endl;
n2 = (a1 * Loutput) + LoutputBias;
std::cout << n2<< std::endl;

cout << "------------------------------------ Activation out of Output Layer" << endl;
a2=sigmoid(n2);
std::cout << a2<< std::endl;
//////////////////////////// forward feed end



cout << "------------------------------------ BACKPROPAGATION" << endl;
cout << "------------------------------------ Delta of Output Layer" << endl;
ftick1=(AOnes-a2);
ftick1=ftick1 % (a2);  //element wise multiply
delta1 = (expected_answer - a2)%(ftick1);
std::cout << delta1<< std::endl;

cout << "------------------------------------ Updates to weights at Output Layer" << endl;
updates1 = delta1.t() * a1;
std::cout << updates1<< std::endl;

cout << "------------------------------------ New updates are at Output Layer" << endl;
new_weights1 = Loutput + (eta *  updates1.t()) ;
std::cout << new_weights1 << std::endl;

here:
cout << "------------------------------------ Delta of Hidden Layer" << endl;
ftick2=(AOnes-Linput);
ftick2=ftick2 % (Linput);  //element wise multiply
delta2=delta1*L1;
delta2=delta2 % ftick2;
std::cout << delta2<< std::endl;
//std::cout << ftick2<< std::endl;

cout << "------------------------------------ Updates to weights at Hidden Layer" << endl;
updates2 = delta2.t() * a1;
std::cout << updates2<< std::endl;

cout << "------------------------------------ New updates are at Hidden Layer" << endl;
new_weights2 = L1 + (eta *  updates2.t()) ;
std::cout << new_weights2 << std::endl;

delta1=delta2;  // if there was another layer ; then goto here:, with L1 and Linput decremented
// then when hit input layer break, and do

L1 = new_weights1;
Loutput = new_weights2;

// goto forward_feed:


}
