#include <iostream>
#include <cmath>
#include <Eigen/Dense>
// g++  -I /usr/local/include/eigen3/ eigen_eg3.cpp
 
using namespace Eigen;
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
void sigmoid( RowVector2d & net)
{
   for (int i=0;i<net.cols();i++)
      net(i) = 1/(1+exp(-net(i)));
}
int main()
{
float L0_Bias_weight = 0;
float L1_Bias_weight = 0.35;
float L2_Bias_weight = 0.60;
float L3_Bias_weight = 0.60;
float Bias = 1.0;

RowVector2d Linput(0.05, 0.1);
RowVector2d LinputBias(0, 0);
RowVector2d L1Bias(0.35, 0.35);
RowVector2d LoutputBias(0.6, 0.6);

RowVector2d q;
RowVector2d w;

MatrixXd L1(2,2);
L1 <<  0.15, 0.25,   // L1N1 weights for I1 (col 1), I2 (col 2)and Bias
       0.2, 0.3;   

MatrixXd Loutput(2,2);
Loutput    <<  0.4, 0.5, 
               0.45, 0.55;

cout << "------------------------------------ INPUT" << endl;

std::cout << Linput << std::endl;
cout << "------------------------------------ Layer1" << endl;
std::cout << L1 << std::endl;
cout << "------------------------------------ Output" << endl;
std::cout << Loutput << std::endl;
cout << "------------------------------------" << endl;




q= (Linput * L1) +L1Bias;
sigmoid(q);
cout << "------------------------------------ Input * L1 r="<< Linput.rows()<< " c="<< Linput.cols() << endl;
cout << "------------------------------------ Input * L1 r="<< L1.rows()<< " c="<< L1.cols() << endl;
cout << "------------------------------------ Input * L1 r="<< q.rows()<< " c="<< q.cols() << endl;
cout << "------------------------------------ Input * L1 r="<< Loutput.rows()<< " c="<< Loutput.cols() << endl;
std::cout << q<< std::endl;
w = (q * Loutput) + LoutputBias;

sigmoid(w);

cout << "------------------------------------ w" << endl;
std::cout << w<< std::endl;
}
