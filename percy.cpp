#include <iostream>
#include <vector>
#include <cmath>

using namespace std;
typedef vector <float> Weights;
typedef Weights Signals;
typedef vector <Weights> Layer;
typedef vector <Layer> Graph;

class neuron {
	float thres_func(float i)
	{
		float outs = 1/(1+exp(-i));
		return outs;
	};
   public:
	vector <float> input_weights;
	float output;
	float bias_val;
	neuron(Weights w) {
		int siz=w.size();

#ifdef DEBUGON
		   cout << "Creating neuron with (at most) " << w.size() << " inputs"<< endl;
#endif
		   for (int i=0;i<siz-1;i++)
	   	{
#ifdef DEBUGON
	   		cout << "Pushing back weight of " << w[i] << endl;
#endif
			   input_weights.push_back(w[i]);
		   }
#ifdef DEBUGON
	   	cout << "Including Bias weighting of " << w[siz-1] << endl;
#endif
	   	bias_val = w[siz-1];
	};

	float input_neuron(Signals s)
	{
		float outp=bias_val;
		for (int i=0;i<s.size();i++)
		{
#ifdef DEBUGON
	            cout << "outp=="<<outp << ",   += " << input_weights[i] << " * " << s[i] << endl;
#endif
                    outp += input_weights[i] * s[i];
		}
		output = thres_func(outp);
#ifdef DEBUGON
		cout << "outp=="<<outp<<endl;
		cout << "output=="<<output<<endl;
#endif
		return output;
	};
};

class neuron_layer {
 public:
   vector <neuron> nodes;
   neuron_layer (Layer nl) {
	   int siz=nl.size();
	   cout << "Creating neuron layer of " << siz-1 << " nodes" << endl;
	   for (int i=0;i<siz;i++)
	   {
		   neuron tmp(nl[i]);
		   nodes.push_back(tmp);
	   }
   }
   Signals input_layer (Signals in)
   {
	   Signals out;
	   for (int i=0;i<nodes.size();i++)
	   {
		   out.push_back(nodes[i].input_neuron(in));
	   }
	   return out;
   };
};

class neuron_network {
 public:
   vector <neuron_layer> graph;
   int num_inputs;
   int num_outputs;
   neuron_network(Graph g)
   {
	   cout << "Creating neuron network with " << g.size() << " layers" << endl;
	   for (int i=0;i<g.size();i++)
	   {
	      cout << "Neuron layer " << i << " : ";
		   neuron_layer tmp(g[i]);
		   graph.push_back(tmp);
	   }
	   num_inputs=g[0].size();
	   num_outputs=g[g.size()-1].size();
	   cout << "Network has " << num_inputs << " inputs and " << num_outputs << " outputs" << endl;
   }
   void show_net()
   {
      for (int i=0;i<graph.size();i++)
	  for (int j=0;j<graph[i].nodes.size();j++)
	     for (int k=0;k<graph[i].nodes[j].input_weights.size();k++)
		  cout << "Layer "<< i << " Node " << j << " input weight "<< k << " is " << graph[i].nodes[j].input_weights[k] << endl;
   };
   Signals forward_feed(Signals inputs)
   {
        if (inputs.size() != num_inputs)
	{
		cout << "Error: Expected " << num_inputs << " inputs, but got " << inputs.size() << endl;
		exit(1);
	}
        Signals output=graph[0].input_layer(inputs);
	for (int i=1;i<graph.size();i++)
	{
		output = graph[i].input_layer(output);
	}
	return output;
   };
};

float L1_Bias=0.35;
float L2_Bias=0.60;
// Layer 1 has 3 nodes, getting input from up to 4 nodes
Layer L1={ 
/* node 1 input weights from Inputs 1,2 */      { 0.15, 0.2, L1_Bias}, // L1N1 has inputs from I1, I2 and I4
/* node 2 input weights from Inputs 1,2 */      { 0.25, 0.3, L1_Bias} // L1N2 has inputs from I1, I2 and I3
};
// Layer 2 has 4 nodes, getting input from up to 3 nodes (ie the output from L1 nodes)
Layer L2={ 
 { 0.4, 0.45, L2_Bias}, //L2N1 has input fro L1N1, L1N2
 { 0.5, 0.55, L2_Bias} //L2N2 has input fro L1N1, L1N2
};

neuron_network perceptron({L1, L2});

int main()
{
	Signals result;
  //  perceptron.show_net();
    result = perceptron.forward_feed({0.05, 0.1});
    cout << "result.size=" << result.size()<< endl;
    for (int i=0;i<result.size();i++)
	    cout << result[i] << " is Output " << i << endl;
// reference for verify 
// https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
}
