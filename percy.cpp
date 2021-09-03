#include <iostream>
#include <vector>
#include <cmath>
using namespace std;
typedef vector<float> RowVector;;
typedef vector<float> Weights;
typedef Weights Signals;
typedef Weights Partials;
typedef vector<Weights> Layers;
typedef vector<Layers> Graph;

class neuron
{
        float eta = 0.5;
        float thres_func(float i)
        {
                float outs = 1 / (1 + exp(-i));
                return outs;
        };
        public:
        vector<float> input_weights;
        vector<float> output_links;
        vector<float> inputs;
        vector<float> new_weights;
        Partials dNdW, dEdW, dEdO, dOdN;
        float netin;
        float output;
        float bias_val;
        string out_label;
        neuron(Weights w, float bias_wgt)
        {
                int siz = w.size();

                for (int i = 0; i < siz; i++)
                {
                        input_weights.push_back(w[i]);
                }
                bias_val = bias_wgt;
        };

        float input_neuron(Signals s, float bias_wgt, float & ni)
        {
                inputs.clear();
                netin = bias_wgt;  // bias_weight * 1
                for (int i = 0; i < s.size(); i++)
                {
                        inputs.push_back(s[i]);
                        netin += input_weights[i] *s[i];
                }
                ni = netin;
                output = thres_func(netin);
                return output;
        };
/*
  \ signal_0
   \ input_weight[0]
    \
     /----\
    / net= \
   (        )---------------- output
    \      /
     \----/
    /
   / signal_1
  / input_weight[1]
*/
        void calc_partial_last(float ans)
        {
                float deeEdeeO, deeOdeeN, deeNdeeW, deeEdeeW;
                //deltai.clear();
                dNdW.clear();
                dEdW.clear();
                dEdO.clear();
                dOdN.clear();
                new_weights.clear();
cout << " out="<<output<<endl;
                deeEdeeO = output - ans; // derivative of (total) error wrt output 
                deeOdeeN = output *(1 - output); // derivative of output wrt net input (sigmoid)
                for (int i = 0; i < input_weights.size(); i++)
                {
                   if (input_weights[i] != 0) //weight of zero means no link from node i, in layer to left
                   {
cout << "     has connection to prev layer Node : " << i+1 << " giving Input "<< inputs[i] << " with weight  = " << input_weights[i] << endl;
                        deeNdeeW = inputs[i];  // derivative of net input wrt to weight (is just the weight)
                        //////////////////////////////////////////////////////////////
                        // chain rule gives us the derivative of error wrt each weight
                        deeEdeeW = deeEdeeO *deeOdeeN * deeNdeeW;
                        //////////////////////////////////////////////////////////////
                   }
                   else
                   {
                       // no connection
                        deeNdeeW = 0;
                        deeEdeeW = 0;
                   }
                   new_weights.push_back(input_weights[i] - eta *deeEdeeW);
                   dNdW.push_back(deeNdeeW);
                   dEdW.push_back(deeEdeeW);
                }
                dEdO.push_back(deeEdeeO);
                dOdN.push_back(deeOdeeN);
        };
        void calc_partial()
        {
                float deeEdeeN, deeEdeeO, deeOdeeN, deeNdeeW, deeEdeeW, deeNdeeO;
                new_weights.clear();
                deeEdeeN = deeEdeeO * deeOdeeN;
                for (int i = 0; i < input_weights.size(); i++)
                {
                        deeNdeeO = input_weights[i];
                        deeNdeeW = inputs[i];
                        deeEdeeW = deeEdeeO *deeOdeeN * deeNdeeW;
                        new_weights.push_back(input_weights[i] - eta *deeEdeeW);
                }
        };
};

class neuron_layer
{
        public:
                vector<neuron> nodes;
        float layer_bias_weight;
        Layers nodes_weights;
        Signals delta;
        Signals outputs;
        Signals netins;
        void mark_node_output(int layer_from, int node_from, int layer_to, int node_to, float val)
        {
            string tmp_str = to_string(node_to);
             if (val == 0)
                tmp_str=" ";
 
             if ( this->nodes[node_from].out_label.length() == 0)
                this->nodes[node_from].out_label = "L"+to_string(layer_from)+":N"+to_string(node_from) + "->L" + to_string(layer_to) + ":N" +tmp_str;
             else
                this->nodes[node_from].out_label += ",N" + tmp_str;
        }
        void list_io ( neuron_layer nextLayer, int m)
        {
             // for each node in 'previous' layer
             // go through 'new' layers nodes
                for (int j=0;j<nextLayer.nodes.size();j++)
                {
                     for (int i=0;i<this->nodes.size();i++)
                     {
                       if (nextLayer.nodes[j].input_weights.size() != this->nodes.size())
                       {
                         cout << "Error in setup: nextLayer.nodes["<< j << "].input_weights.size() " << nextLayer.nodes[j].input_weights.size()<< " != prevLayer->nodes.size()= " << this->nodes.size() << endl;
                         exit (-1);
                       }
                       this->mark_node_output(m-1,i,m,j,nextLayer.nodes[j].input_weights[i]);
                     }
                }
                     for (int i=0;i<this->nodes.size();i++)
                        //cout <<"Node "<< i << " outputs to " << this->nodes[i].out_label << endl;
                        cout << this->nodes[i].out_label << endl;
        }   
        neuron_layer(Layers nl, float bias_wgt)
        {
                nodes_weights = nl;
                int siz = nl.size();
                layer_bias_weight = bias_wgt;
                cout << "Creating neuron layer of " << siz << " nodes" << endl;
                for (int i = 0; i < siz; i++)
                {
                        neuron tmp(nl[i], bias_wgt);
                        nodes.push_back(tmp);
                }
        }
        Signals input_layer(Signals in)
        {
                Signals out;
                for (int i = 0; i < nodes.size(); i++)
                {
                        float ni;
                        out.push_back(nodes[i].input_neuron(in, layer_bias_weight, ni));
                        netins.push_back(ni);
                }
                return out;
        };
        void descend_grad_last(Signals ans, Signals & delta, Signals prev_outputs)
        {
cout << "For output layer:" << endl;
                delta.clear();
                for (int i = 0; i < nodes.size(); i++)
                {
                     float m = 1;
                     float delta_i=((ans[i] - outputs[i])* (outputs[i] *(1 - outputs[i])))/m;
cout << "   Node " << i+1<<" has net in = " << nodes[i].netin << " giving output = " << nodes[i].output << " Comprising from : +Bias = " << layer_bias_weight  << endl;
                    //   nodes[i].calc_partial_last(out[i]);
                     delta.push_back(delta_i);

                }
/*
                for (int i = 0; i < nodes.size(); i++)
                {
                     float m = 1;
                     float delta_i=((ans[i] - nodes[i].output)* (nodes[i].output *(1 - nodes[i].output)))/m;
cout << "   Node " << i+1<<" has net in = " << nodes[i].netin << " giving output = " << nodes[i].output << " Comprising from : +Bias = " << layer_bias_weight  << endl;
                    //   nodes[i].calc_partial_last(out[i]);
                     delta.push_back(delta_i);
                }
*/
        };
        Signals descend_grad()
        {
                Signals out;
                for (int i = 0; i < nodes.size(); i++)
                {
                        nodes[i].calc_partial();
                }
                return out;
        };
};

class neuron_network
{
    public:
        vector<neuron_layer> graph;
        int num_inputs;
        int num_outputs;
        //Signals outputs; // aka next_layer_inputs
        Signals delta;
        neuron_network(Graph g, RowVector b)
        {
                if (b.size() != g.size())
                {
                     cout << "Error bias length " << b.size() << " doesnt match number of layers " << g.size() << " layers" << endl;
                     exit;
                }
                cout << "Creating neuron network with " << g.size() << " layers" << endl;
                for (int i = 0; i < g.size(); i++)
                {
                        cout << "Neuron layer " << i << " : ";
                        neuron_layer tmp(g[i], b[i]);
                        graph.push_back(tmp);
                        if (i > 0)
                        {
                           graph[i-1].list_io(graph[i], i);
                           //graph[i].list_io(graph[i-1], i);
                        }
                }
                num_inputs = g[0].size();
                num_outputs = g[g.size() - 1].size();
                cout << "Network has " << num_inputs << " inputs and " << num_outputs << " outputs" << endl;
        }
        void show_net()
        {
                for (int i = 0; i < graph.size(); i++)
                        for (int j = 0; j < graph[i].nodes.size(); j++)
                                for (int k = 0; k < graph[i].nodes[j].input_weights.size(); k++)
                                        cout << "Layer " << i << " Node " << j << " input weight " << k << " is " << graph[i].nodes[j].input_weights[k] << endl;
        };
        void forward_feed(Signals inputs)
        {
                if (inputs.size() != num_inputs)
                {
                        cout << "Error: Expected " << num_inputs << " inputs, but got " << inputs.size() << endl;
                        exit(1);
                }
                graph[0].outputs = graph[0].input_layer(inputs);
                for (int i = 1; i < graph.size(); i++)
                {
                        graph[i].outputs = graph[i].input_layer( graph[i-1].outputs );
                }
        };
        void back_prop(Signals answer)
        {
        int last=graph.size()-1;
                if ( graph[last].outputs.size() != answer.size())
                {
                        cout << "Error: Expected " << answer.size() << " outputs, but got " << graph[last].outputs.size() << endl;
                        exit(1);
                }
                graph[last].descend_grad_last(answer, delta, graph[last-1].outputs);
                for (int i = last-1; i>=0; i--)
                {
cout << "For previous layer: " << i << endl;
                        graph[i].descend_grad();
                }
        };
void results(Signals answer)
{
        int last=graph.size()-1;
        for (int i = 0; i < graph[last].outputs.size(); i++)
                cout << graph[last].outputs[i] << " is Output " << i << " Expected " << answer[i] << endl;
        // reference for verify
        // https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
}
};


float L1_Bias_weight = 0.35;
float L2_Bias_weight = 0.60;

Layers L1 = {
                { // Node 1
                                  // This node weights input from IN1 and IN2 as follows
                    //11  12           ROW/COL == TO NODE(this layer)/FROM NODE (prev layer)
                    0.15, 0.2     // Array length is 3 so this takes 3 inputs
        },
                { // Node 2
                                   // This node weights input from IN1 and IN2 as follows
                   // 21  22           ROW/COL == TO NODE(this layer)/FROM NODE (prev layer)
                    0.25, 0.3
        }
};

Layers L2 = {
                {  // Node 1
                                  // This node weights inputs from L1N1 and L2N2 as follows
                   // 11  12   13        ROW/COL == TO NODE(this layer)/FROM NODE (prev layer)
                    0.4, 0.45
                },
                {  // Node 2
                                  // This node weights inputs from L1N1 and L2N2 as follows
                  // 21  22    23        ROW/COL == TO NODE(this layer)/FROM NODE (prev layer)
                    0.5, 0.55
                 }
};
RowVector Biases={L1_Bias_weight, L2_Bias_weight};
neuron_network perceptron({ L1, L2 }, Biases);


int main()
{
        Signals result;
        Signals input_train = { 0.05, 0.1 };
        Signals answer = { 0.01, 0.99};
        // perceptron.show_net();
 cout << "*********** Feed Forward" << endl;
        perceptron.forward_feed(input_train);
 cout << "*********** Results" << endl;
        perceptron.results(answer);
 cout << "*********** Back Propagation" << endl;
        perceptron.back_prop(answer);
}
