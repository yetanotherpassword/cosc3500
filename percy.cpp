#include <iostream>
#include <vector>
#include <cmath>
using namespace std;
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
        neuron(Weights w)
        {
                int siz = w.size();

                for (int i = 0; i < siz - 1; i++)
                {
                        input_weights.push_back(w[i]);
                }
                bias_val = w[siz - 1];
        };

        float input_neuron(Signals s)
        {
                inputs.clear();
                netin = bias_val;
                for (int i = 0; i < s.size(); i++)
                {
                        inputs.push_back(s[i]);
                        netin += input_weights[i] *s[i];
                }
                output = thres_func(netin);
                return output;
        };

        void calc_partial_last(float tgt)
        {
                float deeEdeeO, deeOdeeN, deeNdeeW, deeEdeeW;
                dNdW.clear();
                dEdW.clear();
                dEdO.clear();
                dOdN.clear();
                new_weights.clear();
                deeEdeeO = output - tgt;
                deeOdeeN = output *(1 - output);
                for (int i = 0; i < input_weights.size(); i++)
                {
                   if (input_weights[i] != 0) //weight of zero means no link from node i, in layer to left
                   {
                        deeNdeeW = inputs[i];
                        deeEdeeW = deeEdeeO *deeOdeeN * deeNdeeW;
                        new_weights.push_back(input_weights[i] - eta *deeEdeeW);
                   }
                   else
                   {
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
                         cout << "Error in setup: nextLayer.nodes["<< j << "].input_weights.size() " << nextLayer.nodes[j].input_weights.size()<< " != prevLayer->nodes.size()=" << this->nodes.size() << endl;
                         exit (-1);
                       }
                       this->mark_node_output(m-1,i,m,j,nextLayer.nodes[j].input_weights[i]);
                     }
                }
                     for (int i=0;i<this->nodes.size();i++)
                        //cout <<"Node "<< i << " outputs to " << this->nodes[i].out_label << endl;
                        cout << this->nodes[i].out_label << endl;
        }
        neuron_layer(Layers nl)
        {
                int siz = nl.size();
                cout << "Creating neuron layer of " << siz << " nodes" << endl;
                for (int i = 0; i < siz; i++)
                {
                        neuron tmp(nl[i]);
                        nodes.push_back(tmp);
                }
        }
        Signals input_layer(Signals in)
        {
                Signals out;
                for (int i = 0; i < nodes.size(); i++)
                {
                        out.push_back(nodes[i].input_neuron(in));
                }
                return out;
        };
        Signals descend_grad_last(Signals ans)
        {
                Signals out;
                for (int i = 0; i < nodes.size(); i++)
                {
                       nodes[i].calc_partial_last(ans[i]);
                }
                return out;
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
        neuron_network(Graph g)
        {
                cout << "Creating neuron network with " << g.size() << " layers" << endl;
                for (int i = 0; i < g.size(); i++)
                {
                        cout << "Neuron layer " << i << " : ";
                        neuron_layer tmp(g[i]);
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
        Signals forward_feed(Signals inputs)
        {
                if (inputs.size() != num_inputs)
                {
                        cout << "Error: Expected " << num_inputs << " inputs, but got " << inputs.size() << endl;
                        exit(1);
                }
                Signals output = graph[0].input_layer(inputs);
                for (int i = 1; i < graph.size(); i++)
                {
                        output = graph[i].input_layer(output);
                }
                return output;
        };
        Signals back_prop(Signals output, Signals answer)
        {
                if (output.size() != answer.size())
                {
                        cout << "Error: Expected " << answer.size() << " outputs, but got " << output.size() << endl;
                        exit(1);
                }
                graph[graph.size()-1].descend_grad_last(answer);
                for (int i = graph.size()-2; i>=1; i--)
                {
                        graph[i].descend_grad();
                }
                return output;
        };
};

float L1_Bias = 0.35;
float L2_Bias = 0.60;
float L3_Bias = 0.60;
Layers L1 = {
                {
// 3 nodes at L1 imply 3 inputs (plus one Bias) ALWAYS 1 to 1 (input to node)
                0.4, 0.45, L1_Bias // for N1 in L1 its inputs are from I1, and I2 (and bias)
        },        //L2N1 has input fro L1N1, L1N2
                {
                0.4, 0.45, L1_Bias // for N1 in L1 its inputs are from I1, and I2 (and bias)
        },        //L2N1 has input fro L1N1, L1N2
        {
                0.5, 0.55, L1_Bias // for N2 in L1 is inputs are for I1 and I2 (and bias)
        }        //L2N2 has input fro L1N1, L1N2
};
// Layer 1 has 6 nodes, getting input from up to 3 nodes
Layers L2 = { /*node 1 input weights from Inputs 1,2,3 */
        {
                0.9, 0.15, 0, L2_Bias  // Layer 2 Node 1 gets input from 
        },        // L1N1 has inputs from I1, I2 and I4
        /*node 2 input weights from Inputs 1,2 */
        {
                0.5, 0, 0.25, L2_Bias
        },        // L1N2 has inputs from I1, I2 and I3
        {
                0.5, 0.25, 0.3, L2_Bias
        } ,       // L1N2 has inputs from I1, I2 and I3
        {
                0.25, 0.4, 0, L2_Bias
        },        // L1N2 has inputs from I1, I2 and I3
        {
                0.25, 0.5, 0.3, L2_Bias
        } ,       // L1N2 has inputs from I1, I2 and I3
        {
                0.25, 0.25, 0.3, L2_Bias
        }        // L1N2 has inputs from I1, I2 and I3
};
// Layer 3 has 2 nodes, getting input from up to 6 nodes (ie the output from L2 nodes) plus the Bias
// 2 nodes at Last Layer imply 2 outputs ALWAYS 1 to 1 (output from node)
Layers L3 = {
                {
                0.4, 0.45, 0, 0, 0.4, 0, L3_Bias
        },        //L2N1 has input fro L1N1, L1N2
        {
                0.5, 0.55, 0.3, 0.5, 0.2, 0.2, L3_Bias
        }        //L2N2 has input fro L1N1, L1N2
};

neuron_network perceptron({ L1, L2, L3 });

int main()
{
        Signals result;
        Signals input_train = { 0.05, 0.1 , 0.5};
        Signals train_answer = { 0.01, 0.99};
        //  perceptron.show_net();
        result = perceptron.forward_feed(input_train);
        perceptron.back_prop(result, train_answer);

        for (int i = 0; i < result.size(); i++)
                cout << result[i] << " is Output " << i << endl;
        // reference for verify
        // https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
}
