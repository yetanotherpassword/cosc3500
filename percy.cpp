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
        neuron_layer(Layers nl)
        {
                int siz = nl.size();
                cout << "Creating neuron layer of " << siz - 1 << " nodes" << endl;
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
        void set_output_list(Layers s)
        {
             for (int i = 0; i < s.size(); i++)
                 for (int j = 0; j < s[i].size(); j++)
                    if (s[i][j]==1)
                    {
                       nodes[j].output_links.push_back(i);
                       cout << "Node " << j << " links to next layer Node "<<i<<endl;
                    }
        };
        Layers get_input_list()
        {
              Layers b;
              for (int i = 0; i < nodes.size(); i++)
              {
                 Signals a;
                 for (int j = 0; j < nodes[i].input_weights.size(); j++)
                 {
                    if (nodes[i].input_weights[j] == 0)
                        a.push_back(0);
                    else
                        a.push_back(1);
                 }
                 b.push_back(a);
              }      
              return b;
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
                           Layers s = graph[i].get_input_list();
                           graph[i-1].set_output_list(s);
                                          
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
// Layer 1 has 3 nodes, getting input from up to 4 nodes
Layers L1 = { /*node 1 input weights from Inputs 1,2 */
        {
                0.15, 0.2, L1_Bias
        },        // L1N1 has inputs from I1, I2 and I4
        /*node 2 input weights from Inputs 1,2 */
        {
                0.25, 0.3, L1_Bias
        }        // L1N2 has inputs from I1, I2 and I3
};
// Layer 2 has 4 nodes, getting input from up to 3 nodes (ie the output from L1 nodes)
Layers L2 = {
                {
                0.4, 0.45, L2_Bias
        },        //L2N1 has input fro L1N1, L1N2
        {
                0.5, 0.55, L2_Bias
        }        //L2N2 has input fro L1N1, L1N2
};

neuron_network perceptron({ L1, L2 });

int main()
{
        Signals result;
        Signals input_train = { 0.05, 0.1 };
        Signals train_answer = { 0.01, 0.99 };
        //  perceptron.show_net();
        result = perceptron.forward_feed(input_train);
        perceptron.back_prop(result, train_answer);
        cout << "result.size=" << result.size() << endl;
        for (int i = 0; i < result.size(); i++)
                cout << result[i] << " is Output " << i << endl;
        // reference for verify
        // https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
}
