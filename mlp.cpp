#include "mlp.h"

#include<iostream>
#include<vector>
#include<cmath>
#include<algorithm>
#include <fstream>
#include <string>


std::vector<double> mlp(std::vector<double>inputs)
{

    // std::vector<double> weights_1 = {1,2,3,4,5,6};
    // std::vector<double> biases_1 = {5,6};
    // std::vector<double> weights_2 = {0.1,0.2,0.3,0.4};
    // std::vector<double> biases_2 = read_file_arr("array_3.txt");
    // std::vector<double> weights_3 = read_file_arr("array_4.txt");
    // std::vector<double> biases_3 = read_file_arr("array_5.txt");

    std::vector<double> weights_1 = read_file_arr("array_0.txt");
    std::vector<double> biases_1 = read_file_arr("array_1.txt");
    std::vector<double> weights_2 = read_file_arr("array_2.txt");
    std::vector<double> biases_2 = read_file_arr("array_3.txt");
    std::vector<double> weights_3 = read_file_arr("array_4.txt");
    std::vector<double> biases_3 = read_file_arr("array_5.txt");

    // std::vector<double> hidden_layer_1 = compute_layer(inputs, weights_1, biases_1, 2);

    // std::vector output = hidden_layer_1;
    // std::cout << "Output layer values: ";
    // for(int i = 0; i < output.size(); ++i)
    // {
    //     std::cout << output[i] << " ";
    // }
    // std::cout << std::endl;


    std::vector<double> hidden_layer_1 = sigmoid(compute_layer(inputs, weights_1, biases_1, 32));
    std::vector<double> hidden_layer_2 = sigmoid(compute_layer(hidden_layer_1, weights_2, biases_2, 32));
    std::vector<double> outputs = softmax(compute_layer(hidden_layer_2, weights_3, biases_3, 4));

    // print_vector(inputs);
    // std::cout << "Hidden Layer 1 values: ";
    // print_vector(hidden_layer_1);
    // std::cout << "Hidden Layer 2 values: ";
    // print_vector(hidden_layer_2);
    // std::cout << "Output Layer values: ";
    // print_vector(outputs);

    auto largest = std::max_element(outputs.begin(), outputs.end());
    int largest_classified_bucket = std::distance(outputs.begin(), largest);

    return oneHotEncode(largest_classified_bucket, outputs.size());
}



// Test function to check mlp function independently
// int main() {
//     // std::vector<double> inputs = {7, 8, 9};
//     std::vector<double> inputs = read_file_arr("test_input.txt");

//     std::vector<double> output = mlp(inputs);

//     std::cout << "Output layer values: ";
//     for(int i = 0; i < output.size(); ++i)
//     {
//         std::cout << output[i] << " ";
//     }
//     std::cout << std::endl;
//     return 0;
// }

// Sigmoid activation function
std::vector<double> sigmoid(const std::vector<double> &x) {
    std::vector<double> result(x.size());
    for (int i = 0; i < x.size(); i++) {
        result[i] = 1.0 / (1.0 + exp(-x[i]));
    }
    return result;
}

// Softmax activation function
std::vector<double> softmax(const std::vector<double> &x) {
    std::vector<double> result(x.size());
    double sum = 0.0;
    for (int i = 0; i < x.size(); i++) {
        result[i] = exp(x[i]);
        sum += result[i];
    }
    for (int i = 0; i < x.size(); i++) {
        result[i] /= sum;
    }
    return result;
}

std::vector<double> compute_layer(const std::vector<double> &inputs, const std::vector<double> &weights, 
const std::vector<double> &biases, int weights_col)
{
    std::vector<double> outputs(biases.size());

    // sanity check inputs
    // std::cout << "Inputs: " << std::endl;
    // for(int i = 0; i < inputs.size(); i++) {
    //     std::cout << inputs[i] << std::endl;
    // }

    // sanity check weights
    // std::cout << "Weights: " << std::endl;
    // for(int i = 0; i < weights.size(); i++) {
    //     std::cout << weights[i] << std::endl;
    // }

    // printing positions of weights and their respective row and columns
    // std::cout << "Weights: " << std::endl;
    // for(int i = 0; i < weights.size(); i++)
    // {
    //     std::cout << "Position: " << i << " Row: " << i / inputs.size() << " Column: " << i % inputs.size() << std::endl;
    // }


    for(int i = 0; i < outputs.size(); i++)
    {
        double dot_product = 0.0;
        // k=0;
        for(int j = 0; j < inputs.size(); j++)
        {
            // printing multiplication of dot product
            // std::cout << inputs[j] << " * " << weights[j * weights_col + i] << std::endl;
            int idx = j * weights_col + i;
            // std::cout << "idx: " << idx << std::endl;

            dot_product += inputs[j] * weights[idx];
            // k++;
            // std::cout << "Dot product: " << dot_product << std::endl;
        }
        // printing dot product
        // std::cout << "Dot product: " << dot_product << std::endl;
        dot_product += biases[i];
        outputs[i] = dot_product;
    }
    return outputs;
}

// One hot encoding function
std::vector<double> oneHotEncode(int index, int size) {
    std::vector<double> one_hot(size, 0.0);
    one_hot[index] = 1.0;
    return one_hot;
}

