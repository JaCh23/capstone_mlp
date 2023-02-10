#include<iostream>
#include<vector>
#include<cmath>
#include <algorithm>
#include "read_file.cpp"

// Sigmoid activation function
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

std::vector<double> compute_layer(const std::vector<double> &inputs, const std::vector<double> &weights, const std::vector<double> &biases)
{
    std::vector<double> outputs(biases.size());
    for(int i = 0; i < outputs.size(); ++i)
    {
        double dot_product = biases[i];
        for(int j = 0; j < inputs.size(); ++j)
        {
            dot_product += inputs[j] * weights[i * inputs.size() + j];
        }
        outputs[i] = sigmoid(dot_product);
    }
    return outputs;
}

// std::vector<double> mlp(std::vector<double>&inputs)
int mlp(std::vector<double>inputs)
{
    // std::vector<double> inputs = read_file_arr("input.txt");

    // Assign the weights and biases from the header file
    std::vector<double> weights_1 = read_file_arr("array_0.txt");
    std::vector<double> biases_1 = read_file_arr("array_1.txt");
    std::vector<double> weights_2 = read_file_arr("array_2.txt");
    std::vector<double> biases_2 = read_file_arr("array_3.txt");
    std::vector<double> weights_3 = read_file_arr("array_4.txt");
    std::vector<double> biases_3 = read_file_arr("array_5.txt");

    std::vector<double> hidden_layer_1 = compute_layer(inputs, weights_1, biases_1);
    std::vector<double> hidden_layer_2 = compute_layer(hidden_layer_1, weights_2, biases_2);
    std::vector<double> outputs = compute_layer(hidden_layer_2, weights_3, biases_3);

    for(int i = 0; i < outputs.size(); ++i)
    {
        std::cout << outputs[i] << " ";
    }

    std::cout << std::endl;

    auto largest = std::max_element(outputs.begin(), outputs.end());

    int largest_classified_bucket = std::distance(outputs.begin(), largest);

    return largest_classified_bucket;

    // return outputs;
    
    // std::cout << "Output layer values: ";
    // for(int i = 0; i < outputs.size(); ++i)
    // {
    //     std::cout << outputs[i] << " ";
    // }
    // std::cout << std::endl;
    // return 0;
}

// int main() {
//     std::vector<double> inputs = read_file_arr("input.txt");
//     int output = mlp(inputs);
//     std::cout << output << std::endl;
// }



