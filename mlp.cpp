#include "mlp.h"


#include<iostream>
#include<vector>
#include<cmath>
#include<algorithm>
#include <fstream>
#include <string>


std::vector<double> mlp(std::vector<double>inputs)
{
    auto sigmoid = [](const std::vector<double> &x) -> std::vector<double> 
    {
        std::vector<double> result(x.size());
        for (int i = 0; i < x.size(); i++) 
        {
            result[i] = 1.0 / (1.0 + exp(-x[i]));
        }
        return result;
    };

    auto softmax = [](const std::vector<double> &x) -> std::vector<double>
    {
        std::vector<double> result(x.size());
        double sum = 0.0;
        for (int i = 0; i < x.size(); i++)
        {
            result[i] = exp(x[i]);
            sum += result[i];
        }
        for (int i = 0; i < x.size(); i++)
        {
            result[i] /= sum;
        }
        return result;
    };

    auto oneHotEncode = [](int index, int size) {
        std::vector<double> one_hot(size, 0);
        one_hot[index] = 1.0;
        return one_hot;
    };

    auto compute_layer = [](const std::vector<double> &inputs, const std::vector<std::vector<double>> &weights, const std::vector<double> &biases)
    {
        std::vector<double> outputs(biases.size());

        for(int i = 0; i < outputs.size(); i++)
        {
            double dot_product = 0.0;
            for(int j = 0; j < inputs.size(); j++)
            {
                dot_product += inputs[j] * weights[j][i];
            }
            outputs[i] = dot_product;
        }
        return outputs;
    };


    std::vector<std::vector<double>> weights_1 = read_file("array_0.txt");
    std::vector<double> biases_1 = read_file_arr("array_1.txt");
    std::vector<std::vector<double>> weights_2 = read_file("array_2.txt");
    std::vector<double> biases_2 = read_file_arr("array_3.txt");
    std::vector<std::vector<double>> weights_3 = read_file("array_4.txt");
    std::vector<double> biases_3 = read_file_arr("array_5.txt");

    std::vector<double> hidden_layer_1 = sigmoid(compute_layer(inputs, weights_1, biases_1));
    std::vector<double> hidden_layer_2 = sigmoid(compute_layer(hidden_layer_1, weights_2, biases_2));
    std::vector<double> outputs = softmax(compute_layer(hidden_layer_2, weights_3, biases_3));

    auto largest = std::max_element(outputs.begin(), outputs.end());
    int largest_classified_bucket = std::distance(outputs.begin(), largest);

    return oneHotEncode(largest_classified_bucket, outputs.size());
}



// Test function to check mlp function independently
// int main() {
//     // std::vector<double> inputs = {0.36, 0.06};
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
// std::vector<double> sigmoid(const std::vector<double> &x) {
//     std::vector<double> result(x.size());
//     for (int i = 0; i < x.size(); i++) {
//         result[i] = 1.0 / (1.0 + exp(-x[i]));
//     }
//     return result;
// }

// Softmax activation function
// std::vector<double> softmax(const std::vector<double> &x) {
//     std::vector<double> result(x.size());
//     double sum = 0.0;
//     for (int i = 0; i < x.size(); i++) {
//         result[i] = exp(x[i]);
//         sum += result[i];
//     }
//     for (int i = 0; i < x.size(); i++) {
//         result[i] /= sum;
//     }
//     return result;
// }

// std::vector<double> compute_layer(const std::vector<double> &inputs, const std::vector<std::vector<double>> &weights, const std::vector<double> &biases)
// {
//     std::vector<double> outputs(biases.size());

//     for(int i = 0; i < outputs.size(); i++)
//     {
//         double dot_product = 0.0;
//         for(int j = 0; j < inputs.size(); j++)
//         {
//             dot_product += inputs[j] * weights[j][i];
//         }
//         outputs[i] = dot_product;
//     }
//     return outputs;
// }

    // One hot encoding function
    // std::vector<double> oneHotEncode(int index, int size) {
    //     std::vector<double> one_hot(size, 0.0);
    //     one_hot[index] = 1.0;
    //     return one_hot;
    // }





