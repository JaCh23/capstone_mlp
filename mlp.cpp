#include<iostream>
#include<vector>
#include<cmath>
#include <algorithm>
#include "read_file.cpp"

// std::pair<int, int> get_dimensions(const std::vector<std::vector<int>>& array) {
//     return {array.size(), array[0].size()};
// }

void print_vector(const std::vector<double>& v) {
    for (auto i : v) {
        std::cout << i << " ";
    }
    std::cout << std::endl;
}

// One hot encoding function
std::vector<double> oneHotEncode(int index, int size) {
    std::vector<double> one_hot(size, 0.0);
    one_hot[index] = 1.0;
    return one_hot;
}

// Sigmoid activation function
std::vector<double> sigmoid(const std::vector<double> &x) {
    std::vector<double> result(x.size());
    for (int i = 0; i < x.size(); i++) {
        result[i] = 1.0 / (1.0 + exp(-x[i]));
    }
    return result;
}

// 0.51349672045, 0.52398158497

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

std::vector<double> compute_layer(const std::vector<double> &inputs, const std::vector<std::vector<double>> &weights, const std::vector<double> &biases)
{
    std::vector<double> outputs(biases.size());

    for(int i = 0; i < outputs.size(); i++)
    {
        double dot_product = 0.0;
        for(int j = 0; j < inputs.size(); j++)
        {
            //multiplying input by weight
            // std::cout << "multiplying input by weight: " << inputs[j] << " * " << weights[j][i] << std::endl;
            dot_product += inputs[j] * weights[j][i];
        }
        outputs[i] = dot_product;
        // std::cout << outputs[i] << std::endl;
    }
    return outputs;
}

// std::vector<double> mlp(std::vector<double>&inputs)
std::vector<double> mlp(std::vector<double>inputs)
{
    // std::vector<double> inputs = read_file_arr("input.txt");

    // Assign the weights and biases from the header file
    // std::vector<double> weights_1 = {0.1, 0.2, 0.3, 0.4};
    // std::vector<double> biases_1 = {0.5, 0.6};
    // std::vector<double> weights_2 = {0.7, 0.8, 0.9, 1.0};
    // std::vector<double> biases_2 = {1.1, 1.2};
    // std::vector<double> weights_3 = {1.3, 1.4, 1.5, 1.6};
    // std::vector<double> biases_3 = {1.7, 1.8};

    std::vector<std::vector<double>> weights_1 = read_file("array_0.txt");
    std::vector<double> biases_1 = read_file_arr("array_1.txt");
    std::vector<std::vector<double>> weights_2 = read_file("array_2.txt");
    std::vector<double> biases_2 = read_file_arr("array_3.txt");
    std::vector<std::vector<double>> weights_3 = read_file("array_4.txt");
    std::vector<double> biases_3 = read_file_arr("array_5.txt");

    
   

    // std::cout << "biases_1 size;" << biases_1.size() << std::endl;
    // std::cout << "weights_2 size;" << weights_2.size() << std::endl;
    // std::cout << "biases_2 size;" << biases_2.size() << std::endl;
    // std::cout << "weights_3 size;" << weights_3.size() << std::endl;
    // std::cout << "biases_3 size;" << biases_3.size() << std::endl;


    // std::vector<double> hidden_layer_1 = compute_layer(inputs, weights_1, biases_1);
    std::vector<double> hidden_layer_1 = sigmoid(compute_layer(inputs, weights_1, biases_1));

    std::vector<double> hidden_layer_2 = sigmoid(compute_layer(hidden_layer_1, weights_2, biases_2));
    std::vector<double> outputs = softmax(compute_layer(hidden_layer_2, weights_3, biases_3));

    // std::cout << "hidden layer 1:" ;
    // print_vector(hidden_layer_1);

    // std::cout << "sigmoid hidden layer 1:" ;
    // print_vector(sigmoid(hidden_layer_1));
    // print_vector(hidden_layer_2);
    // print_vector(outputs);

    // for(int i = 0; i < outputs.size(); i++)
    // {
    //     std::cout << outputs[i] << " ";
    // }

    // std::cout << std::endl;

    auto largest = std::max_element(outputs.begin(), outputs.end());

    int largest_classified_bucket = std::distance(outputs.begin(), largest);

    return oneHotEncode(largest_classified_bucket, outputs.size());

    // return outputs; test
    
    // std::cout << "Output layer values: ";
    // for(int i = 0; i < outputs.size(); ++i)
    // {
    //     std::cout << outputs[i] << " ";
    // }
    // std::cout << std::endl;
    // return 0;
}

// int main() {
//     std::vector<double> inputs = {0.36, 0.06};
//     // std::vector<double> inputs = read_file_arr("test_input.txt");

//     std::vector<double> output = mlp(inputs);

//     std::cout << "Output layer values: ";
//     for(int i = 0; i < output.size(); ++i)
//     {
//         std::cout << output[i] << " ";
//     }
//     std::cout << std::endl;
//     return 0;
// }



