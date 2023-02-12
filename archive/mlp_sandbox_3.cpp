#include<iostream>
#include<vector>
#include<cmath>
#include <algorithm>
#include "read_file.cpp"



void print_vector(const std::vector<double>& v) {
    for (auto i : v) {
        std::cout << i << " ";
    }
    std::cout << std::endl;
}


void print2DVector(const std::vector<std::vector<double>> &vec) {
  for (int i = 0; i < vec.size(); ++i) {
    for (int j = 0; j < vec[i].size(); ++j) {
      std::cout << "vec[" << i << "][" << j << "] = " << vec[i][j] << std::endl;
    }
  }
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


std::vector<double> compute_layer(const std::vector<double> &inputs, const std::vector<std::vector<double>> &weights, const std::vector<double> &biases)
{
    std::vector<double> outputs(biases.size());

    // std::cout << "inputs: ";
    // print_vector(inputs);

    for(int i = 0; i < outputs.size(); i++)
    {
        double dot_product = 0.0;
        for(int j = 0; j < inputs.size(); j++)
        {
            //multiplying input by weight
            std::cout << "multiplying input by weight: " << inputs[j] << " * " << weights[j][i] << std::endl;
            dot_product += inputs[j] * weights[j][i];
        }
        dot_product += biases[i];
        outputs[i] = dot_product;
        std::cout << outputs[i] << std::endl;
    }

    // std::cout << "outputs: ";
    // print_vector(outputs);
    return outputs;
}

// std::vector<double> mlp(std::vector<double>&inputs)
std::vector<double> mlp(std::vector<double>inputs)
{
    // std::vector<double> inputs = read_file_arr("input.txt");

    // Assign the weights and biases from the header file
    std::vector<std::vector<double>> weights_1 = read_file("weights_1.txt");
    print2DVector(weights_1);


    std::vector<double> biases_1 = {5,6};
    std::vector<double> weights_2 = {0.7, 0.8, 0.9, 1.0};
    std::vector<double> biases_2 = {1.1, 1.2};
    std::vector<double> weights_3 = {1.3, 1.4, 1.5, 1.6};
    std::vector<double> biases_3 = {1.7, 1.8};

    // std::vector<double> weights_1 = read_file_arr("array_0.txt");
    // std::vector<double> biases_1 = read_file_arr("array_1.txt");
    // std::vector<double> weights_2 = read_file_arr("array_2.txt");
    // std::vector<double> biases_2 = read_file_arr("array_3.txt");
    // std::vector<double> weights_3 = read_file_arr("array_4.txt");
    // std::vector<double> biases_3 = read_file_arr("array_5.txt");

    
   

    // std::cout << "biases_1 size;" << biases_1.size() << std::endl;
    // std::cout << "weights_2 size;" << weights_2.size() << std::endl;
    // std::cout << "biases_2 size;" << biases_2.size() << std::endl;
    // std::cout << "weights_3 size;" << weights_3.size() << std::endl;
    // std::cout << "biases_3 size;" << biases_3.size() << std::endl;


    std::vector<double> hidden_layer_1 = compute_layer(inputs, weights_1, biases_1);
    // std::vector<double> hidden_layer_1 = sigmoid(compute_layer(inputs, weights_1, biases_1));

    std::cout << "hidden layer 1: " ;
    print_vector(hidden_layer_1);

    std::cout << "sigmoid hidden layer 1:" ;
    print_vector(sigmoid(hidden_layer_1));
    // print_vector(hidden_layer_2);
    // print_vector(outputs);
    return {{}};

    // return outputs; test
    
    // std::cout << "Output layer values: ";
    // for(int i = 0; i < outputs.size(); ++i)
    // {
    //     std::cout << outputs[i] << " ";
    // }
    // std::cout << std::endl;
    // return 0;
}

int main() {
    std::vector<double> inputs = {7,8};
    // std::vector<double> inputs = read_file_arr("test_input.txt");

    std::vector<double> output = mlp(inputs);

    
    return 0;
}
