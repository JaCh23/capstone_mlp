
#include <iostream>
#include <vector>
#include <cmath>

#include "read_file.cpp"
// #include "weights_biases.h"

// Sigmoid activation function
// double sigmoid(double x) {
//     return 1.0 / (1.0 + exp(-x));
// }

// std::vector<double> forward_propagation(
//         const std::vector<double>& input,
//         const std::vector<std::vector<double>>& weights_1,
//         const std::vector<double>& biases_1,
//         const std::vector<std::vector<double>>& weights_2,
//         const std::vector<double>& biases_2,
//         const std::vector<std::vector<double>>& weights_3,
//         const std::vector<double>& biases_3
// ) {

//     // Hidden layer 1
//     std::vector<double> hidden_layer_1(biases_1.size());
//     for (int i = 0; i < hidden_layer_1.size(); i++) {
//         hidden_layer_1[i] = biases_1[i];
    
//         // std::cout << hidden_layer_1.size() << std::endl;


//         for (int j = 0; j < input.size(); j++) {
//             hidden_layer_1[i] += input[j] * weights_1[i][j];
//         }
//         hidden_layer_1[i] = sigmoid(hidden_layer_1[i]);
//         std::cout << hidden_layer_1[i] << std::endl;
        
//     }
//     std::cout << "----------------" << std::endl;
//     std::cout << hidden_layer_1.size() << std::endl;


//     // Hidden layer 2
//     std::vector<double> hidden_layer_2(biases_2.size());
//     for (int i = 0; i < hidden_layer_2.size(); i++) {
//         hidden_layer_2[i] = biases_2[i];

//         for (int j = 0; j < hidden_layer_1.size(); j++) {
//             hidden_layer_2[i] += hidden_layer_1[j] * weights_2[i][j];
//         }
//         hidden_layer_2[i] = sigmoid(hidden_layer_2[i]);
//     }

// // Output layer 
//     std::vector<double> output(biases_3.size());
//     for (std::size_t i = 0; i < output.size(); ++i) {
//         output[i] = biases_3[i];


//         for (std::size_t j = 0; j < hidden_layer_2.size(); ++j) {
//             output[i] += weights_3[i][j] * hidden_layer_2[j];
//         }
//         output[i] = sigmoid(output[i]);
//     }

    
//     return output;
// }



// int main() {
//     // Example input
//     std::vector<double> input = read_file_arr("input.txt");

//     // Assign the weights and biases from the header file
//     std::vector<std::vector<double>> weights_1 = read_file("array_0.txt");
//     std::vector<double> biases_1 = read_file_arr("array_1.txt");
//     std::vector<std::vector<double>> weights_2 = read_file("array_2.txt");
//     std::vector<double> biases_2 = read_file_arr("array_3.txt");
//     std::vector<std::vector<double>> weights_3 = read_file("array_4.txt");
//     std::vector<double> biases_3 = read_file_arr("array_5.txt");


//     // for (std::size_t i = 0; i < input.size(); ++i) {
//     //     std::cout << input[i] << std::endl;
//     // }
    
//     // std::vector<std::vector<double>> weights_1 = weights_1_values;
//     // std::vector<double> biases_1 = biases_1_values;
//     // std::vector<std::vector<double>> weights_2 = weights_2_values;
//     // std::vector<double> biases_2 = biases_2_values;
//     // std::vector<std::vector<double>> weights_3 = weights_3_values;
//     // std::vector<double> biases_3 = biases_3_values;
    
//     // Perform forward propagation
//     std::vector<double> output = forward_propagation(
//         input,
//         weights_1,
//         biases_1,
//         weights_2,
//         biases_2,
//         weights_3,
//         biases_3
//     );
    
//     // Print the output
//     std::cout << "Output: [";
//     for (std::size_t i = 0; i < output.size(); ++i) {
//         std::cout << output[i];
//         if (i < output.size() - 1) {
//             std::cout << ", ";
//         }
//     }
//     std::cout << "]" << std::endl;
    
//     return 0;

// }