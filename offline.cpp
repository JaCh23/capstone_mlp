#include "mlp.h"

#include<iostream>
#include<vector>
#include<cmath>
#include<algorithm>
#include <fstream>
#include <string>

#include "offline_functions.cpp"

int main()
{

    auto print_vector = [](const std::vector<double>& v) -> void {
        for (auto i : v) {
        std::cout << i << " ";
        }
        std::cout << std::endl;
        };


    std::vector<std::vector<double> > weights_1 = read_file("array_0.txt");



    std::vector<double> biases_1 = read_file_arr("array_1.txt");

    save_2d_vector("weights_1.txt", weights_1);

    // std::vector<std::vector<double> > weights_2 = read_file("array_2.txt");
    // std::vector<double> biases_2 = read_file_arr("array_3.txt");
    // std::vector<std::vector<double> > weights_3 = read_file("array_4.txt");
    // std::vector<double> biases_3 = read_file_arr("array_5.txt");

    // std::vector<double> hidden_layer_1 = sigmoid(compute_layer(inputs, weights_1, biases_1));
    // std::vector<double> hidden_layer_2 = sigmoid(compute_layer(hidden_layer_1, weights_2, biases_2));
    // std::vector<double> outputs = softmax(compute_layer(hidden_layer_2, weights_3, biases_3));

    // auto largest = std::max_element(outputs.begin(), outputs.end());
    // int largest_classified_bucket = std::distance(outputs.begin(), largest);

    // return oneHotEncode(largest_classified_bucket, outputs.size());

    return 0;
}
