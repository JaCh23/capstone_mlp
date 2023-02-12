#ifndef MLP_H
#define MLP_H

#include <vector>
#include <string>

#include "helper_functions.h"

std::vector<double> mlp(std::vector<double>inputs);
std::vector<double> sigmoid(const std::vector<double> &x);
std::vector<double> softmax(const std::vector<double> &x);
std::vector<double> compute_layer(const std::vector<double> &inputs, const std::vector<double> &weights, const std::vector<double> &biases, int weights_col);
std::vector<double> oneHotEncode(int index, int size);


#endif