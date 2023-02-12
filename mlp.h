#ifndef MLP_H
#define MLP_H

#include <vector>
#include <string>

#include "helper_functions.h"


std::vector<double> mlp(std::vector<double>inputs);
std::vector<double> sigmoid(const std::vector<double> &x);
std::vector<double> softmax(const std::vector<double> &x);
std::vector<double> compute_layer(const std::vector<double> &inputs, const std::vector<std::vector<double> > &weights, const std::vector<double> &biases);
std::vector<double> oneHotEncode(int index, int size);
std::vector<std::vector<double> > getWeights1();
std::vector<double> getBiases1();
std::vector<std::vector<double> > getWeights2();
std::vector<double> getBiases2();
std::vector<std::vector<double> > getWeights3();
std::vector<double> getBiases3();

std::vector<std::vector<double> > getTestData();
std::vector<std::vector<double> > getTestLabels();

   

#endif