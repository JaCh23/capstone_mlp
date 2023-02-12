#include <iostream>
#include <vector>
#include <cmath>

#include "read_file.cpp"


using namespace std;

// Sigmoid activation function
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Softmax activation function
vector<double> softmax(vector<double> x) {
    double sum = 0.0;
    for (double i : x) {
        sum += exp(i);
    }
    for (double &i : x) {
        i = exp(i) / sum;
    }
    return x;
}

int main() {

// Input data
vector<double> x = read_file_arr("input.txt");

// Define the weights and biases for the hidden layer
vector<vector<double>> w1 = read_file("array_0.txt");
vector<double> b1 = read_file_arr("array_1.txt");

// Define the weights and biases for the output layer
vector<vector<double>> w2 =  read_file("array_2.txt");
vector<double> b2 = read_file_arr("array_3.txt");

// Assign the weights and biases from the header file
    // std::vector<double> inputs = read_file_arr("input.txt");
    // std::vector<double> weights_1 = read_file_arr("array_0.txt");
    // std::vector<double> biases_1 = read_file_arr("array_1.txt");
    // std::vector<double> weights_2 = read_file_arr("array_2.txt");
    // std::vector<double> biases_2 = read_file_arr("array_3.txt");
    // std::vector<double> weights_3 = read_file_arr("array_4.txt");
    // std::vector<double> biases_3 = read_file_arr("array_5.txt");

// Calculate the dot product for the hidden layer
vector<double> z1(4);
for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 10; j++) {
        z1[i] += x[j] * w1[j][i];
    }
    z1[i] += b1[i];
}

// Apply activation function to the dot product
vector<double> a1(4);
for (int i = 0; i < 4; i++) {
    a1[i] = sigmoid(z1[i]);
}

// Calculate the dot product for the output layer
vector<double> z2(4);
for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
        z2[i] += a1[j] * w2[j][i];
    }
    z2[i] += b2[i];
}

// Apply activation function to the dot product
vector<double> y(4);
for (int i = 0; i < 4; i++) {
    y[i] = sigmoid(z2[i]);
}

// Find the largest class
int largest_class = 0;
double max_prob = 0.0;

std::cout << "Output layer values: ";
for(int i = 0; i < y.size(); ++i)
{
    std::cout << y[i] << " ";
}
std::cout << std::endl;

for (int i = 0; i < 4; i++) {
    if (y[i] > max_prob) {
        max_prob = y[i];
        largest_class = i;
    }
}

cout << "The largest class is: " << largest_class << endl;

return 0;
}