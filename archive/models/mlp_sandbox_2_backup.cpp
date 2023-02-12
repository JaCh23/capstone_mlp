#include <iostream>
#include <cmath>
#include<vector>



// Sigmoid activation function
double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

void print_vector(const std::vector<double>& v) {
    for (auto i : v) {
        std::cout << i << " ";
    }
    std::cout << std::endl;
}

//iterate through an array
void iterateArray(double *array, int size) {
  for (int i = 0; i < size; i++) {
    std::cout << array[i] << ' ';
  }
  std::cout << '\n';
}


// Derivative of sigmoid function
// double sigmoidDerivative(double x) {
//     return x * (1 - x);
// }

int main() {
    // Input data
    double inputs[2] = {0.36747999, 0.00683330};
    // Weights for the first hidden layer
    double w1[2][2] = {{0.1, 0.2}, {0.3, 0.4}};
    // Bias for the first hidden layer
    double b1[2] = {0.5, 0.6};
    // Output from the first hidden layer
    double z1[2];
    
    // Weights for the second hidden layer
    double w2[2][2] = {{0.7, 0.8}, {0.9, 1.0}};
    // Bias for the second hidden layer
    double b2[2] = {1.1, 1.2};
    // Output from the second hidden layer
    double z2[2];

    // Weights for the output layer
    double w3[2][2] = {{1.3, 1.4}, {1.5, 1.6}};
    // Bias for the output layer
    double b3[2] = {1.7, 1.8};
    // Output from the output layer
    double output[2];

    // Calculate the output of the first hidden layer
    for (int i = 0; i < 2; i++) {
        double sum = 0;
        for (int j = 0; j < 2; j++) {
            sum += inputs[j] * w1[j][i];
        }
        z1[i] = sigmoid(sum + b1[i]);
    }

    // Calculate the output of the second hidden layer
    for (int i = 0; i < 2; i++) {
        double sum = 0;
        for (int j = 0; j < 2; j++) {
            sum += z1[j] * w2[j][i];
        }
        z2[i] = sigmoid(sum + b2[i]);
    }


    // Calculate the final output
    for (int i = 0; i < 2; i++) {
        double sum = 0;
        for (int j = 0; j < 2; j++) {
            sum += z2[j] * w3[j][i];
        }
        output[i] = sigmoid(sum + b3[i]);
    }

    iterateArray(z1,2);
    iterateArray(z2,2);
    iterateArray(output,2);

    std::cout << "Output layer values: ";
    for(int i = 0; i < 2; ++i)
    {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;


    // Classify the output into 2 actions
    // if (output < 0.5) {
    //     cout << "Action 1" << endl;
    // } else {
    //     cout << "Action 2" << endl;
    // }

    return 0;
}