
#include <math.h>
#include <iostream>

#define NUM_INPUTS 2
#define NUM_HIDDEN 2
#define NUM_OUTPUTS 2

int mlp(double inputs[NUM_INPUTS]) {

	double hidden_layer1[NUM_HIDDEN];
	double hidden_layer2[NUM_HIDDEN];
	double outputs[NUM_OUTPUTS];
	double softmax_outputs[NUM_OUTPUTS];

    // Initialize weights and biases for the hidden layers and output layer
	double weights_hidden_layer1[NUM_INPUTS][NUM_HIDDEN] = {{0, 1}, {2, 3}};
	double biases_hidden_layer1[NUM_HIDDEN] = {1, 2};
	double weights_hidden_layer2[NUM_HIDDEN][NUM_HIDDEN] = {{1, 2}, {3, 4}};
	double biases_hidden_layer2[NUM_HIDDEN] = {1, 2};
	double weights_output_layer[NUM_HIDDEN][NUM_OUTPUTS] = {{1, 2}, {5, 6}};
	double biases_output_layer[NUM_OUTPUTS] = {3, 4};

	// Compute values for hidden layer 1
	for(int i = 0; i < NUM_HIDDEN; i++){
		hidden_layer1[i] = 0;
		for(int j = 0; j < NUM_INPUTS; j++){
			hidden_layer1[i] += inputs[j] * weights_hidden_layer1[j][i];
		}
		hidden_layer1[i] += biases_hidden_layer1[i];
		hidden_layer1[i] = hidden_layer1[i] > 0 ? hidden_layer1[i] : 0;
	}

	// Compute values for hidden layer 2
	for(int i = 0; i < NUM_HIDDEN; i++){
		hidden_layer2[i] = 0;
		for(int j = 0; j < NUM_HIDDEN; j++){
			hidden_layer2[i] += hidden_layer1[j] * weights_hidden_layer2[j][i];
		}
		hidden_layer2[i] += biases_hidden_layer2[i];
		hidden_layer2[i] = hidden_layer2[i] > 0 ? hidden_layer2[i] : 0;
	}

	// softmax implementation
	double offset = outputs[0];
	double sum = 0.0;


	// Compute values for output layer
	for(int i = 0; i < NUM_OUTPUTS; i++){
		outputs[i] = 0;
		for(int j = 0; j < NUM_HIDDEN; j++){
			outputs[i] += hidden_layer2[j] * weights_output_layer[j][i];
		}
		outputs[i] += biases_output_layer[i];
	}

	// Compute the exponential of each input
	for (int i = 0; i < NUM_OUTPUTS; i++) {
		softmax_outputs[i] = exp(outputs[i]-offset);
		sum += softmax_outputs[i];
	}

	// Normalize the outputs by the sum of exponentials
	for (int i = 0; i < NUM_OUTPUTS; i++) {
		softmax_outputs[i] /= sum;
	}

	double max = softmax_outputs[0];
	int ans = 0;	 

	for (int i = 0; i < NUM_OUTPUTS; i++) {
        if (softmax_outputs[i] > max) {
            ans = i;
        }
    }

    return ans;
}

int main() {
    double inputs[NUM_INPUTS] = {0.3, 4.5};
    // print mlp(inputs)
    std::cout << mlp(inputs) << std::endl;

    return 0;
}