#include <iostream>
using namespace std;

double dotProduct(double arr1[], double arr2[], int n) {
    double dot = 0;
    for(int i=0; i<n; i++) {
        dot += arr1[i] * arr2[i];
    }
    return dot;
}

int main() {
    double arr1[] = {0.017314464190467227, 0.7821381516657591, 1.0, 0.7851502881374393, 0.8214685624410034, 0.777309756018898, 1.0, 0.7438152161080576, 0.8758875965614796, 0.0, 0.8176854178020512, 0.6521847636448815, 0.9915502727104807, 0.14720052571656392, 0.2867605526436922, 0.0, 0.0, 1.0, 0.0, 0.8176854178020512, 0.6521847636448815, 1.0, 0.0, 0.0};
    double arr2[] = {0.102506444, 0.19679374, 0.09388371, -0.27791741, 0.014275927, 0.09055922, -0.19166091, -0.12475899, -0.11866661, 0.0035446535, -0.22332294, 0.026845362, 0.19756798, -0.19461705, 0.048656672, -0.1615423, -0.013381894, -0.08020575, -0.11188429, 0.2842856, 0.13158199, -0.0945655, 0.26772913, -0.3217155};
    int n = sizeof(arr1)/sizeof(arr1[0]);

    double dot = dotProduct(arr1, arr2, n);

    cout << "Dot Product: " << dot << endl;

    return 0;
}




