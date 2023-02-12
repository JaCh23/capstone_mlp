#include "mlp.h"
#include "helper_functions.h"

#include <fstream>
#include <sstream>
#include <vector>
#include <iostream>
#include <cmath>
#include <string>


using namespace std;

int main() {
    vector<vector<double>> inputData = arrayToVector(getTestData(), ROW_SIZE=100, COL_SIZE=4);
    vector<vector<double>> expectedData = arrayToVector(getTestLabels(), ROW_SIZE=100, COL_SIZE=4);
    vector<vector<double>> outputData;
    
    for (const vector<double>& row : inputData) {
        outputData.push_back(mlp(row));
    }
    
    // writing data to file for sanity check
    // writeData(outputData, "output.txt");

    int count = 0;

   if (outputData.size() != expectedData.size()) {
        cout << "The number of data rows and expected data rows do not match." << endl;
        return 0;
    }
    for (int i = 0; i < outputData.size(); i++) {
        bool flag = compareDataRow(outputData[i], expectedData[i]);
        if (!flag) {
            cout << "Data row " << i << " does not match the expected data." << endl;
        }
        else {
            cout << "Data row " << i << " matches the expected data." << endl;
            count++;
        }
    }

    cout << "Number of test cases passed: " << count << endl;
    cout << "Accuracy of model: " << (count*100)/outputData.size() << "%" << endl;

    return 0;
}









// REMOVED! This function is not needed for now
// void writeData(const vector<vector<double>>& data, const string& filename) {
//     ofstream file(filename);
//     if (file.is_open()) {
//         for (int i = 0; i < data.size(); i++) {
//             for (int j = 0; j < data[i].size(); j++) {
//                 file << data[i][j] << " ";
//             }
//             file << endl;
//         }
//         file.close();
//     } else {
//         cout << "Unable to open file " << filename << endl;
//     }
// }