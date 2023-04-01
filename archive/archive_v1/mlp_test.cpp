#include "mlp.h"


#include <fstream>
#include <sstream>
#include <vector>
#include <iostream>
#include <cmath>
#include <string>

#include<iostream>
#include<vector>
#include<cmath>
#include<algorithm>
#include <fstream>
#include <string>


using namespace std;

void writeData(const vector<vector<double>>& data, const string& filename) {
    ofstream file(filename);
    if (file.is_open()) {
        for (int i = 0; i < data.size(); i++) {
            for (int j = 0; j < data[i].size(); j++) {
                file << data[i][j] << " ";
            }
            file << endl;
        }
        file.close();
    } else {
        cout << "Unable to open file " << filename << endl;
    }
}

int main() {

    vector<double> inputData;
    vector<double> expectedData;

    // vector<double> inputData = read_file_arr("sim_data.txt");
    // vector<double> expectedData = read_file_arr("sim_labels.txt");

    vector<vector<double>> outputData2d;
    int COLUMN_FEATURES = 24;
    int LABEL_SIZE = 4;

    vector<vector<double>> inputData2d = create_2d_vector_from_1d(inputData, 24);
    vector<vector<double>> expectedData2d = create_2d_vector_from_1d(expectedData, 4);

    for (int i = 0; i < inputData2d.size(); i++) {
        outputData2d.push_back(mlp(inputData2d[i]));
    }

    // print output data size
    cout << "Output data size: " << outputData2d.size() << endl;

    // print expectedData size
    cout << "Expected data size: " << expectedData2d.size() << endl;
    
    // writing data to file for sanity check
    writeData(outputData2d, "output.txt");

    int count = 0;

    for (int i = 0; i < expectedData2d.size(); i++) {
        bool flag = compareDataRow(outputData2d[i], expectedData2d[i]);
        if (!flag) {
            cout << "Data row " << i << " does not match the expected data." << endl;
        }
        else {
            cout << "Data row " << i << " matches the expected data." << endl;
            count++;
        }
    }

    cout << "Number of test cases passed: " << count << endl;
    cout << "Accuracy of model: " << (count*100)/outputData2d.size() << "%" << endl;

    return 0;
}
