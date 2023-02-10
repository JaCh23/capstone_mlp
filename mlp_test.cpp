#include <fstream>
#include <sstream>
#include <vector>
#include <iostream>
#include <cmath>

#include "mlp.cpp"

using namespace std;

vector<vector<double>> readData(const string& filename) {
    vector<vector<double>> data;
    ifstream file(filename);

    if (file.is_open()) {
        string line;
        while (getline(file, line)) {
            vector<double> row;
            stringstream ss(line);
            double value;
            while (ss >> value) {
                row.push_back(value);
            }
            data.push_back(row);
        }
        file.close();
    } else {
        cout << "Unable to open file " << filename << endl;
    }

    return data;
}


bool compareData(const string& expectedFilename, const vector<int>& data) {
    ifstream expectedFile(expectedFilename);

    if (expectedFile.is_open()) {
        int rowNum = 0;
        int expectedValue;
        while (expectedFile >> expectedValue) {
            int value = data[rowNum];
            if (value != expectedValue) {
                cout << "Test case failed at row " << rowNum << endl;
                cout << "Expected value: " << expectedValue << ", but got: " << value << endl;
                // return false;
            }
            rowNum++;
        }
        expectedFile.close();
    } else {
        cout << "Unable to open file " << expectedFilename << endl;
        // return false;
    }

    return true;
}

int main() {
    vector<vector<double>> inputData = readData("sim_data.txt");
    vector<int> outputData;
    for (const vector<double>& row : inputData) {
        // cout << row.size() << endl;
        outputData.push_back(mlp(row));
    }
    if (compareData("sim_labels.txt", outputData)) {
        cout << "All test cases passed." << endl;
    } else {
        cout << "Test case(s) failed." << endl;
    }
    return 0;
}