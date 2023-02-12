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

// bool compareData(const string& expectedFilename, const vector<vector<double>>& data) {
//     ifstream expectedFile(expectedFilename);

//     if (expectedFile.is_open()) {
//         int rowNum = 0;
//         string line;
//         while (getline(expectedFile, line)) {
//             stringstream ss(line);
//             double expectedValue;
//             int colNum = 0;
//             while (ss >> expectedValue) {
//                 double value = data[rowNum][colNum];
//                 if (fabs(value - expectedValue) > 0.001) {
//                     cout << "Test case failed at row " << rowNum << " column " << colNum << endl;
//                     cout << "Expected value: " << expectedValue << ", but got: " << value << endl;
//                     // return false;
//                 }
//                 else {
//                     cout << "Test case passed at row " << rowNum << " column " << colNum << endl;
//                     cout << "Expected value: " << expectedValue << ", got: " << value << endl;
//                 }
//                 colNum++;
//             }
//             rowNum++;
//         }
//         expectedFile.close();
//     } else {
//         cout << "Unable to open file " << expectedFilename << endl;
//         return false;
//     }

//     return true;
// }

bool compareDataRow(const vector<double>& dataRow, const vector<double>& expectedDataRow) {
    bool flag = true;
    if (dataRow.size() != expectedDataRow.size()) {
        return false;
    }

    for (int i = 0; i < dataRow.size(); i++) {
        if (dataRow[i] != expectedDataRow[i]) {
            flag = false;
        }
    }

    if (flag) {
        cout << "Test case passed." << endl;
        cout << "Expected vector: " ;
        print_vector(expectedDataRow);

        cout << "Actual vector: " ;
        print_vector(dataRow);

    }
    else {
        cout << "Test case failed." << endl;
        cout << "Expected vector: " ;
        print_vector(expectedDataRow);

        cout << "Actual vector: " ;
        print_vector(dataRow);

    }

    return flag;
}


int main() {
    vector<vector<double>> inputData = readData("sim_data.txt");
    vector<vector<double>> expectedData = readData("sim_labels.txt");
    vector<vector<double>> outputData;
    for (const vector<double>& row : inputData) {
        // cout << "input row: " ;
        // print_vector(row);
        outputData.push_back(mlp(row));
    }
    
    // writing data to file for sanity check
    writeData(outputData, "output.txt");

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
    //accuracy of model
    cout << "Accuracy of model: " << (count*100)/outputData.size() << "%" << endl;

    return 0;
}