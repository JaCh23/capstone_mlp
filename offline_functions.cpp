#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <string>

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

// bool compareDataRow(const std::vector<double>& dataRow, const std::vector<double>& expectedDataRow) {
//     auto print_vector = [](const std::vector<double>& v) -> void {
//         for (auto i : v) {
//         std::cout << i << " ";
//         }
//         std::cout << std::endl;
//         };

//     bool flag = true;
//     if (dataRow.size() != expectedDataRow.size()) {
//         return false;
//     }

//     for (int i = 0; i < dataRow.size(); i++) {
//         if (dataRow[i] != expectedDataRow[i]) {
//             flag = false;
//         }
//     }

//     cout << endl;

//     if (flag) {
//         cout << "Test case passed." << endl;
//     }
//     else {
//         cout << "Test case failed." << endl;
//     }
//     cout << "Expected vector: " ;
//     print_vector(expectedDataRow);

//     cout << "Actual vector: " ;
//     print_vector(dataRow);
        
//     return flag;
// }

std::vector<double> read_file_arr(const std::string &filename) {
    std::ifstream input_file(filename);
    std::vector<double> data;

    double value;
    while (input_file >> value) {
        data.push_back(value);
    }

    return data;
}

std::vector<std::vector<double>> read_file(const std::string &filename) {
    std::ifstream input_file(filename);
    std::vector<std::vector<double>> data;
    std::vector<double> row;

    double value;
    while (input_file >> value) {
        row.push_back(value);
        if (input_file.peek() == '\n') {
            data.push_back(row);
            row.clear();
        }
    }

    return data;
}

void save_2d_vector(const std::string& filename, const std::vector<std::vector<double> >& data) {
    std::ofstream file(filename);
    for (int i = 0; i < data.size(); ++i) {
        for (int j = 0; j < data[i].size(); ++j) {
            file << data[i][j] << " ";
        }
        file << std::endl;
    }
    file.close();
}