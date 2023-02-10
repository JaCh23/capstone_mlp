// File: read_file.cpp
#include <fstream>
#include <vector>
#include <iostream>

// std::vector<std::vector<double>> read_file(const std::string &filename) {
//     std::ifstream input_file(filename);
//     std::vector<std::vector<double>> data;
//     std::vector<double> row;

//     double value;
//     while (input_file >> value) {
//         row.push_back(value);
//         if (input_file.peek() == '\n') {
//             data.push_back(row);
//             row.clear();
//         }
//     }

//     return data;
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
