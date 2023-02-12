#include "mlp.h"

#include <fstream>
#include <iostream>

std::vector<double> read_file_arr(const std::string &filename) {
    std::ifstream input_file(filename);
    std::vector<double> data;

    double value;
    while (input_file >> value) {
        data.push_back(value);
    }

    return data;
}
