// helper_functions.h
#ifndef HELPER_FUNCTIONS_H
#define HELPER_FUNCTIONS_H

#include <vector>
#include <string>

std::vector<std::vector<double>> readData(const std::string& filename);
bool compareDataRow(const std::vector<double>& dataRow, const std::vector<double>& expectedDataRow);
std::vector<double> read_file_arr(const std::string &filename);
std::vector<std::vector<double>> read_file(const std::string &filename);


#endif