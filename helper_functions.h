// helper_functions.h
#ifndef HELPER_FUNCTIONS_H
#define HELPER_FUNCTIONS_H

#include <vector>
#include <string>

bool compareDataRow(const std::vector<double>& dataRow, const std::vector<double>& expectedDataRow);
std::vector<std::vector<double> > arrayToVector(const double inputArray, int ROW_SIZE, int COL_SIZE);
std::vector<double> read_file_arr(const std::string &filename);
void print_vector(std::vector<double> v);
std::vector<std::vector<double> > create_2d_vector_from_1d(const std::vector<double>& input, int num_cols);

#endif