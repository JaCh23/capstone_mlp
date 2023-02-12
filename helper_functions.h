// helper_functions.h
#ifndef HELPER_FUNCTIONS_H
#define HELPER_FUNCTIONS_H

#include <vector>
#include <string>

bool compareDataRow(const std::vector<double>& dataRow, const std::vector<double>& expectedDataRow);
std::vector<std::vector<double>> arrayToVector(const double inputArray, int ROW_SIZE, int COL_SIZE);

#endif