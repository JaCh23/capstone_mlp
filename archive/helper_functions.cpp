#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <string>

#include "helper_functions.h"

using namespace std;

// function to iterate and print a vector
void print_vector(std::vector<double> v)
{
    for(int i = 0; i < v.size(); ++i)
    {
        std::cout << v[i] << " ";
    }
    std::cout << std::endl;
}

std::vector<std::vector<double> > create_2d_vector_from_1d(const std::vector<double>& input, int num_cols)
{
    std::vector<std::vector<double> > result;
    std::vector<double> row;
    for (int i = 0; i < input.size(); i++)
    {
        row.push_back(input[i]);
        if ((i + 1) % num_cols == 0)
        {
            result.push_back(row);
            row.clear();
        }
    }
    if (!row.empty())
    {
        result.push_back(row);
    }
    return result;
}

bool compareDataRow(const std::vector<double>& dataRow, const std::vector<double>& expectedDataRow) {

    bool flag = true;
    if (dataRow.size() != expectedDataRow.size()) {
        return false;
    }

    for (int i = 0; i < dataRow.size(); i++) {
        if (dataRow[i] != expectedDataRow[i]) {
            flag = false;
        }
    }

    cout << endl;

    if (flag) {
        cout << "Test case passed." << endl;
    }
    else {
        cout << "Test case failed." << endl;
    }
    cout << "Expected vector: " ;
    print_vector(expectedDataRow);

    cout << "Actual vector: " ;
    print_vector(dataRow);
        
    return flag;
}

std::vector<double> arrayToVector(double arr[], double size) {
    std::vector<double> vec;
    for (int i = 0; i < size; i++) {
        vec.push_back(arr[i]);
    }
    return vec;
}

// std::vector<double> read_file_arr(const std::string &filename) {
//     std::ifstream input_file(filename);
//     std::vector<double> data;

//     double value;
//     while (input_file >> value) {
//         data.push_back(value);
//     }

//     return data;
// }