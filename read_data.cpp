#include "mlp.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <iostream>
#include <cmath>

using namespace std;

// vector<vector<double>> readData(const string& filename) {
//     vector<vector<double>> data;
//     ifstream file(filename);

//     if (file.is_open()) {
//         string line;
//         while (getline(file, line)) {
//             vector<double> row;
//             stringstream ss(line);
//             double value;
//             while (ss >> value) {
//                 row.push_back(value);
//             }
//             data.push_back(row);
//         }
//         file.close();
//     } else {
//         cout << "Unable to open file " << filename << endl;
//     }

//     return data;
// }