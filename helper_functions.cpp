#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <string>

#include "helper_functions.h"

using namespace std;


bool compareDataRow(const std::vector<double>& dataRow, const std::vector<double>& expectedDataRow) {
    auto print_vector = [](const std::vector<double>& v) -> void {
        for (auto i : v) {
        std::cout << i << " ";
        }
        std::cout << std::endl;
        };

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
