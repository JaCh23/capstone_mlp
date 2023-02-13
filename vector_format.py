import numpy as np

def loadVector(filename):
    data = np.loadtxt(f"{filename}.txt")
    vector = data.flatten()
    print(vector)
    with open("vector_list.cpp", "a") as f:
            f.write(f"std::vector<double> {filename} = \n" + "{")

            # iterate through each element in vector except last element and write it into file
            for i in range(len(vector) - 1):
                f.write(str(vector[i]) + ", ")
            
            # write last element into file
            f.write(str(vector[len(vector) - 1]))
            f.write("};\n")

            
filenames = ['array_0', 'array_1', 'array_2', 'array_3', 'array_4', 'array_5', 'sim_data', 'sim_labels']

for filename in filenames:
    loadVector(filename)
