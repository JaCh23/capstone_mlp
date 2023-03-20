# capstone_mlp

## File Dependencies

The following files are required for this project:

- ultra.py (MAIN) 
- pca_mlp_1.bit (Hardware)
- pca_mlp_1.hwh (Hardware)
- arrays.txt (Scaler and PCA arrays)
- mlp_model.joblib (Used for sanity check but can remove)

## Instructions

How to run code

1. Log in to Ultra
2. sudo -s
3. export XILINX_XRT=/usr
4. python3 ultra.py

Output should look something like this with the MLP being triggered around once every 2-5 seconds
![image](https://user-images.githubusercontent.com/24263853/226246431-7412900e-b46c-4c25-8354-6f245a17c75a.png)

