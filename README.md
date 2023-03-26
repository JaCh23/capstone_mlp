# capstone_mlp

## File Dependencies

The following files are required for this project:

- ultra.py (MAIN) 
- all_arrays.json (Scaler, PCA, MLP arrays)
- test_actions.json (Test data of one Grenade, Shield, Logout)
- pca_mlp_1.bit (Hardware)
- pca_mlp_1.hwh (Hardware)

## Instructions

How to run code

1. Log in to Ultra
2. sudo -s
3. export XILINX_XRT=/usr
4. python3 ultra.py

Output should look something like this with the MLP being triggered around once every 2-5 seconds
![image](https://user-images.githubusercontent.com/24263853/226246431-7412900e-b46c-4c25-8354-6f245a17c75a.png)

## Instructions for Ext Comms

Where to refactor to demo with live data - 3 TODOs as shown here.

![image](https://user-images.githubusercontent.com/24263853/226247141-7d0a540b-42cb-4883-b364-a20d0526195c.png)


![image](https://user-images.githubusercontent.com/24263853/226247211-d399dc3a-7994-4f03-9a98-d79016b26a9b.png)
