# Capstone AI MLP V1.6

Summary of project:
- Performing laser tag action recognition to detect in-game action
- Actions are namely Grenade, Logout, Reload, Shield (G,L,R,S)
- Input: IMU Acceelration and Gyroscope X,Y,Z sensor data
- Output: Classification of action 
- This is then passed to Game Engine for further processing


## File Dependencies

The following files are required for this project:

ultra.py (MAIN) 
- features_v1.6.npz (Scaler, PCA, MLP arrays)
- pca_mlp_1_5.bit (Hardware bitstream)
- pca_mlp_1_6.hwh (Hardware bitstream)

## Summary

(Final V1 model)
1. Feature extraction on 30x6 input data to yield 100 features
2. Standard Scaler on extracted features 
3. PCA Analysis on scalarised features
4. MLP on PCA Top K components 

Key features:
- Mean sliding window on sensor data to reduce noise
- For the experimented dataset and project, PCA with 90% explained variance implied Top 24 components aka K=24 (see image below)
- MLP with 2 hidden layers and ReLU activation
    - Input = 24 
    - Hidden Layer 1, N=16
    - Hidden Layer 2, N=8
    - Softmax = 4, outputs aka G,L,R,S classification

![image](https://user-images.githubusercontent.com/24263853/231349024-3f7ed20e-38ff-49fc-9516-5ae851826618.png)

Image: PCA Explained Variance Ratio vs. Number of Components

(Decommissioned V3 Model)
- Applying Gaussian blur to 3D parabolic data traced by acceleration path
- Feature engineering: 
    - Total distance change over 3 axes
    - Top 2 axes dimension change
    - Finding ratios across all 3 movement axes
    - Gap ratio for XZ axes to seive out Grenade actions
- Feeding into PCA to obtain top X eigenvectors aka components
- Combining together into MLP to output classification

## Revision History 
- V3: Movement Trajectory in 3D Space + PCA (Decommissioned due to underperforming results)
    - V3.5 Gap ratio feature engineering
    - V3.4 Distance feature
    - V3.3 Half movement trajectories instead of complete movement 
    - V3.2 Include Logout 
    - V3.1 Excluded Logout
- V2: Splintering into 8 sub-datasets per time quarter and conducting standard feature engineering techniques, eg mean variance median IQR
- V1: Standard feature engineering techniques across all 6 data metrics Gyro and Acc X,Y,Z
    - V1.5, V1.6: Expanded training dataset to collecting with non-teammates actions [FINAL]
    - V1.5.4: New Logout action
    - V1.5.3: MVP of Features-Scaler-PCA-MLP workflow 

## Instructions

How to run code
1. Log in to Ultra
2. sudo -s
3. export XILINX_XRT=/usr
4. python3 ultra.py

## Instructions for Ext Comms

![image](https://user-images.githubusercontent.com/24263853/227785317-bc19a23a-e920-4279-828f-0fbff27af7ff.png)

![image](https://user-images.githubusercontent.com/24263853/227785277-90e0dd91-4821-4d44-a474-874cc2f8732b.png)
