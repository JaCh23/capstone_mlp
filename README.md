# Capstone AI MLP 

## File Dependencies

The following files are required for this project:

ultra.py (MAIN) 
- features.npz (Scaler, PCA, MLP arrays)
- test_actions.json (Test data of one Grenade, Shield, Logout, Reload)
- pca_mlp_2.bit (Hardware bitstream)
- pca_mlp_2.hwh (Hardware bitstream)

## Summary
- Applying Gaussian blur to 3d parabolic data traced by acceleration path
- Feature engineering: 
    - Total distance change over 3 axes
    - Top 2 axes dimension change
    - Finding ratios across all 3 movement axes
    - Gap ratio for XZ axes to seive out Grenade actions
- Feeding into PCA to obtain top X eigenvectors aka components
- Combining together into MLP to output classification

## Revision History 
- V3: Movement Trajectory in 3D Space + PCA
    - V3.5 Gap ratio feature engineering [NEW]
    - V3.4 Distance feature
    - V3.3 Half movement trajectories instead of complete movement 
    - V3.2 Include Logout 
    - V3.1 Excluded Logout
- V2: Splintering into 8 sub-datasets per time quarter and conducting standard feature engineering techniques, eg mean variance median IQR
- V1: Standard feature engineering techniques across all 6 data metrics Gyro X,Y,Z and Acc X,Y,Z

## Instructions

How to run code
1. Log in to Ultra
2. sudo -s
3. export XILINX_XRT=/usr
4. python3 ultra.py

## Instructions for Ext Comms

![image](https://user-images.githubusercontent.com/24263853/227785317-bc19a23a-e920-4279-828f-0fbff27af7ff.png)

![image](https://user-images.githubusercontent.com/24263853/227785277-90e0dd91-4821-4d44-a474-874cc2f8732b.png)
