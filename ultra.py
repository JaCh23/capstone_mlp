
import threading
import traceback
import random

import time
import numpy as np
import pandas as pd
from scipy import stats
import joblib
import json
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter1d

# import pynq
# from pynq import Overlay


class AIModel(threading.Thread):
    def __init__(self):
        super().__init__()

        # Flags
        self.shutdown = threading.Event()

        # Load all_arrays.json
        with open('all_arrays.json', 'r') as f:
            all_arrays = json.load(f)

        # Retrieve values from all_arrays
        self.scaling_factors = np.array(all_arrays['scaling_factors'])
        self.mean = np.array(all_arrays['mean'])
        self.variance = np.array(all_arrays['variance'])
        self.pca_eigvecs = np.array(all_arrays['pca_eigvecs'])
        self.weights = [np.array(w) for w in all_arrays['weights']]

        # Reshape scaling_factors, mean and variance to (1, 3)
        self.scaling_factors = self.scaling_factors.reshape(40, 3)
        self.mean = self.mean.reshape(40, 3)
        self.variance = self.variance.reshape(40, 3)

        # read in the test actions from the JSON file
        with open('test_actions.json', 'r') as f:
            test_actions = json.load(f)

        # extract the test data for each action from the dictionary
        self.test_g = np.array(test_actions['G'])
        self.test_s = np.array(test_actions['S'])
        self.test_r = np.array(test_actions['R'])

        # define the available actions
        self.test_actions = ['G', 'S', 'R']
        
        # PYNQ overlay
        # self.overlay = Overlay("pca_mlp_1.bit")
        # self.dma = self.overlay.axi_dma_0

        # # Allocate input and output buffers once
        # self.in_buffer = pynq.allocate(shape=(35,), dtype=np.float32)
        # self.out_buffer = pynq.allocate(shape=(4,), dtype=np.float32)

    def sleep(self, seconds):
        start_time = time.time()
        while time.time() - start_time < seconds:
            pass
    
    def blur_3d_movement(self, acc_df):
        acc_df = pd.DataFrame(acc_df)
        fs = 20 # sampling frequency
        dt = 1/fs

        # Apply median filtering column-wise
        filtered_acc = acc_df.apply(lambda x: medfilt(x, kernel_size=7))
        filtered_acc = gaussian_filter1d(filtered_acc.values.astype(float), sigma=3, axis=0)
        filtered_acc_df = pd.DataFrame(filtered_acc, columns=acc_df.columns)
        
        ax = filtered_acc_df[0]
        ay = filtered_acc_df[1]
        az = filtered_acc_df[2]

        vx = np.cumsum(ax) * dt
        vy = np.cumsum(ay) * dt
        vz = np.cumsum(az) * dt

        x = np.cumsum(vx) * dt
        y = np.cumsum(vy) * dt
        z = np.cumsum(vz) * dt

        xyz = np.column_stack((x, y, z))

        return xyz
    
    # Define Scaler
    def scaler(self, X):
        return (X - self.mean) / np.sqrt(self.variance)

    # Define PCA
    def pca(self, X):
        return np.dot(X, self.pca_eigvecs.T)


    def rng_test_action(self):
        # choose a random action from the list
        chosen_action = random.choice(self.test_actions)

        # use the chosen action to select the corresponding test data
        if chosen_action == 'G':
            test_data = self.test_g
        elif chosen_action == 'S':
            test_data = self.test_s
        else:
            test_data = self.test_r

        return test_data


    # Define MLP
    def mlp(self, X):
        H1 = np.dot(X, self.weights[0]) + self.weights[1]
        H1_relu = np.maximum(0, H1)
        H2 = np.dot(H1_relu, self.weights[2]) + self.weights[3]
        H2_relu = np.maximum(0, H2)
        Y = np.dot(H2_relu, self.weights[4]) + self.weights[5]
        Y_softmax = np.exp(Y) / np.sum(np.exp(Y), axis=1, keepdims=True)
        return Y_softmax

    def get_action(self, softmax_array):
        max_index = np.argmax(softmax_array)
        action_dict = {0: 'G', 1: 'R', 2: 'S'}
        action = action_dict[max_index]
        return action


    # def MLP_Overlay(self, data):
    #     start_time = time.time()

    #     # reshape data to match in_buffer shape
    #     data = np.reshape(data, (35,))

    #     self.in_buffer[:] = data

    #     self.dma.sendchannel.transfer(self.in_buffer)
    #     self.dma.recvchannel.transfer(self.out_buffer)

    #     # wait for transfer to finish
    #     self.dma.sendchannel.wait()
    #     self.dma.recvchannel.wait()

    #     # print output buffer
    #     print("mlp done with output: " + " ".join(str(x) for x in self.out_buffer))

    #     print(f"MLP time taken so far output: {time.time() - start_time}")

    #     return self.out_buffer

    def AIDriver(self, test_input):
        test_input = test_input.reshape(40, 6)
        acc_df = test_input[:, -3:]
        
        # Transform data using Scaler and PCA
        blurred_data = self.blur_3d_movement(acc_df.reshape(40,3))
        data_scaled = self.scaler(blurred_data)
        data_pca = self.pca(data_scaled.reshape(1,120))

        # Make predictions using MLP
        predictions = self.mlp(data_pca)
        action = self.get_action(predictions)

        print(predictions)
        print(action)

        return action
        
    def close_connection(self):
        self.shutdown.set()

        print("Shutting Down Connection")

    def run(self):
        # Set the threshold value for movement detection based on user input
        K = 10
        # K = float(input("threshold value? "))

        # Initialize arrays to hold the current and previous data packets
        current_packet = np.zeros((5,6))
        previous_packet = np.zeros((5,6))
        data_packet = np.zeros((40,6))
        is_movement_counter = 0
        movement_watchdog = False
        loop_count = 0

        # live integration loop
        while True:
            if ai_queue: # TODO re-enable for live integration
            # if 1 == 1: # TODO DIS-enable for live integration
                # runs loop 6 times and packs the data into groups of 6

                    q_data = ai_queue.get() # TODO re-enable for live integration
                    ai_queue.task_done() # TODO re-enable for live integration
                    new_data = np.array(q_data) # TODO re-enable for live integration
                    new_data[-3:] = [x/100.0 for x in new_data[-3:]] # TODO re-enable for live integration
                    
                    # new_data = np.random.randn(6) # TODO DIS-enable for live integration
                    # print(" ".join([f"{x:.3f}" for x in new_data]))
                
                    # Pack the data into groups of 6
                    current_packet[loop_count] = new_data
                
                    # Update loop_count
                    loop_count = (loop_count + 1) % 5

                    if loop_count % 5 == 0:
                        curr_mag = np.sum(np.square(np.mean(current_packet[:, -3:], axis=1)))
                        prev_mag = np.sum(np.square(np.mean(previous_packet[:, -3:], axis=1)))

                        # Check for movement detection
                        if not movement_watchdog and curr_mag - prev_mag > K:
                            print("Movement detected!")
                            # print currr and prev mag for sanity check
                            print(f"curr_mag: {curr_mag} \n")
                            print(f"prev_mag: {prev_mag} \n")
                            movement_watchdog = True
                            # append previous and current packet to data packet
                            data_packet = np.concatenate((previous_packet, current_packet), axis=0)

                        # movement_watchdog activated, count is_movement_counter from 0 up 6 and append current packet each time
                        if movement_watchdog:
                            if is_movement_counter < 6:
                                data_packet = np.concatenate((data_packet, current_packet), axis=0)
                                is_movement_counter += 1
                            
                            # If we've seen 6 packets since the last movement detection, preprocess and classify the data
                            else:
                                # print dimensions of data packet
                                # print(f"data_packet dimensions: {data_packet.shape} \n")

                                # rng_test_action = self.rng_test_action() # TODO DIS-enable for live integration
                                # action = self.AIDriver(rng_test_action) # TODO DIS-enable for live integration

                                action = self.AIDriver(data_packet) # TODO re-enable for live integration
                                print(f"action from MLP in main: {action} \n")  # print output of MLP

                                # movement_watchdog deactivated, reset is_movement_counter
                                movement_watchdog = False
                                is_movement_counter = 0
                                # reset arrays to zeros
                                current_packet = np.zeros((5,6))
                                previous_packet = np.zeros((5,6))
                                data_packet = np.zeros((40,6))

                        # Update the previous packet
                        previous_packet = current_packet.copy()


if __name__ == '__main__':
    # AI Model
    print("Starting AI Model Thread")
    ai_model = AIModel()
    ai_model.start()
    print('--------------------------------------------\n')
