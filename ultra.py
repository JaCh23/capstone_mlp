
import threading
import traceback
import random

import time
import numpy as np
import pandas as pd
from scipy import stats
import json
from scipy.ndimage import gaussian_filter

# import pynq
# from pynq import Overlay


class AIModel(threading.Thread):
    def __init__(self):
        super().__init__()

        # Flags
        self.shutdown = threading.Event()

        features = np.load('features_v3.3.2.npz', allow_pickle=True)
        self.mean = features['mean']
        self.variance = features['variance']
        self.pca_eigvecs = features['pca_eigvecs']
        self.weights = features['weights_list']

        # Reshape scaling_factors, mean and variance to (1, 3)
        self.mean = self.mean.reshape(40, 3)
        self.variance = self.variance.reshape(40, 3)

        # read in the test actions from the JSON file
        with open('test_actions_v3.3.2.json', 'r') as f:
            test_actions = json.load(f)

        # extract the test data for each action from the dictionary
        self.test_g = np.array(test_actions['G'])
        self.test_s = np.array(test_actions['S'])
        self.test_r = np.array(test_actions['R'])
        self.test_l = np.array(test_actions['L'])


        # define the available actions
        self.test_actions = ['G', 'S', 'R', 'L']
        
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
        acc_df = acc_df.apply(pd.to_numeric)
        fs = 20 # sampling frequency
        dt = 1/fs

        filtered_acc_df = acc_df.apply(lambda x: gaussian_filter(x, sigma=5))
        
        ax = filtered_acc_df[0]
        ay = filtered_acc_df[1]
        az = filtered_acc_df[2]

        vx = np.cumsum(ax) * dt
        vy = np.cumsum(ay) * dt
        vz = np.cumsum(az) * dt

        x = np.cumsum(vx) * dt
        y = np.cumsum(vy) * dt
        z = np.cumsum(vz) * dt

        x_arr = np.array(x)
        y_arr = np.array(y)
        z_arr = np.array(z)

        x_disp = x_arr[-1] - x_arr[0]
        y_disp = y_arr[-1] - y_arr[0]
        z_disp = z_arr[-1] - z_arr[0]

        xyz = np.column_stack((x, y, z))

        return xyz, [x_disp, y_disp, z_disp]
    
    def get_top_2_axes(self, row):
        row = np.array(row)
        abs_values = np.abs(row)
        top_2_idx = abs_values.argsort()[-2:][::-1]
        return (top_2_idx[0], top_2_idx[1])
    
    def get_metric_ratios(self, row):
        row = np.array(row)
        # Compute ratios of x, y, z metrics
        return np.array([
            row[0]/row[1],
            row[0]/row[2],
            row[1]/row[2]
        ])
    
    # Define Scaler
    def scaler(self, X):
        return (X - self.mean) / np.sqrt(self.variance)

    # Define PCA
    def pca(self, X):
        return np.dot(X, self.pca_eigvecs.T)


    def rng_test_action(self):
        # choose a random action from the list
        chosen_action = random.choice(self.test_actions)
        
        # # print chosen action
        print(f'Chosen action: {chosen_action} \n')
        
        # use the chosen action to select the corresponding test data
        if chosen_action == 'G':
            test_data = self.test_g
        elif chosen_action == 'S':
            test_data = self.test_s
        elif chosen_action == 'L':
            test_data = self.test_l
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
        action_dict = {0: 'G', 1: 'L', 2: 'R', 3: 'S'}
        action = action_dict[max_index]
        return action


    # def mlp_vivado(self, data):
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

    def mlp_vivado_mockup(self, data):
        action = data[0:120].reshape(40, 3)
        scaled_action = self.scaler(action)
        pca_action = self.pca(scaled_action.reshape(1,120))
        mlp_input = np.hstack((pca_action.reshape(1,6), data[120:128].reshape(1,8)))
        Y_softmax = self.mlp(mlp_input)
        return Y_softmax

    def AIDriver(self, test_input):
        test_input = test_input.reshape(40, 6)
        acc_df = test_input[:, -3:]
        
        # Transform data using Scaler and PCA
        blurred_data, disp_change = self.blur_3d_movement(acc_df.reshape(40,3))
        top_2 = self.get_top_2_axes(disp_change)
        metric_ratios = self.get_metric_ratios(disp_change)

        vivado_input = np.hstack((np.array(blurred_data).reshape(1,120), 
                          np.array(disp_change).reshape(1,3), 
                          np.array(top_2).reshape(1,2),
                          np.array(metric_ratios).reshape(1,3)
                          )).flatten()

        # vivado_predictions = self.mlp_vivado(vivado_input)
        vivado_predictions = self.mlp_vivado_mockup(vivado_input)

        action = self.get_action(vivado_predictions)

        print(vivado_predictions)
        print(action)

        return action
        
    def close_connection(self):
        self.shutdown.set()

        print("Shutting Down Connection")

    def run(self):
        # Set the threshold value for movement detection based on user input
        K = 5
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
            # if ai_queue: # TODO re-enable for live integration
            if 1 == 1: # TODO DIS-enable for live integration
                # runs loop 6 times and packs the data into groups of 6

                    # q_data = ai_queue.get() # TODO re-enable for live integration
                    # ai_queue.task_done() # TODO re-enable for live integration
                    # new_data = np.array(q_data) # TODO re-enable for live integration
                    # new_data = new_data / 100.0 # TODO re-enable for live integration
                    
                    new_data = np.random.randn(6) # TODO DIS-enable for live integration
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

                                rng_test_action = self.rng_test_action() # TODO DIS-enable for live integration
                                action = self.AIDriver(rng_test_action) # TODO DIS-enable for live integration

                                # action = self.AIDriver(data_packet) # TODO re-enable for live integration
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
