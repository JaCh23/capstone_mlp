
import threading
import traceback
import random

import time
import numpy as np
import pandas as pd
import json
from scipy.ndimage import gaussian_filter

# import pynq
# from pynq import Overlay
class AIModel(threading.Thread):
    def __init__(self, player, action_engine_model, queue_added, K):
        super().__init__()

        self.player = player
        self.action_engine = action_engine_model

        # Flags
        self.shutdown = threading.Event()

        features = np.load('dependencies/features_v1.5.6.npz', allow_pickle=True)
        self.pca_eigvecs = features['pca_eigvecs']
        self.weights = features['weights_list']
        self.mean_vec = features['mean_vec']
        self.scale = features['scale']
        self.mean = features['mean']

        self.K = K
        self.TOTAL_PACKET_COUNT = 30

        self.ai_queue = queue_added

        # PYNQ overlay NEW - pca_mlp_v3.5
#         self.overlay = Overlay("dependencies/pca_mlp_3_5.bit")
#         self.overlay.download()
#         self.dma = self.overlay.axi_dma_0
#         self.in_buffer = pynq.allocate(shape=(129,), dtype=np.float32)
#         self.out_buffer = pynq.allocate(shape=(3,), dtype=np.float32)

        # PYNQ overlay OLD backup - pca_mlp_1
        # self.overlay = Overlay("dependencies/pca_mlp_1.bit")
        # self.dma = self.overlay.axi_dma_0
        # self.in_buffer = pynq.allocate(shape=(125,), dtype=np.float32)
        # self.out_buffer = pynq.allocate(shape=(3,), dtype=np.float32)

    def sleep(self, seconds):
        start_time = time.time()
        while time.time() - start_time < seconds:
            pass
        
    def extract_features(self, raw_sensor_data):

        # Apply median filtering column-wise using the rolling function, window=5
        sensor_data = raw_sensor_data.rolling(5, min_periods=1, axis=0).mean()
        sensor_data = sensor_data.to_numpy()

        # Compute statistical features
        mean = np.mean(sensor_data, axis=0)
        std = np.std(sensor_data, axis=0)
        abs_diff = np.abs(np.diff(sensor_data, axis=0)).mean(axis=0)
        minimum = np.min(sensor_data, axis=0)
        maximum = np.max(sensor_data, axis=0)
        max_min_diff = maximum - minimum
        median = np.median(sensor_data, axis=0)
        mad = np.median(np.abs(sensor_data - np.median(sensor_data, axis=0)), axis=0)
        iqr = np.percentile(sensor_data, 75, axis=0) - np.percentile(sensor_data, 25, axis=0)
        negative_count = np.sum(sensor_data < 0, axis=0)
        positive_count = np.sum(sensor_data > 0, axis=0)
        values_above_mean = np.sum(sensor_data > mean, axis=0)
        
        peak_counts = np.array(np.apply_along_axis(lambda x: len(find_peaks(x)[0]), 0, sensor_data)).flatten()

        skewness = np.array(pd.DataFrame(sensor_data.reshape(-1,6)).skew().values).flatten()
        kurt = np.array(pd.DataFrame(sensor_data.reshape(-1,6)).kurtosis().values).flatten()
        energy = np.array(np.sum(sensor_data**2, axis=0)).flatten()

        # Compute the average resultant for gyro and acc columns
        gyro_cols = sensor_data[:, :3]
        acc_cols = sensor_data[:, 3:]
        gyro_avg_result = np.array(np.sqrt((gyro_cols**2).sum(axis=1)).mean()).flatten()
        acc_avg_result = np.array(np.sqrt((acc_cols**2).sum(axis=1)).mean()).flatten()

        # Compute the signal magnitude area for gyro and acc columns
        gyro_sma = np.array((np.abs(gyro_cols) / 100).sum(axis=0).sum()).flatten()
        acc_sma = np.array((np.abs(acc_cols) / 100).sum(axis=0).sum()).flatten()

        # Concatenate features and return as a list
        temp_features = np.concatenate([mean, std, abs_diff, minimum, maximum, max_min_diff, median, mad, iqr,
                                        negative_count, positive_count, values_above_mean, peak_counts, skewness, kurt, energy,
                                        gyro_avg_result, acc_avg_result, gyro_sma, acc_sma])
    
        return temp_features.tolist()

    # def rng_test_action(self):
    #     # choose a random action from the list
    #     chosen_action = random.choice(self.test_actions)

    #     # # print chosen action
    #     print(f'Chosen action: {chosen_action} \n')

    #     # use the chosen action to select the corresponding test data
    #     if chosen_action == 'G':
    #         test_data = self.test_g
    #     elif chosen_action == 'S':
    #         test_data = self.test_s
    #     elif chosen_action == 'L':
    #         test_data = self.test_l
    #     else:
    #         test_data = self.test_r

    #     return test_data

    # Define MLP - 3 layers
    def mlp_math(self, X):
        H1 = np.dot(X, self.weights[0]) + self.weights[1]
        H1_relu = np.maximum(0, H1)
        H2 = np.dot(H1_relu, self.weights[2]) + self.weights[3]
        H2_relu = np.maximum(0, H2)
        Y = np.dot(H2_relu, self.weights[4]) + self.weights[5]
        Y_softmax = np.exp(Y) / np.sum(np.exp(Y))
        return Y_softmax

    def get_action(self, softmax_array):
        max_index = np.argmax(softmax_array)
        action_dict = {0: 'G', 1: 'L', 2: 'R', 3: 'S'}
        # action_dict = {0: 'G', 1: 'R', 2: 'S'}
        action = action_dict[max_index]
        return action

#     def mlp_vivado(self, data):
#         start_time = time.time()

#         # reshape data to match in_buffer shape
#         data = np.reshape(data, (129,))

#         self.in_buffer[:] = data

#         self.dma.sendchannel.transfer(self.in_buffer)
#         self.dma.recvchannel.transfer(self.out_buffer)

#         # wait for transfer to finish
#         self.dma.sendchannel.wait()
#         self.dma.recvchannel.wait()

#         # print output buffer
#         print("mlp done with output: " + " ".join(str(x) for x in self.out_buffer))

#         print(f"MLP time taken so far output: {time.time() - start_time}")

#         return self.out_buffer

#     def mlp_vivado(data):
#         sensor_data = data.reshape(40, 6)
#         sensor_features = self.extract_features(sensor_data)
#         pca_action = self.pca_math(sensor_features)
#         mlp_softmax = self.mlp_math(pca_action)
#         return mlp_softmax

    def AIDriver(self, test_input):        
        sanity_data = test_input.reshape(1,-1)
        scaled_action_df = pd.DataFrame(sanity_data.reshape(-1,6))

        # 1. Feature extraction
        feature_vec = np.array(self.extract_features(scaled_action_df)).reshape(1,-1)

        # 2. Scaler using features
        scaled_action_math = (feature_vec - self.mean) / self.scale

        # 3. PCA using scaler
        pca_test_centered = scaled_action_math - self.mean_vec.reshape(1,-1)
        pca_vec_math = np.dot(pca_test_centered, self.pca_eigvecs.T).astype(float)

        # 4. MLP using PCA
        pred_math = self.mlp_math(np.array(pca_vec_math).reshape(1,-1))
        action_math = self.get_action(pred_math)

        print(pred_math)
        return str(action_math)

    def close_connection(self):
        self.shutdown.set()

        print("Shutting Down Connection")

    def run(self):
        # Set the threshold value for movement detection based on user input
        # K = 5
        # K = float(input("threshold value? "))

        # Initialize arrays to hold the current and previous data packets
        current_packet = np.zeros((5, 6))
        previous_packet = np.zeros((5, 6))
        data_packet = np.zeros((self.TOTAL_PACKET_COUNT, 6))
        is_movement_counter = 0
        movement_watchdog = False
        loop_count = 0
        # g2_acc_offset = [-0.810, 0.680, 11.660]
        # g1_acc_offset = [47.0, -981.0, -33.0]


        # live integration loop
        while True:
            if BlunoBeetle.packet_queue:  # TODO re-enable for live integration
                # if 1 == 1: # TODO DIS-enable for live integration

                # runs loop 6 times and packs the data into groups of 6
                q_data = BlunoBeetle.packet_queue.get()  # TODO re-enable for live integration
                new_data = np.array(q_data).astype(float)  # TODO re-enable for live integration

                # new_data[-3:] -= g1_acc_offset

                new_data = new_data / 100.0  # TODO re-enable for live integration


                # new_data = np.random.randn(6) # TODO DIS-enable for live integration
                print(" ".join([f"{x:.3f}" for x in new_data]))
                print(".\n")

                # Pack the data into groups of 6
                current_packet[loop_count] = new_data

                # Update loop_count
                loop_count = (loop_count + 1) % 5

                if loop_count % 5 == 0:
                    curr_mag =  np.sum(np.square(np.mean(current_packet[:, -3:], axis=1)))
                    prev_mag = np.sum(np.square(np.mean(previous_packet[:, -3:], axis=1)))

                    # Check for movement detection
                    if not movement_watchdog and curr_mag - prev_mag > self.K:
                        print("Movement detected!")
                        # print currr and prev mag for sanity check
                        print(f"curr_mag: {curr_mag} \n")
                        print(f"prev_mag: {prev_mag} \n")
                        movement_watchdog = True
                        # append previous and current packet to data packet
                        data_packet = np.concatenate((previous_packet, current_packet), axis=0)

                    # movement_watchdog activated, count is_movement_counter from 0 up 6 and append current packet each time
                    if movement_watchdog:
                        if is_movement_counter < ((self.TOTAL_PACKET_COUNT - 10)/5):
                            data_packet = np.concatenate((data_packet, current_packet), axis=0)
                            is_movement_counter += 1

                        # If we've seen 6 packets since the last movement detection, preprocess and classify the data
                        else:
                            # print dimensions of data packet
                            print(f"data_packet dimensions: {data_packet.shape} \n")
                            demo = pd.DataFrame(data_packet)
                            print(demo.head(self.TOTAL_PACKET_COUNT))

                            # rng_test_action = self.rng_test_action() # TODO DIS-enable for live integration
                            # action = self.AIDriver(rng_test_action) # TODO DIS-enable for live integration

                            action = self.AIDriver(data_packet)  # TODO re-enable for live integration
                            print(f"action from MLP in main: {action} \n")  # print output of MLP

                            # if action == 'G':
                            #     self.action_engine.handle_grenade(self.player)
                            # elif action == 'S':
                            #     self.action_engine.handle_shield(self.player)
                            # elif action == 'R':
                            #     self.action_engine.handle_reload(self.player)
                            # elif action == 'L':
                            #     self.action_engine.handle_logout(self.player)

                            # movement_watchdog deactivated, reset is_movement_counter
                            movement_watchdog = False
                            is_movement_counter = 0
                            # reset arrays to zeros
                            # current_packet = np.zeros((5, 6))
                            # previous_packet = np.zeros((5, 6))
                            data_packet = np.zeros((self.TOTAL_PACKET_COUNT, 6))

                    # Update the previous packet
                    previous_packet = current_packet.copy()


if __name__ == '__main__':
    # AI Model
    print("Starting AI Model Thread")
    ai_model = AIModel()
    ai_model.start()
    print('--------------------------------------------\n')
