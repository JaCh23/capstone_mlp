
import threading
import traceback
import random

import time
import numpy as np
import pandas as pd
from scipy import stats
import joblib
import json
# import pynq
# from pynq import Overlay


class AIModel(threading.Thread):
    def __init__(self):
        super().__init__()

        # Flags
        self.shutdown = threading.Event()

        self.columns = ['gx', 'gy', 'gz', 'accX', 'accY', 'accZ']

        self.factors = ['mean', 'std', 'variance', 'range', 'peak_to_peak_amplitude',
                        'mad', 'root_mean_square', 'interquartile_range', 'percentile_75',
                        'energy']

        self.num_groups = 8
        self.headers = [f'grp_{i + 1}_{column}_{factor}' for i in range(self.num_groups)
                        for column in self.columns for factor in self.factors]

        self.headers.extend(['action'])

        # defining game action dictionary
        self.action_map = {0: 'G', 1: 'L', 2: 'R', 3: 'S'}

        # load PCA model
        # read the contents of the arrays.txt file
        with open("arrays.txt", "r") as f:
            data = json.load(f)

        # extract the weights and bias arrays
        self.scaling_factor = data['scaling_factor']
        self.mean = data['mean']
        self.variance = data['variance']
        self.pca_eigvecs_list = data['pca_eigvecs_list']

        self.pca_eigvecs_transposed = [list(row) for row in zip(*self.pca_eigvecs_list)]
        
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

    def generate_simulated_data(self):
        gx = random.uniform(-9, 9)  # TODO - assumption: gyro x,y,z change btwn -9 to 9
        gy = random.uniform(-9, 9)
        gz = random.uniform(-9, 9)
        accX = random.uniform(-9, 9)
        accY = random.uniform(-9, 9)
        accZ = random.uniform(-9, 9)
        return [gx, gy, gz, accX, accY, accZ]

    # simulate game movement with noise and action
    def generate_simulated_wave(self):

        # base noise 10s long -> 20Hz*10 = 200 samples
        t = np.linspace(0, 5, 200)  # Define the time range
        x1 = 0.2 * np.sin(t) + 0.2 * np.random.randn(200)
        x1[(x1 > -1) & (x1 < 1)] = 0.0  # TODO - sensor noise within margin of error auto remove

        # movement motion
        period = 2  # seconds
        amplitude = 5
        t = np.linspace(0, 2, int(2 / 0.05))  # Define the time range
        x2 = amplitude * np.sin(2 * np.pi * t / period)[:40]  # Compute the sine wave for only one cycle

        x = x1
        # Add to the 40th-80th elements
        x[20:60] += x2
        x[80:120] += x2

        return x

    # 10 features
    def preprocess_data(self, data):
        # standard data processing techniques
        mean = np.mean(data)
        std = np.std(data)
        variance = np.var(data)
        range = np.max(data) - np.min(data)
        peak_to_peak_amplitude = np.abs(np.max(data) - np.min(data))
        mad = np.median(np.abs(data - np.median(data)))
        root_mean_square = np.sqrt(np.mean(np.square(data)))
        interquartile_range = stats.iqr(data)
        percentile_75 = np.percentile(data, 75)
        energy = np.sum(data ** 2)

        output_array = np.empty((1, 10))
        output_array[0] = [mean, std, variance, range, peak_to_peak_amplitude, mad, root_mean_square,
                           interquartile_range, percentile_75, energy]

        return output_array

    def preprocess_dataset(self, arr):
        processed_data = []

        # Set the window size for the median filter
        window_size = 7

        df = pd.DataFrame(arr)
        df_filtered = df.rolling(window_size, min_periods=1, center=True).median()

        arr = df_filtered.values

        # Split the rows into 8 groups
        group_size = 5
        num_groups = 8

        # Loop through each group and column, and compute features
        for i in range(num_groups):
            start_idx = i * group_size
            end_idx = start_idx + group_size
            group = arr[start_idx:end_idx, :]

            group_data = []
            for column in range(arr.shape[1]):
                column_data = group[:, column]
                column_data = column_data.reshape(1, -1)

                temp_processed = self.preprocess_data(column_data)
                temp_processed = temp_processed.flatten()

                group_data.append(temp_processed)

            processed_data.append(np.concatenate(group_data))

        processed_data_arr = np.concatenate(processed_data)

        # print(f"len processed_data_arr={len(processed_data_arr)}\n")

        return processed_data_arr

    def PCA_MLP(self, data):
        start_time = time.time()

        # reshape data to match in_buffer shape
        data = np.reshape(data, (35,))

        self.in_buffer[:] = data

        self.dma.sendchannel.transfer(self.in_buffer)
        self.dma.recvchannel.transfer(self.out_buffer)

        # wait for transfer to finish
        self.dma.sendchannel.wait()
        self.dma.recvchannel.wait()

        # print output buffer
        print("mlp done with output: " + " ".join(str(x) for x in self.out_buffer))

        print(f"MLP time taken so far output: {time.time() - start_time}")

        return self.out_buffer

    def instantMLP(self, data):
        # Load the model from file and preproessing
        # localhost
        mlp = joblib.load('mlp_model.joblib')

        # board
        # mlp = joblib.load('/home/xilinx/mlp_model.joblib')

        # sample data for sanity check
        test_input = np.array([0.1, 0.2, 0.3, 0.4] * 120).reshape(1, -1)

        # Scaler
        # test_input_rescaled = (data - self.mean) / np.sqrt(self.variance) # TODO - use this for real data
        test_input_rescaled = (test_input - self.mean) / np.sqrt(self.variance)
        print(f"test_input_rescaled: {test_input_rescaled}\n")

        # PCA
        test_input_math_pca = np.dot(test_input_rescaled, self.pca_eigvecs_transposed)
        print(f"test_input_math_pca: {test_input_math_pca}\n")

        # MLP - TODO PYNQ Overlay
        # predicted_labels = self.PCA_MLP(test_input_math_pca) # return 1x4 softmax array
        # print(f"MLP pynq overlay predicted: {predicted_labels} \n")
        # np_output = np.array(predicted_labels)
        # largest_index = np_output.argmax()

        # predicted_label = self.action_map[largest_index]

        # # print largest index and largest action of MLP output
        # print(f"largest index: {largest_index} \n")
        # print(f"MLP overlay predicted: {predicted_label} \n")

        # MLP - LIB Overlay
        predicted_label = mlp.predict(test_input_math_pca.reshape(1, -1))
        print(f"MLP lib overlay predicted: {predicted_label} \n")

        return predicted_label

    def close_connection(self):
        self.shutdown.set()

        print("Shutting Down Connection")

    def run(self):
        # live integration loop
        window_size = 11
        threshold_factor = 3

        buffer_size = 500
        buffer = np.zeros((buffer_size, len(self.columns)))
        # Define the window size and threshold factor

        # Define N units for flagging movement, 20Hz -> 2s = 40 samples
        N = 40

        # Initialize empty arrays for data storage
        x = np.zeros(buffer_size)
        filtered = np.zeros(buffer_size)
        threshold = np.zeros(buffer_size)
        last_movement_time = -N  # set last movement time to negative N seconds ago
        wave = self.generate_simulated_wave()
        i = 0
        buffer_index = 0

        # while not self.shutdown.is_set():
        while True:
            if 1 == 1:
            # if ai_queue:

                # Live data
                # data = np.array(ai_queue.popleft())
                # data[-3:] = [x/100.0 for x in data[-3:]]

                # # Simulated data
                data = self.generate_simulated_data()
                # self.sleep(0.05)
                # print("Data: ")
                # print(" ".join([f"{x:.3f}" for x in data]))
                # print("\n")

                # Append new data
                buffer[buffer_index] = data

                # Update circular buffer index
                buffer_index = (buffer_index + 1) % buffer_size

                # Compute absolute acceleration values
                # x[buffer_index] = np.abs(np.sum(np.square(data[3:6])))  # abs of accX, accY, accZ
                x[buffer_index] = wave[i]  # abs of accX, accY, accZ

                i += 1
                if i >= len(wave):
                    i = 0

                # Compute moving window median
                filtered[buffer_index] = np.median(x[buffer_index - window_size + 1:buffer_index + 1], axis=0)

                # Compute threshold using past median data, threshold = mean + k * std
                past_filtered = filtered[buffer_index - window_size + 1:buffer_index + 1]
                threshold[buffer_index] = np.mean(past_filtered, axis=0) + (
                                threshold_factor * np.std(past_filtered, axis=0))

                # Identify movement
                if filtered[buffer_index] > threshold[buffer_index]:
                    last_movement_time = buffer_index  # update last movement time
                    print(f"Movement detected at sample {buffer_index}")

                # if N samples from last movement time have been accumulated, preprocess and feed into neural network
                if (last_movement_time != -N) and (buffer_index - last_movement_time + 1) % buffer_size == N:
                    start = last_movement_time
                    # +1 needed for python syntax, eg we want [1,40] but syntax is [1,41]
                    end = (buffer_index + 1) % buffer_size
                    if end <= start:
                        movement_data = np.concatenate((buffer[start:, :], buffer[:end, :]), axis=0)
                    else:
                        movement_data = buffer[start:end, :]

                    # print the start and end index of the movement
                    print(f"Processing movement detected from sample {start} to {end}")

                    # perform data preprocessing
                    preprocessed_data = self.preprocess_dataset(movement_data) # multithread

                    # feed preprocessed data into neural network
                    predicted_label = self.instantMLP(preprocessed_data) # multithread

                    print(f"output from MLP: \n {predicted_label} \n")  # print output of MLP


            # except Exception as _:
            #     traceback.print_exc()
            #     self.close_connection()
            #     print("an error occurred")


if __name__ == '__main__':
    # AI Model
    print("Starting AI Model Thread")
    ai_model = AIModel()
    ai_model.start()
    print('--------------------------------------------')
