
import threading
import traceback
import random

import time
import csv
import numpy as np
import pandas as pd
import pywt 
import scipy.signal as sig
from scipy import signal, stats
from scipy.stats import entropy, kurtosis, skew
import joblib
import json
# import pynq
# from pynq import Overlay


class Training(threading.Thread):
    def __init__(self):
        super().__init__()

        # Flags
        self.shutdown = threading.Event()

        self.columns = ['gx', 'gy', 'gz', 'accX', 'accY', 'accZ']

        self.factors = ['mean', 'std', 'variance', 'range', 'peak_to_peak_amplitude',
                    'mad', 'root_mean_square', 'interquartile_range', 'percentile_75',
                   'energy']

        self.num_groups = 8
        self.headers = [f'grp_{i+1}_{column}_{factor}' for i in range(self.num_groups)
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
        gx = random.uniform(-9, 9) # TODO - assumption: gyro x,y,z change btwn -9 to 9
        gy = random.uniform(-9, 9)
        gz = random.uniform(-9, 9)
        accX = random.uniform(-9, 9)
        accY = random.uniform(-9, 9)
        accZ = random.uniform(-9, 9)
        return [gx, gy, gz, accX, accY, accZ]
    
    # simulate game movement with noise and action
    def generate_simulated_wave(self):

        # base noise 10s long -> 20Hz*10 = 200 samples
        t = np.linspace(0, 5, 200) # Define the time range
        x1 = 0.2 * np.sin(t) + 0.2 * np.random.randn(200) 
        x1[(x1 > -1) & (x1 < 1)] = 0.0 # TODO - sensor noise within margin of error auto remove
        
        # movement motion
        period = 2  # seconds
        amplitude = 5
        t = np.linspace(0, 2, int(2 / 0.05)) # Define the time range
        x2 = amplitude * np.sin(2 * np.pi * t / period)[:40] # Compute the sine wave for only one cycle

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
        # skewness = stats.skew(data.reshape(-1, 1))[0]
        # kurtosis = stats.kurtosis(data.reshape(-1, 1))[0]
        energy = np.sum(data**2)
        output_array = [mean, std, variance, range, peak_to_peak_amplitude,
                        mad, root_mean_square, interquartile_range, percentile_75,
                        energy]

        output_array = np.array(output_array)                        

        return output_array.reshape(1, -1)
    
    def preprocess_dataset(self, df):
        processed_data = []

        # Set the window size for the median filter
        window_size = 7

        # Apply the median filter to each column of the DataFrame
        df_filtered = df.rolling(window_size, min_periods=1, center=True).median()

        df = df_filtered

        # Split the rows into 8 groups
        group_size = 5
        data_groups = [df.iloc[i:i+group_size,:] for i in range(0, len(df), group_size)]

        # Loop through each group and column, and compute features
        for group in data_groups:
            group_data = []
            for column in df.columns:
                column_data = group[column].values
                column_data = column_data.reshape(1, -1)

                temp_processed = self.preprocess_data(column_data)
                temp_processed = temp_processed.flatten()

                group_data.append(temp_processed)
                
            processed_data.append(np.concatenate(group_data))

        # Combine the processed data for each group into a single array
        processed_data_arr = np.concatenate(processed_data)

        print(f"len processed_data_arr={len(processed_data_arr)}\n")

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
        test_input = np.array([0.1, 0.2, 0.3, 0.4] * 120).reshape(1,-1)

        arr = np.array([-9.20434773, -4.93421279, -0.7165668, -5.35652778, 1.16597442, 0.83953718,
                2.46925983, 0.55131264, -0.1671036, 0.82080829, -1.87265269, 3.34199444,
                0.09530707, -3.77394007, 1.68183889, 1.97630386, 1.48839111, -3.00986825,
                4.13786954, 1.46723819, 8.08842927, 10.94846901, 2.22280215, -1.85681443,
                4.47327707, 3.15918201, -0.77879694, -0.11557772, 0.21580221, -2.62405631,
                -3.42924226, -7.01213438, 7.75544419, -3.72408571, 3.46613566])

        # Scaler
        # test_input_rescaled = (data - self.mean) / np.sqrt(self.variance) # TODO - use this for real data
        test_input_rescaled = (test_input - self.mean) / np.sqrt(self.variance)
        print(f"test_input_rescaled: {test_input_rescaled}\n")

        # PCA
        test_input_math_pca = np.dot(test_input_rescaled, self.pca_eigvecs_transposed)
        print(f"test_input_math_pca: {test_input_math_pca}\n")

        # arr = np.array([-9.20434773, -4.93421279, -0.7165668, -5.35652778, 1.16597442, 0.83953718,
        #         2.46925983, 0.55131264, -0.1671036, 0.82080829, -1.87265269, 3.34199444,
        #         0.09530707, -3.77394007, 1.68183889, 1.97630386, 1.48839111, -3.00986825,
        #         4.13786954, 1.46723819, 8.08842927, 10.94846901, 2.22280215, -1.85681443,
        #         4.47327707, 3.15918201, -0.77879694, -0.11557772, 0.21580221, -2.62405631,
        #         -3.42924226, -7.01213438, 7.75544419, -3.72408571, 3.46613566])

        # assert np.allclose(test_input_math_pca, arr)

        # MLP
        # predicted_labels = self.PCA_MLP(test_input_math_pca) # return 1x4 softmax array
        # print(f"MLP pynq overlay predicted: {predicted_labels} \n")
        # np_output = np.array(predicted_labels)
        # largest_index = np_output.argmax()

        # predicted_label = self.action_map[largest_index]

        # # print largest index and largest action of MLP output
        # print(f"largest index: {largest_index} \n")
        # print(f"MLP overlay predicted: {predicted_label} \n")


        predicted_label = mlp.predict(test_input_math_pca.reshape(1, -1)) 
        print(f"MLP lib overlay predicted: {predicted_label} \n")

        

        # output is a single char
        return predicted_label


    def close_connection(self):
        self.shutdown.set()

        print("Shutting Down Connection")

    def run(self):
        
        # live integration loop
        df = pd.DataFrame(np.zeros((500, len(self.columns))), columns=self.columns)
        # Define the window size and threshold factor
        window_size = 11
        threshold_factor = 2

        # Define N units for flagging movement, 20Hz -> 2s = 40 samples
        N = 40

        # Initialize empty arrays for data storage
        x = []
        filtered = []
        threshold = []
        movement_detected = []
        last_movement_time = -N  # set last movement time to negative N seconds ago
        wave = self.generate_simulated_wave()
        i = 0
        buffer_index = 0

        # while not self.shutdown.is_set():
        while True:
                data = self.generate_simulated_data()
                self.sleep(0.05)
                print("Data: ")
                print(" ".join([f"{x:.8g}" for x in data]))
                print("\n")

                # Append new data to dataframe
                df.iloc[buffer_index] = data

                # Increment buffer index and reset to zero if we reach the end of the buffer
                buffer_index += 1
                if buffer_index >= 500:
                    buffer_index = 0

                # Compute absolute acceleration values
                # x.append(np.abs(data[3:6])) # abs of accX, accY, accZ
                x.append(wave[i]) # abs of accX, accY, accZ


                # Compute moving window median
                if len(x) < window_size:
                    filtered.append(0)
                else:
                    filtered.append(np.median(x[-window_size:], axis=0))

                # Compute threshold using past median data, threshold = mean + k * std
                if len(filtered) < window_size:
                    threshold.append(0)
                else:
                    past_filtered = filtered[-window_size:]
                    threshold.append(np.mean(past_filtered, axis=0) + (threshold_factor * np.std(past_filtered, axis=0)))
                
                # Identify movement
                if len(filtered) > window_size:
                    # checking if val is past threshold and if last movement was more than N samples ago
                    if np.all(filtered[-1] > threshold[-1]) and len(filtered) - last_movement_time >= N:
                        movement_detected.append(buffer_index)
                        last_movement_time = len(filtered)  # update last movement time
                        print(f"Movement detected at sample {buffer_index}")

                # if movement has been detected for more than N samples, preprocess and feed into neural network
                if len(movement_detected) > 0 and buffer_index - movement_detected[-1] >= N:
                    # extract movement data
                    start = movement_detected[-1]
                    end = buffer_index if buffer_index > start else buffer_index + 500
                    movement_data = df.iloc[start:end, :]

                    # print the start and end index of the movement
                    print(f"Processing movement detected from sample {start} to {end}")

                    # perform data preprocessing
                    preprocessed_data = self.preprocess_dataset(movement_data)
            
                    # feed preprocessed data into neural network
                    predicted_label = self.instantMLP(preprocessed_data)
                    
                    print(f"output from MLP: \n {predicted_label} \n") # print output of MLP

                    # reset movement_detected list
                    movement_detected.clear()

                i += 1
                if i == 200:
                    i = 0

            # except Exception as _:
            #     traceback.print_exc()
            #     self.close_connection()
            #     print("an error occurred")


if __name__ == '__main__':
    # AI Model
    print("Starting AI Model Thread")
    ai_model = Training()
    ai_model.start()
    print('--------------------------------------------')
