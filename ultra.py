
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

# import matplotlib.pyplot as plt
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
        self.action_map = {0: 'GRENADE', 1: 'LOGOUT', 2: 'SHIELD', 3: 'RELOAD'}

        # PYNQ overlay - TODO
        # self.overlay = Overlay("design_3.bit")
        # self.dma = self.overlay.axi_dma_0

    def sleep(self, seconds):
        start_time = time.time()
        while time.time() - start_time < seconds:
            pass

    def generate_simulated_data(self):
        gx = random.uniform(-180, 180)
        gy = random.uniform(-180, 180)
        gz = random.uniform(-180, 180)
        accX = random.uniform(-9000, 9000)
        accY = random.uniform(-9000, 9000)
        accZ = random.uniform(-9000, 9000)
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
    
    def MLP(self, data):
        start_time = time.time()
        # allocate in and out buffer
        in_buffer = pynq.allocate(shape=(24,), dtype=np.double)

        # print time taken so far 
        print(f"MLP time taken so far in_buffer: {time.time() - start_time}")
        # out buffer of 1 integer
        out_buffer = pynq.allocate(shape=(1,), dtype=np.int32)
        print(f"MLP time taken so far out_buffer: {time.time() - start_time}")

        # # TODO - copy all data to in buffer
        # for i, val in enumerate(data):
        #     in_buffer[i] = val

        for i, val in enumerate(data[:24]):
            in_buffer[i] = val

        print(f"MLP time taken so far begin trf: {time.time() - start_time}")

        self.dma.sendchannel.transfer(in_buffer)
        self.dma.recvchannel.transfer(out_buffer)

        print(f"MLP time taken so far end trf: {time.time() - start_time}")


        # wait for transfer to finish
        self.dma.sendchannel.wait()
        self.dma.recvchannel.wait()

        print(f"MLP time taken so far wait: {time.time() - start_time}")


        # print("mlp done \n")

        # print output buffer
        for output in out_buffer:
            print(f"mlp done with output {output}")
        
        print(f"MLP time taken so far output: {time.time() - start_time}")

        return [random.random() for _ in range(4)]
    

    def instantMLP(self, data):
        # Define the input weights and biases
        # w1 = np.random.rand(24, 10)
        # b1 = np.random.rand(10)
        # w2 = np.random.rand(10, 20)
        # b2 = np.random.rand(20)
        # w3 = np.random.rand(20, 4)
        # b3 = np.random.rand(4)

        # # Perform the forward propagation
        # a1 = np.dot(data[:24], w1) + b1
        # h1 = np.maximum(0, a1)  # ReLU activation
        # a2 = np.dot(h1, w2) + b2
        # h2 = np.maximum(0, a2)  # ReLU activation
        # a3 = np.dot(h2, w3) + b3

        # c = np.max(a3)
        # exp_a3 = np.exp(a3 - c)
        # softmax_output = exp_a3 / np.sum(exp_a3)  # Softmax activation
        
        # return softmax_output

        # Load the model from file and preproessing
        # localhost
        mlp = joblib.load('mlp_model.joblib')
        scaler = joblib.load('scaler.joblib')
        pca = joblib.load('pca.joblib')
        
        # board
        # mlp = joblib.load('/home/xilinx/mlp_model.joblib')
        # scaler = joblib.load('/home/xilinx/scaler.joblib')
        # pca = joblib.load('/home/xilinx/pca.joblib')

        test_data_std = scaler.transform(data.reshape(1,-1))
        test_data_pca = pca.transform(test_data_std)

        # Use the loaded MLP model to predict labels for the test data
        predicted_labels = mlp.predict(test_data_pca)

        predicted_label = str(predicted_labels[0].item()) # convert to single char

        return predicted_label


    def close_connection(self):
        self.shutdown.set()

        print("Shutting Down Connection")

    def run(self):
        
        # live integration loop
        # while not self.shutdown.is_set():
        f = True
        while f:
            f = False

            df = pd.DataFrame(columns=self.columns)
            # Define the window size and threshold factor
            window_size = 11
            threshold_factor = 2

            # Define N units for flagging movement, 20Hz -> 2s = 40 samples
            N = 40

            # Initialize empty arrays for data storage
            t = []
            x = []
            filtered = []
            threshold = []
            movement_detected = []
            last_movement_time = -N  # set last movement time to negative N seconds ago
            wave = self.generate_simulated_wave()
            i = 0
            timenow = 0

            print(f"entering while loop \n")

            while True:
                # Create plot window
                # plt.ion()
                # plt.show()

                data = self.generate_simulated_data()
                self.sleep(0.05)
                print("Data: ")
                print(" ".join([f"{x:.8g}" for x in data]))
                print("\n")

                # Append new data to dataframe
                df.loc[len(df)] = data

                # Compute absolute acceleration values
                # x.append(np.abs(data[5:8])) # abs of accX, accY, accZ
                x.append(wave[i]) # abs of accX, accY, accZ

                # time
                t.append(timenow)

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
                    if np.all(filtered[-1] > threshold[-1]) and len(t) - last_movement_time >= N:
                        movement_detected.append(len(df) - 1)
                        last_movement_time = len(t)  # update last movement time
                        print(f"Movement detected at sample {len(df) - 1}")

                # if movement has been detected for more than N samples, preprocess and feed into neural network
                if len(movement_detected) > 0 and len(df) - movement_detected[-1] >= N:
                    # extract movement data
                    start = movement_detected[-1]
                    end = len(df)
                    movement_data = df.iloc[start:end, :]

                    # print the start and end index of the movement
                    print(f"Processing movement detected from sample {start} to {end}")

                    # perform data preprocessing
                    preprocessed_data = self.preprocess_dataset(movement_data)

                    # print preprocessed data
                    print(f"preprocessed data to feed into MLP: \n {preprocessed_data} \n")
                    
                    # feed preprocessed data into neural network
                    # output = self.MLP(preprocessed_data) 
                    predicted_label = self.instantMLP(preprocessed_data)
                    
                    print(f"output from MLP: \n {predicted_label} \n") # print output of MLP

                    # np_output = np.array(output)
                    # largest_index = np_output.argmax()

                    # largest_action = self.action_map[largest_index]

                    # print largest index and largest action of MLP output
                    # print(f"largest index: {largest_index} \n")
                    # print(f"largest action: {largest_action} \n")

                    # reset movement_detected list
                    movement_detected.clear()

                i += 1
                timenow += 1

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