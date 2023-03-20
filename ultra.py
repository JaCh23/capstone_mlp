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
