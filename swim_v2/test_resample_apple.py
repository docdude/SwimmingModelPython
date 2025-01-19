import pandas as pd
import numpy as np
from scipy.signal import resample
import tensorflow as tf
import os

# Load Apple data
scaled_watch_path = 'data/processed_30hz_relabeled/0/freestyle_watch.csv'
scaled_watch_test = pd.read_csv(scaled_watch_path)

# Define columns
sensor_columns = ["ACC_0", "ACC_1", "ACC_2", "GYRO_0", "GYRO_1", "GYRO_2"]

# Resample Apple data to 30 Hz
def resample_data(data, columns, original_rate, target_rate):
    factor = original_rate // target_rate

    # Resample only the sensor data
    resampled_sensor_data = {
        col: resample(data[col].values, len(data[col]) // factor) for col in columns
    }

    # Keep the non-sensor data untouched
    non_sensor_columns = [col for col in data.columns if col not in columns]
    if len(non_sensor_columns) > 0:
        non_sensor_data = data[non_sensor_columns].iloc[:len(resampled_sensor_data[columns[0]])]
    else:
        non_sensor_data = pd.DataFrame()

    # Combine resampled sensor data and non-sensor data
    resampled_df = pd.DataFrame(resampled_sensor_data)
    resampled_df = pd.concat([resampled_df, non_sensor_data.reset_index(drop=True)], axis=1)

    return resampled_df

resampled_apple_data = resample_data(scaled_watch_test, sensor_columns, original_rate=100, target_rate=30)

# Normalize the resampled data
def normalize_data_statistical(data, columns):
    normalized_data = data.copy()
    for col in columns:
        mean = normalized_data[col].mean()
        std = normalized_data[col].std()
        normalized_data[col] = (normalized_data[col] - mean) / std
    return normalized_data

normalized_resampled_apple_data = normalize_data_statistical(resampled_apple_data, sensor_columns)

# Extract windows
def extract_windows(data, columns, win_len, slide_len):
    data_array = data[columns].values
    windows = [
        data_array[i : i + win_len] 
        for i in range(0, len(data_array) - win_len + 1, slide_len)
    ]
    return np.array(windows)

# Save the resampled data to a new CSV file
def save_resampled_data(data, original_path, suffix="_resampled"):
    # Extract the directory and filename from the original path
    directory, filename = os.path.split(original_path)
    base_name, ext = os.path.splitext(filename)
    
    # Construct the new filename
    new_filename = f"{base_name}{suffix}{ext}"
    
    # Save the resampled data
    save_path = os.path.join(directory, new_filename)
    data.to_csv(save_path, index=False)
    print(f"Resampled data saved to: {save_path}")


win_len = 180
slide_len = 30
windows_resampled_apple = extract_windows(normalized_resampled_apple_data, sensor_columns, win_len, slide_len)

# Reshape for model
windows_resampled_apple_reshaped = windows_resampled_apple.reshape(
    windows_resampled_apple.shape[0], windows_resampled_apple.shape[1], windows_resampled_apple.shape[2], 1
)

# Load the model
model_path = '/Users/juanloya/Documents/SwimmingModelPython/swim_v2/tutorial_save_path_epoch60_mag/22/model_best.keras'
model = tf.keras.models.load_model(model_path)

# Run predictions
predictions_resampled_apple = model.predict(windows_resampled_apple_reshaped)

# Analyze predictions
predicted_classes = np.argmax(predictions_resampled_apple, axis=1)
predicted_probabilities = predictions_resampled_apple.max(axis=1)

print("Predicted Classes (First 20):", predicted_classes[:20])
print("Predicted Probabilities (First 20):", predicted_probabilities[:20])

# Save the resampled data with "_resampled" suffix
save_resampled_data(resampled_apple_data, scaled_watch_path)
