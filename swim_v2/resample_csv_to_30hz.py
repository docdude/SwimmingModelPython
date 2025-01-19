import pandas as pd
import numpy as np
from scipy.signal import resample
import os

# Load Apple data
scaled_watch_path = 'data/processed_30hz_relabeled/0/frog.csv'
scaled_watch_test = pd.read_csv(scaled_watch_path)

# Define columns
sensor_columns = ["ACC_0", "ACC_1", "ACC_2", "GYRO_0", "GYRO_1", "GYRO_2", "MAG_0", "MAG_1", "MAG_2"]
timestamp_column = "timestamp"  # Replace with the actual column name for the timestamp
label_column = "label"  # Replace with the actual column name for the label

# Correct Apple axis orientations
def correct_apple_axis(data, sensor_columns, platform="apple"):
    """
    Correct axis orientation for Apple Watch data to align with Android.

    Parameters:
    - data: DataFrame containing the sensor data.
    - sensor_columns: List of sensor columns to adjust.
    - platform: Specify the platform ('apple' or 'android').

    Returns:
    - Adjusted DataFrame.
    """
    corrected_data = data.copy()
    
    if platform.lower() == "apple":
        # Invert accelerometer axes
        acc_columns = [col for col in sensor_columns if "ACC" in col]
        for col in acc_columns:
            corrected_data[col] = -corrected_data[col]  # Invert acceleration values
        
        # Invert gyroscope Z-axis
        """
        gyro_z_column = "GYRO_2"
        if gyro_z_column in corrected_data.columns:
            corrected_data[gyro_z_column] = -corrected_data[gyro_z_column]  # Invert Gyro Z values
        """
        # Invert gyroscope axis
        gyro_columns = [col for col in sensor_columns if "GYRO" in col]
        for col in gyro_columns:
            corrected_data[col] = -corrected_data[col]  # Invert gyro values

    return corrected_data

scaled_watch_test = correct_apple_axis(scaled_watch_test, sensor_columns, platform="apple")

# Resample Apple data to 30 Hz
def resample_data(data, sensor_columns, timestamp_column, label_column, original_rate, target_rate):
    """
    Resample sensor data while preserving timestamp and other non-sensor columns.

    Parameters:
    - data: DataFrame containing the data
    - sensor_columns: List of sensor columns to resample
    - timestamp_column: Column containing the timestamp
    - label_column: Column containing the label
    - original_rate: Original sampling rate of the data
    - target_rate: Desired sampling rate of the data

    Returns:
    - DataFrame with resampled sensor data and untouched non-sensor columns
    """
    factor = original_rate // target_rate

    # Resample only the sensor data
    resampled_sensor_data = {
        col: resample(data[col].values, len(data[col]) // factor) for col in sensor_columns
    }

    # Resample the timestamp to match the new length
    if timestamp_column in data.columns:
        resampled_sensor_data[timestamp_column] = resample(data[timestamp_column].values, len(data[sensor_columns[0]]) // factor)

    # Resample the label to match the new length
    if label_column in data.columns:
        resampled_sensor_data[label_column] = resample(data[label_column].values, len(data[sensor_columns[0]]) // factor)

    # Create a DataFrame from resampled data
    resampled_df = pd.DataFrame(resampled_sensor_data)

    # Ensure the correct column order
    column_order = [timestamp_column] + sensor_columns + [label_column]
    resampled_df = resampled_df[column_order]

    return resampled_df

resampled_apple_data = resample_data(
    scaled_watch_test, 
    sensor_columns=sensor_columns, 
    timestamp_column=timestamp_column, 
    label_column=label_column, 
    original_rate=100, 
    target_rate=30
)

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

# Save the resampled data with "_resampled" suffix
save_resampled_data(resampled_apple_data, scaled_watch_path)
