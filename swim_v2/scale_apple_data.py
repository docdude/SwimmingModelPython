import pandas as pd
import numpy as np

# Load the Apple Watch data from the provided CSV file
file_path = 'data/processed_30hz_relabeled/0/freestyle_watch.csv'
apple_watch_data = pd.read_csv(file_path)

# Example statistics from Android and Apple Watch data
android_stats = {
    "mean": [-6.631457, -2.106495, 1.759229, -0.712199, 0.008983, 0.056397],
    "std": [7.595074, 7.840088, 6.930109, 3.080702, 3.437194, 2.766478]
}

apple_stats = {
    "mean": [0.266803, 0.215007, 0.062129, -0.705502, -0.196389, -0.258351],
    "std": [0.401326, 0.437971, 0.339910, 1.993001, 2.445922, 2.327683]
}

# Convert stats to arrays for easier manipulation
android_mean = np.array(android_stats["mean"])
android_std = np.array(android_stats["std"])
apple_mean = np.array(apple_stats["mean"])
apple_std = np.array(apple_stats["std"])

# Scaling function
def scale_apple_to_android(apple_data):
    """
    Scales Apple Watch data to match Android sensor data scale.

    Args:
    - apple_data (pd.DataFrame or np.ndarray): Apple Watch sensor data. Columns must match Android stats.

    Returns:
    - scaled_data (np.ndarray): Scaled data to Android scale.
    """
    # Ensure input is a NumPy array
    apple_data = np.array(apple_data)

    # Scale data: Adjust mean and standard deviation
    scaled_data = ((apple_data - apple_mean) / apple_std) * android_std + android_mean
    return scaled_data

# Ensure the columns align with the expected order (ACC_0, ACC_1, ACC_2, GYRO_0, GYRO_1, GYRO_2)
columns_to_scale = ["ACC_0", "ACC_1", "ACC_2", "GYRO_0", "GYRO_1", "GYRO_2"]

# Replace the original data with the scaled data in the same columns
apple_watch_data[columns_to_scale] = scale_apple_to_android(apple_watch_data[columns_to_scale])

# Save the scaled data to a new CSV for validation
scaled_output_path = 'data/processed_30hz_relabeled/0/freestyle_watch_scaled.csv'
apple_watch_data.to_csv(scaled_output_path, index=False)

print(f"Scaled data saved to {scaled_output_path}")
