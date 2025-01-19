# Basic tutorial in how to load data
import pandas as pd
import utils
import matplotlib.pyplot as plt
import numpy as np

p_res = 'data/processed_30hz_relabeled/0/freestyle_watch_resampled.csv'


# Use this to load a re-sampled recording
df_res = pd.read_csv(p_res)

def calculate_magnitude(data):
    """Calculate the magnitude from the sensor data."""
    return np.sqrt(np.sum(data**2, axis=1))
# Calculate magnitudes for each accelerometer axis
magnitude_acc_0 = calculate_magnitude(df_res[['ACC_0']].values)
magnitude_acc_1 = calculate_magnitude(df_res[['ACC_1']].values)
magnitude_acc_2 = calculate_magnitude(df_res[['ACC_2']].values)

# Create a new figure for the magnitude plot
fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True)

# Calculate the combined magnitude for the three accelerometer channels
combined_acc_magnitude = calculate_magnitude(df_res[['ACC_0', 'ACC_1', 'ACC_2']].values)
combined_gyro_magnitude = calculate_magnitude(df_res[['GYRO_2']].values)

# Create a new figure for the combined magnitude plot
fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True)

# Plot the combined magnitude against the timestamp
ax.plot(df_res['timestamp'].values, combined_acc_magnitude, label='Combined ACC Magnitude')
ax.plot(df_res['timestamp'].values, combined_gyro_magnitude, label='Combined GYRO Magnitude')

# Set titles and labels
ax.set_title("Combined Magnitude of Accelerometer Readings")
ax.set_xlabel("Timestamp")
ax.set_ylabel("Magnitude")
ax.legend()  # Add a legend to identify the line

# Show the plot
plt.show()


