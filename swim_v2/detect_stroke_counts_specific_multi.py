# detect_stroke_counts_specific.py
import os
import pickle
import numpy as np
import tensorflow as tf
import learning_data
#import utils
import scipy.signal  # For peak detection
import matplotlib.pyplot as plt
from collections import deque
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean as scipy_euclidean
import pywt
from tqdm import tqdm  # Change the import statement
import pandas as pd

import multiprocessing
from multiprocessing import Pool, Manager, Process

label_names_abb = ['Null', 'Fr', 'Br', 'Ba', 'Bu']

# Stroke-style-specific axis and parameter mapping
stroke_axis_params = {
    1: {"acc_axes": [0, 1, 2], "gyro_axes": [0, 1, 2], "prominence": 1.5, "distance": 35},  # Freestyle
    2: {"acc_axes": [0, 1, 2], "gyro_axes": [0, 1, 2], "prominence": 1.6, "distance": 50},  # Breaststroke
    3: {"acc_axes": [0, 1, 2], "gyro_axes": [0, 1, 2], "prominence": 1.2, "distance": 50},  # Backstroke
    4: {"acc_axes": [0, 1, 2], "gyro_axes": [0, 1, 2], "prominence": 1.6, "distance": 45},  # Butterfly
}

def plot_data_with_peaks(
    sensor_data,
    acc_peaks,
    gyro_peaks,
    predicted_style,
    window_index,
    user_number=None,
    file_name=None
):
    """
    Plot accelerometer and gyroscope data with detected peaks.
    """
    # Debug print
    print(f"Plotting peaks for style: {predicted_style}")
    print("Received acc_peaks:", acc_peaks)
    print("Received gyro_peaks:", gyro_peaks)

    # Create figure
    fig, axs = plt.subplots(2, 5, figsize=(15, 8))
    fig.suptitle(
        f"User: {user_number}, File: {file_name}, Window: {window_index} (Style: {label_names_abb[predicted_style]})",
        fontsize=16,
    )

    # Plot raw accelerometer axes
    for i, axis_label in enumerate(["X", "Y", "Z"]):
        axs[0, i].plot(sensor_data["acc_filtered"][:, i], label=f"Acc {axis_label}", color="blue")
        axs[0, i].set_title(f"Accelerometer {axis_label}")
        axs[0, i].grid(True)
        axs[0, i].legend()

    # Plot raw accelerometer magnitude
    axs[0, 3].plot(sensor_data["acc_magnitude_filtered"], label="Acc Magnitude", color="green")
    axs[0, 3].set_title("Accelerometer Magnitude")
    axs[0, 3].grid(True)
    axs[0, 3].legend()

    # Plot style-specific accelerometer data
    acc_label = sensor_data["style_acc_peak_key"]
    axs[0, 4].plot(sensor_data["style_acc"], label=acc_label, color="purple")
    if acc_label in acc_peaks:
        peak_indices = acc_peaks[acc_label]
        axs[0, 4].plot(
            peak_indices,
            sensor_data["style_acc"][peak_indices],
            "ro",
            label=f"{acc_label} Peaks"
        )
    axs[0, 4].set_title(f"Style-Specific {acc_label}")
    axs[0, 4].grid(True)
    axs[0, 4].legend(loc='best')

    # Plot raw gyroscope axes
    for i, axis_label in enumerate(["X", "Y", "Z"]):
        axs[1, i].plot(sensor_data["gyro_filtered"][:, i], label=f"Gyro {axis_label}", color="orange")
        axs[1, i].set_title(f"Gyroscope {axis_label}")
        axs[1, i].grid(True)
        axs[1, i].legend()

    # Plot raw gyroscope magnitude
    axs[1, 3].plot(sensor_data["gyro_magnitude_filtered"], label="Gyro Magnitude", color="brown")
    axs[1, 3].set_title("Gyroscope Magnitude")
    axs[1, 3].grid(True)
    axs[1, 3].legend()

    # Plot style-specific gyroscope data
    gyro_label = sensor_data["style_gyro_peak_key"]
    axs[1, 4].plot(sensor_data["style_gyro"], label=gyro_label, color="red")
    if gyro_label in gyro_peaks:
        peak_indices = gyro_peaks[gyro_label]
        axs[1, 4].plot(
            peak_indices,
            sensor_data["style_gyro"][peak_indices],
            "ro",
            label=f"{gyro_label} Peaks"
        )
    axs[1, 4].set_title(f"Style-Specific {gyro_label}")
    axs[1, 4].grid(True)
    axs[1, 4].legend(loc='best')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    
class RealTimePlotter:
    def __init__(self, window_size=180, history_size=540, user_number=None, file_name=None):
        self.window_size = window_size
        self.history_size = history_size
        self.user_number = user_number
        self.file_name = file_name
        self.fig = plt.figure(figsize=(20, 12))
        self.gs = self.fig.add_gridspec(3, 4, height_ratios=[2, 2, 1])  # Updated for additional plots
        self.ax = [
            [self.fig.add_subplot(self.gs[0, i]) for i in range(4)],  # Accelerometer plots
            [self.fig.add_subplot(self.gs[1, i]) for i in range(4)],  # Gyroscope plots
            self.fig.add_subplot(self.gs[2, :])  # Stroke count bar chart
        ]
        self.history_acc = deque(maxlen=history_size)
        self.history_gyro = deque(maxlen=history_size)
        self.style_colors = {
            0: 'gray',    # Null/Turn
            1: 'blue',    # Front Crawl
            2: 'green',   # Breaststroke
            3: 'red',     # Backstroke
            4: 'purple'   # Butterfly
        }
        self.stroke_counts = {label: 0 for label in label_names_abb}
        self.setup_plot()

    def setup_plot(self):
        for i in range(3):  # Individual axes plots
            self.ax[0][i].set_xlim(0, self.history_size)
            self.ax[0][i].set_ylim(-5, 5)  # Adjust as per data
            self.ax[0][i].grid(True)
            self.ax[1][i].set_xlim(0, self.history_size)
            self.ax[1][i].set_ylim(-5, 5)  # Adjust as per data
            self.ax[1][i].grid(True)

        self.ax[0][3].set_xlim(0, self.history_size)  # Magnitude plots
        self.ax[0][3].set_ylim(0, 10)  # Adjust as per data
        self.ax[0][3].grid(True)

        self.ax[1][3].set_xlim(0, self.history_size)
        self.ax[1][3].set_ylim(0, 10)  # Adjust as per data
        self.ax[1][3].grid(True)

        self.ax[2].set_xlim(-1, len(label_names_abb))
        self.ax[2].set_ylim(0, 10)
        self.ax[2].grid(True)
        self.ax[2].set_xticks(range(len(label_names_abb)))
        self.ax[2].set_xticklabels(label_names_abb)
        
        self.fig.tight_layout()
        plt.ion()

    def update_stroke_counts(self, predicted_style, acc_peaks, gyro_peaks):
        if predicted_style != 0:
            style_name = label_names_abb[predicted_style]
            valid_strokes = 0
            used_acc_peaks = set()
            used_gyro_peaks = set()

            for acc_peak in acc_peaks:
                for gyro_peak in gyro_peaks:
                    if abs(acc_peak - gyro_peak) <= 3:  # Within 3 samples
                        if acc_peak not in used_acc_peaks and gyro_peak not in used_gyro_peaks:
                            valid_strokes += 1
                            used_acc_peaks.add(acc_peak)
                            used_gyro_peaks.add(gyro_peak)
                            break

            self.stroke_counts[style_name] += valid_strokes

    def update_plot(
        self, acc_data, gyro_data, acc_magnitude, gyro_magnitude,
        predicted_style, acc_peaks, gyro_peaks, style_confidence, window_index
    ):
        # Ensure predicted_style is an integer and valid
        if isinstance(predicted_style, np.ndarray):
            predicted_style = int(np.argmax(predicted_style))  # Convert from ndarray to integer

        if predicted_style not in self.style_colors:
            print(f"Invalid predicted_style: {predicted_style}. Defaulting to 'Null'")
            predicted_style = 0  # Default to 'Null' style

        self.history_acc.extend(acc_data)
        self.history_gyro.extend(gyro_data)

        acc_hist = np.array(self.history_acc)
        gyro_hist = np.array(self.history_gyro)

        # Clear all individual and magnitude plots
        for row in self.ax[:2]:
            for ax in row:
                ax.clear()
                ax.grid(True)

        style_color = self.style_colors[predicted_style]

        # Plot accelerometer magnitude
        self.ax[0][3].plot(acc_hist[:, 0], label='Acc X', alpha=0.6)
        self.ax[0][3].plot(acc_hist[:, 1], label='Acc Y', alpha=0.6)
        self.ax[0][3].plot(acc_hist[:, 2], label='Acc Z', alpha=0.6)
        self.ax[0][3].set_title(f"Accelerometer Axes (Style: {label_names_abb[predicted_style]})")

        # Plot gyroscope magnitude
        self.ax[1][3].plot(gyro_hist[:, 0], label='Gyro X', alpha=0.6)
        self.ax[1][3].plot(gyro_hist[:, 1], label='Gyro Y', alpha=0.6)
        self.ax[1][3].plot(gyro_hist[:, 2], label='Gyro Z', alpha=0.6)
        self.ax[1][3].set_title("Gyroscope Axes")

        self.ax[0][3].legend(loc="upper right")
        self.ax[1][3].legend(loc="upper right")

        # Update stroke count bar chart
        x = np.arange(len(label_names_abb))
        counts = [self.stroke_counts[label] for label in label_names_abb]
        bars = self.ax[2].bar(x, counts, color=[self.style_colors[i] for i in range(len(label_names_abb))])

        for bar, count in zip(bars, counts):
            height = bar.get_height()
            self.ax[2].text(bar.get_x() + bar.get_width() / 2., height, f'{int(count)}', ha='center', va='bottom')

        self.ax[2].set_title('Cumulative Stroke Counts')
        self.ax[2].set_xticks(x)
        self.ax[2].set_xticklabels(label_names_abb)

        plt.pause(0.01)

    def close_plot(self):
        plt.ioff()
        plt.close(self.fig)

def debug_synchronization(signal1, signal2, label1="Signal 1 (Reference)", label2="Signal 2 (Original)", predicted_style=None):
    """
    Debug synchronization by plotting signals before and after alignment.

    Parameters:
    -----------
    signal1 : ndarray
        Reference signal (e.g., accelerometer magnitude).
    signal2 : ndarray
        Signal to align with the reference (e.g., gyroscope magnitude).
    label1 : str
        Label for the reference signal.
    label2 : str
        Label for the original signal.
    """
    aligned_signal2 = synchronize_signals_dtw(signal1, signal2, predicted_style)

    plt.figure(figsize=(12, 8))
    plt.plot(signal1, label=label1, alpha=0.7)
    plt.plot(signal2, label=label2, alpha=0.7)
    plt.plot(aligned_signal2, label=f"{label2} (Aligned)", alpha=0.7, linestyle="--")
    plt.legend()
    plt.title("Signal Synchronization Debug")
    plt.grid(True)
    plt.show()


def extract_window_data_and_predictions(swimming_data, model, data_parameters, user, rec):
    """
    Extract window data, predictions, and sensor values.

    Parameters:
    -----------
    swimming_data : LearningData object
        The loaded swimming data.
    model : tensorflow.keras.Model
        The trained CNN model.
    user : str
        User identifier.
    rec : str
        Recording identifier.

    Returns:
    --------
    dict containing window information.
    """
    # Get window starts and stops
    win_starts, win_stops = swimming_data.window_locs['original'][user][rec]

    # Prepare arrays
    num_windows = len(win_starts)
    windows = np.zeros((num_windows, swimming_data.win_len, len(swimming_data.data_columns))) #initialize sensor_windows 
    dup_windows = np.zeros((num_windows, swimming_data.win_len, len(swimming_data.columns)))  #initialize copy of data set 

    sensor_windows = np.zeros_like(dup_windows)
    #dup_windows = np.zeros_like(dup_windows)

    # Process each window
    for i, (start, stop) in enumerate(zip(win_starts, win_stops)):
        # Raw sensor data
        raw_data = swimming_data.data_dict['original'][user][rec][swimming_data.data_columns].iloc[start:stop+1].values
        dup_data = swimming_data.data_dict['original'][user][rec][swimming_data.columns].iloc[start:stop+1].values

        sensor_windows[i] = dup_data #add timestamps + sensor data + label columns + row index
        # Normalized data
        windows[i] = swimming_data.normalize_window(raw_data, norm_type=data_parameters['window_normalization'])

    # Reshape and predict
    reshaped_windows = windows[..., np.newaxis]
    predictions = model.predict(reshaped_windows, verbose=0)

    return {
        'recording': rec,
        'normalized_windows': windows,
        'raw_windows': sensor_windows,
        'predicted_windows': predictions
    }


def calculate_magnitude(data):
    """Calculate the magnitude from the sensor data."""
    return np.sqrt(np.sum(data**2, axis=1))

def smooth_data_with_savgol(data, window_size=5, poly_order=2):
    """Smooth the data using a Savitzky-Golay filter."""
    window_size = min(window_size, len(data)) if window_size % 2 == 1 else max(3, window_size + 1)
    return scipy.signal.savgol_filter(data, window_length=window_size, polyorder=poly_order)

def butter_highpass_filter(data, cutoff, fs, order=4):
    """Apply a high-pass Butterworth filter to remove baseline drift."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = scipy.signal.butter(order, normal_cutoff, btype='high', analog=False)
    return scipy.signal.filtfilt(b, a, data)

def custom_distance_function(signal1, signal2, style=None):
    """
    Compute a custom distance between two signals based on swimming style.
    
    Parameters:
    -----------
    signal1 : ndarray
        First input signal.
    signal2 : ndarray
        Second input signal.
    style : int, optional
        Swimming style identifier for style-specific distance calculation.
    
    Returns:
    --------
    float
        Computed distance between the signals.
    """
    # Ensure inputs are numpy arrays
    u = np.asarray(signal1)
    v = np.asarray(signal2)

    # Debugging: Print types and contents
    print(f"Type of u: {type(u)}, Type of v: {type(v)}")
    print(f"Contents of u: {u}, Contents of v: {v}")

    # Check for empty signals
    if len(u) == 0 or len(v) == 0:
        raise ValueError("Input signals must not be empty.")

    # Check for NaN or Inf values
    if np.any(np.isnan(u)) or np.any(np.isnan(v)):
        raise ValueError("Input signals contain NaN values.")
    if np.any(np.isinf(u)) or np.any(np.isinf(v)):
        raise ValueError("Input signals contain infinite values.")

    # Style-specific distance calculations
    if style == 1:  # Freestyle
        # Weighted Euclidean distance with emphasis on overall pattern
        distance = np.sqrt(np.mean((u - v) ** 2))
    
    elif style == 2:  # Breaststroke
        # Manhattan distance with peak detection sensitivity
        distance = np.sum(np.abs(u - v))
    
    elif style == 3:  # Backstroke
        # Combination of Euclidean and correlation-based distance
        if len(u) < 2 or len(v) < 2:
            return np.sqrt(np.mean((u - v) ** 2))  # Fallback to Euclidean if not enough data
        euclidean_dist = np.sqrt(np.mean((u - v) ** 2))
        correlation = np.corrcoef(u, v)[0, 1] if np.std(u) > 0 and np.std(v) > 0 else 0
        distance = 0.7 * euclidean_dist + 0.3 * (1 - correlation)
    
    elif style == 4:  # Butterfly
        # Enhanced distance metric with peak and amplitude sensitivity
        peak_sensitivity = np.abs(np.max(u) - np.max(v)) + np.abs(np.min(u) - np.min(v))
        base_distance = np.sqrt(np.mean((u - v) ** 2))
        distance = base_distance + 0.2 * peak_sensitivity
    
    else:
        # Default to standard Euclidean distance
        distance = np.sqrt(np.mean((u - v) ** 2))
    
    return distance

def custom_euclidean(u, v):
    """
    Custom Euclidean distance function that handles both scalars and arrays.
    
    Parameters:
    -----------
    u : float or ndarray
        First input value or array.
    v : float or ndarray
        Second input value or array.
    
    Returns:
    --------
    float
        Computed Euclidean distance.
    """
    # Ensure inputs are numpy arrays
    u = np.asarray(u)
    v = np.asarray(v)
    
    # Check if inputs are scalars
    if u.ndim == 0 and v.ndim == 0:
        return abs(u - v)  # Return absolute difference for scalars
    else:
        return scipy_euclidean(u, v)  # Use scipy's Euclidean for arrays

def synchronize_signals_dtw(signal1, signal2, style=None, config=None):
    """
    Synchronize two signals using Dynamic Time Warping (DTW) with style-specific distance.
    
    Parameters:
    -----------
    signal1 : ndarray
        Reference signal (e.g., accelerometer magnitude).
    signal2 : ndarray
        Signal to align with the reference (e.g., gyroscope magnitude).
    style : int, optional
        Swimming style identifier for style-specific tuning.
    
    Returns:
    --------
    aligned_signal2 : ndarray
        Aligned signal2.
    """

    # Ensure signals are NumPy arrays and flatten them to 1-D
    signal1 = np.asarray(signal1).ravel()
    signal2 = np.asarray(signal2).ravel()

    # Check for NaN or Inf values
    if np.any(np.isnan(signal1)) or np.any(np.isnan(signal2)):
        raise ValueError("Input signals contain NaN values.")
    if np.any(np.isinf(signal1)) or np.any(np.isinf(signal2)):
        raise ValueError("Input signals contain infinite values.")

    # Determine window size and distance function based on style
    if style == 4:  # Butterfly
        window_size = 10  # Smaller window for more precise alignment
    elif style == 2:  # Breaststroke
        window_size = 15  # Moderate window size
    else:
        window_size = 20  # Default window size

    # Create a partial function with the style parameter
    def style_distance(u, v):
        return custom_distance_function(u, v, style)

    # Compute the DTW distance and path with a Sakoe-Chiba band
    distance, path = fastdtw(signal1, signal2, dist=custom_euclidean, radius=window_size)

    # Create an aligned version of signal2 based on the DTW path
    aligned_signal2 = np.zeros_like(signal1)
    
    for i, j in path:
        aligned_signal2[i] = signal2[j]

    # Optional: Print debugging information
    if config and config.DEBUG_SYNCHRONIZATION:
        print(f"Style {style} DTW Synchronization:")
        print(f"Window Size: {window_size}")
        print(f"Total DTW Distance: {distance}")

    return aligned_signal2



def preprocess_signal(signal, cutoff=0.3, fs=30, savgol_window=5, savgol_poly=3):
    def butter_highpass_filter(data, cutoff, fs, order=4):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = scipy.signal.butter(order, normal_cutoff, btype='high', analog=False)
        return scipy.signal.filtfilt(b, a, data)

    def smooth_data_with_savgol(data, window_size=5, poly_order=2):
        window_size = min(window_size, len(data)) if window_size % 2 == 1 else max(3, window_size + 1)
        return scipy.signal.savgol_filter(data, window_length=window_size, polyorder=poly_order)

    filtered_signal = butter_highpass_filter(signal, cutoff, fs)
    smoothed_signal = smooth_data_with_savgol(filtered_signal, savgol_window, savgol_poly)
    return smoothed_signal

def preprocess_sensor_data(window, predicted_style, window_index=None, config=None):
    """
    Preprocess accelerometer and gyroscope data for peak detection, 
    including style-specific preprocessing logic and visualization.

    Parameters:
    -----------
    window : ndarray
        Sensor data window containing accelerometer and gyroscope data.
    predicted_style : int
        Predicted swimming style (integer label).
    debug : bool
        Whether to visualize debug plots for synchronization.
    window_index : int or None
        Current window index for labeling in debug plots.

    Returns:
    --------
    dict
        Processed sensor data including synchronized style-specific metrics.
    """
    acc_data = window[:, :3]
    gyro_data = window[:, 3:6]
    """
    # Apply filtering to each axis first
    filtered_x = butter_highpass_filter(gyro_data[:, 0], 0.3, 30)
    filtered_y = butter_highpass_filter(gyro_data[:, 1], 0.3, 30)
    filtered_z = butter_highpass_filter(gyro_data[:, 2], 0.3, 30)

    # Calculate magnitude from filtered axes
    magnitude_before = np.sqrt(filtered_x**2 + filtered_y**2 + filtered_z**2)
    #  Calculate magnitude from raw axes
    magnitude_after = np.sqrt(gyro_data[:, 0]**2 + gyro_data[:, 1]**2 + gyro_data[:, 2]**2)

    # Apply filtering to the magnitude signal
    filtered_magnitude_after = butter_highpass_filter(magnitude_after, 0.3, 30)

    plt.figure(figsize=(10, 6))
    plt.plot(magnitude_before, label="Filtered Before Magnitude", alpha=0.7)
    plt.plot(filtered_magnitude_after, label="Filtered After Magnitude", alpha=0.7)
    plt.legend()
    plt.title("Comparison of Filtering Before vs After Magnitude")
    plt.show()
    """


    # Calculate raw magnitudes
    acc_magnitude = calculate_magnitude(acc_data)
    gyro_magnitude = calculate_magnitude(gyro_data)

    # Filter and smooth individual axes
    acc_filtered = np.array([preprocess_signal(acc_data[:, j]) for j in range(3)]).T
    gyro_filtered = np.array([preprocess_signal(gyro_data[:, j]) for j in range(3)]).T

    # Filter and smooth magnitudes
    acc_magnitude_filtered = preprocess_signal(acc_magnitude)
    gyro_magnitude_filtered = preprocess_signal(gyro_magnitude)

    # Initialize the dictionary with default magnitudes
    sensor_data = {
        "acc_filtered": acc_filtered,
        "gyro_filtered": gyro_filtered,
        "acc_magnitude_filtered": acc_magnitude_filtered,
        "gyro_magnitude_filtered": gyro_magnitude_filtered,
        
        # Style-specific keys
        "style_acc": None,
        "style_gyro": None,
        "style_acc_peak_key": None,
        "style_gyro_peak_key": None
    }

    # Style-specific adjustments
    if predicted_style == 1:  # Freestyle
        acc_y_neg_filtered = -acc_filtered[:, 1]  #  Y Axis
        gyro_z_pos_filtered = gyro_filtered[:, 2]  # Z axis
        gyro_z_pos_synced = synchronize_signals_dtw(acc_y_neg_filtered, gyro_z_pos_filtered, predicted_style)
        sensor_data.update({
            "style_acc": acc_y_neg_filtered,
            "style_gyro": gyro_z_pos_synced,
            "style_acc_peak_key": "acc_y_negative",
            "style_gyro_peak_key": "gyro_z_positive"
        })

        # Debugging visualization
        if config and config.DEBUG_SYNCHRONIZATION:            
            print(f"Debugging synchronization for Freestyle (Window {window_index})")
            debug_synchronization(
                acc_y_neg_filtered, gyro_z_pos_filtered,
                label1="Acc Y Negative Filtered (Freestyle)", 
                label2="Gyro Z Positive Filtered (Freestyle)", 
                predicted_style=predicted_style
            )

    elif predicted_style == 2:  # Breaststroke
        #acc_yz_magnitude = np.sqrt(acc_filtered[:, 1]**2 + acc_filtered[:, 2]**2)
        #acc_yz_magnitude_filtered = preprocess_signal(acc_yz_magnitude)
        #acc_z_neg_filtered = -acc_filtered[:, 2]  # Negated Z
        gyro_z_neg_filtered = -gyro_filtered[:, 2]  # X axis
        gyro_z_neg_synced = synchronize_signals_dtw(acc_magnitude_filtered, gyro_z_neg_filtered, predicted_style)
        sensor_data.update({
            "style_acc": acc_magnitude_filtered,
            "style_gyro": gyro_z_neg_synced,
            "style_acc_peak_key": "acc_magnitude",
            "style_gyro_peak_key": "gyro_z_negative"
        })

        # Debugging visualization
        if config and config.DEBUG_SYNCHRONIZATION:  
            print(f"Debugging synchronization for Breaststroke (Window {window_index})")
            debug_synchronization(
                acc_magnitude_filtered, gyro_z_neg_filtered,
                label1="Acc Magnitude Filtered (Breaststroke)", 
                label2="Gyro Z Negative Filtered (Breaststroke)",
                predicted_style=predicted_style
            )

    elif predicted_style == 3:  # Backstroke
        acc_z_neg_filtered = -acc_filtered[:, 2]  # Z axis
        gyro_y_neg_filtered = -gyro_filtered[:, 1]  # Y axis
        gyro_y_neg_synced = synchronize_signals_dtw(acc_z_neg_filtered, gyro_y_neg_filtered, predicted_style)
        sensor_data.update({
            "style_acc": acc_z_neg_filtered,
            "style_gyro": gyro_y_neg_synced,
            "style_acc_peak_key": "acc_z_negative",
            "style_gyro_peak_key": "gyro_y_negative"
        })

        # Debugging visualization
        if config and config.DEBUG_SYNCHRONIZATION:  
            print(f"Debugging synchronization for Backstroke (Window {window_index})")
            debug_synchronization(
                acc_z_neg_filtered, gyro_y_neg_filtered,
                label1="Acc Z Negative Filtered (Backstroke)", 
                label2="Gyro Y Negative Filtered (Backstroke)",
                predicted_style=predicted_style
            )

    elif predicted_style == 4:  # Butterfly
        gyro_y_neg_filtered = -gyro_filtered[:, 1]  # Negated Y axis
        gyro_y_neg_synced = synchronize_signals_dtw(acc_magnitude_filtered, gyro_y_neg_filtered, predicted_style)
        sensor_data.update({
            "style_acc": acc_magnitude_filtered,
            "style_gyro": gyro_y_neg_synced,
            "style_acc_peak_key": "acc_magnitude_filtered",
            "style_gyro_peak_key": "gyro_y_negative"
        })

        # Debugging visualization
        if config and config.DEBUG_SYNCHRONIZATION:  
            print(f"Debugging synchronization for Butterfly (Window {window_index})")
            debug_synchronization(
                acc_magnitude_filtered, gyro_y_neg_filtered,
                label1="Acc Magnitude (Butterfly)", 
                label2="Gyro Y Negative Filtered (Butterfly)",
                predicted_style=predicted_style
            )

    return sensor_data


# New function to calculate global prominence
def calculate_global_prominence(windows, base_prominence=0.5, config=None):
    """
    Calculate global prominence based on all windows.

    Parameters:
    -----------
    windows : list of ndarray
        List of all windows containing sensor data.
    base_prominence : float
        Base prominence value for peak detection.

    Returns:
    --------
    float
        Global prominence value.
    """
    all_data = np.concatenate(windows, axis=0)
    global_std = np.std(all_data, ddof=1)
    global_range = np.ptp(all_data)

    # Calculate global prominence
    global_prominence = (
        base_prominence +
        0.01 * np.log1p(global_range) +
        0.05 * np.sqrt(global_std)
    )

    # Bounds
    min_prominence = 0.5
    max_prominence = 3.0
    # Debug output
    if config and config.DEBUG_DETECT_PEAKS:  
        print(f"Global Signal analysis:")
        print(f"  Range: {global_range:.4f}")
        print(f"  Standard Deviation: {global_std:.4f}")
        print(f"  Log(Range + 1): {np.log1p(global_range):.4f}")
        print(f"  Sqrt(STD): {np.sqrt(global_std):.4f}")
        print(f"Calculated global prominence: {global_prominence:.4f}")

    return np.clip(global_prominence, min_prominence, max_prominence)

# Modify the detect_peaks function
def detect_peaks(filtered_sensor_data, predicted_style, base_prominence=0.5, raw_sensor_data=None, distance=5, global_prominence=None, config=None):
    """
    Detect peaks for accelerometer and gyroscope data based on the predicted stroke style.

    Parameters:
    -----------
    sensor_data : dict
        Dictionary containing the processed sensor data.
    predicted_style : int
        Predicted swimming style (integer label).
    base_prominence : float
        Base prominence value for peak detection.
    raw_sensor_data : dict, optional
        Dictionary containing the raw sensor data.
    distance : int
        Minimum distance between peaks.
    global_prominence : float, optional
        Global prominence value calculated over all windows.

    Returns:
    --------
    acc_peaks : dict
        Detected peaks in the accelerometer data.
    gyro_peaks : dict
        Detected peaks in the gyroscope data.
    """

    # Debug print
    if config and config.DEBUG_DETECT_PEAKS: 
        print(f"Detecting peaks for style: {predicted_style}")

    acc_peaks = {}
    gyro_peaks = {}

    def calculate_dynamic_prominence(signal, base_prominence=0.5, style=predicted_style):
        """
        Calculate dynamic prominence with style-specific adjustments.

        Parameters:
        -----------
        signal : ndarray
            The input signal from which to detect peaks.
        base_prominence : float
            The base prominence value for peak detection.
        style : int, optional
            Swimming style identifier for potential style-specific tuning.

        Returns:
        --------
        float
            The calculated prominence for peak detection.
        """
        local_std = np.std(signal, ddof=1)
        signal_range = np.ptp(signal)  # Peak-to-peak range

        # Style-specific adjustments
        if style == 1:  # Freestyle
            range_scale = 0.01
            std_scale = 0.05
        elif style == 2:  # Breaststroke
            range_scale = 0.02
            std_scale = 0.08
        elif style == 3:  # Backstroke
            range_scale = 0.015
            std_scale = 0.07
        elif style == 4:  # Butterfly
            range_scale = 0.025
            std_scale = 0.1
        else:
            range_scale = 0.01
            std_scale = 0.05

        dynamic_prominence = (
            base_prominence +
            (range_scale * np.log1p(signal_range)) +
            (std_scale * np.sqrt(local_std))
        )

        # Bounds
        min_prominence = 0.5
        max_prominence = 3.0

        dynamic_prominence = np.clip(dynamic_prominence, min_prominence, max_prominence)

        # Debug output
        if config and config.DEBUG_DETECT_PEAKS: 
            print(f"Signal analysis:")
            print(f"  Range: {signal_range:.4f}")
            print(f"  Standard Deviation: {local_std:.4f}")
            print(f"  Log(Range + 1): {np.log1p(signal_range):.4f}")
            print(f"  Sqrt(STD): {np.sqrt(local_std):.4f}")
            print(f"Calculated prominence: {dynamic_prominence:.4f}")

        return dynamic_prominence

    # Multi-scale analysis using wavelet transforms
    def wavelet_denoise(signal, wavelet='db1', level=1):
        """
        Perform wavelet denoising with safeguards against division by zero.

        Parameters:
        -----------
        signal : ndarray
            Input signal to be denoised.
        wavelet : str
            Wavelet type to use (e.g., 'db1', 'sym4').
        level : int
            Decomposition level for wavelet transform.

        Returns:
        --------
        denoised_signal : ndarray
            Denoised version of the input signal.
        """
        # Check for signal variability
        if np.std(signal) < 1e-6:
            print("Warning: Input signal has very low variability.")
            return signal  # Return the original signal as it cannot be meaningfully denoised

        coeffs = pywt.wavedec(signal, wavelet, level=level)
        threshold = np.median(np.abs(coeffs[-1])) / 0.6745  # Median absolute deviation (MAD)

        # Safeguard thresholding
        def safe_threshold(c, t):
            magnitude = np.abs(c)
            if np.all(magnitude > 0):
                return pywt.threshold(c, t, mode='soft')
            else:
                if config and config.DEBUG_DETECT_PEAKS: 
                    print("Warning: Zero magnitude encountered in wavelet coefficients.")
                return c  # Return unchanged coefficients for problematic cases
        denoised_coeffs = [safe_threshold(c, threshold) for c in coeffs]
        return pywt.waverec(denoised_coeffs, wavelet)

    # Apply wavelet denoising
    acc_signal =filtered_sensor_data["style_acc"]#wavelet_denoise(filtered_sensor_data["style_acc"], level=1)
    gyro_signal = filtered_sensor_data["style_gyro"]#wavelet_denoise(filtered_sensor_data["style_gyro"], level=1)

    if config and config.DEBUG_PLOT_PEAKS: 
        # Visualize the raw and denoised signals
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(filtered_sensor_data["style_acc"], label='Raw Accelerometer Signal', color='blue', alpha=0.5)
        plt.plot(acc_signal, label='Denoised Accelerometer Signal', color='blue')
        plt.title('Accelerometer Signal (Raw and Denoised)')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(filtered_sensor_data["style_gyro"], label='Raw Gyroscope Signal', color='orange', alpha=0.5)
        plt.plot(gyro_signal, label='Denoised Gyroscope Signal', color='orange')
        plt.title('Gyroscope Signal (Raw and Denoised)')
        plt.legend()

        plt.tight_layout()
        plt.show()

    # Calculate dynamic prominence using raw data if available
    if raw_sensor_data is not None:
        if config and config.DEBUG_DETECT_PEAKS:  print("\nCalculating prominence from raw accelerometer data:")
        acc_prominence = calculate_dynamic_prominence(raw_sensor_data["style_acc"], base_prominence)
        if config and config.DEBUG_DETECT_PEAKS:  print("\nCalculating prominence from raw gyroscope data:")
        gyro_prominence = calculate_dynamic_prominence(raw_sensor_data["style_gyro"], base_prominence)
    else:
        if config and config.DEBUG_DETECT_PEAKS:  print("\nCalculating prominence from processed accelerometer data:")
        acc_prominence = calculate_dynamic_prominence(acc_signal, base_prominence)
        if config and config.DEBUG_DETECT_PEAKS:  print("\nCalculating prominence from processed gyroscope data:")
        gyro_prominence = calculate_dynamic_prominence(gyro_signal, base_prominence)

    # Debugging output for prominence
    if config and config.DEBUG_DETECT_PEAKS:  
        print(f"Calculated Acc Prominence: {acc_prominence}")
        print(f"Calculated Gyro Prominence: {gyro_prominence}")

    # Detect peaks in accelerometer data
    acc_peaks[filtered_sensor_data["style_acc_peak_key"]] = scipy.signal.find_peaks(
        acc_signal,
        prominence=acc_prominence,
        distance=distance
    )[0]

    # Detect peaks in gyroscope data
    gyro_peaks[filtered_sensor_data["style_gyro_peak_key"]] = scipy.signal.find_peaks(
        gyro_signal,
        prominence=gyro_prominence,
        distance=distance
    )[0]

    # Visualization of the detected peaks
    if config and config.DEBUG_PLOT_PEAKS: 
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(acc_signal, label='Accelerometer Signal', color='blue')
        plt.plot(acc_peaks[filtered_sensor_data["style_acc_peak_key"]], acc_signal[acc_peaks[filtered_sensor_data["style_acc_peak_key"]]], "x", label='Detected Peaks', color='red')
        plt.title('Accelerometer Signal with Detected Peaks')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(gyro_signal, label='Gyroscope Signal', color='orange')
        plt.plot(gyro_peaks[filtered_sensor_data["style_gyro_peak_key"]], gyro_signal[gyro_peaks[filtered_sensor_data["style_gyro_peak_key"]]], "x", label='Detected Peaks', color='red')
        plt.title('Gyroscope Signal with Detected Peaks')
        plt.legend()

        plt.tight_layout()
        plt.show()

    # Debug print
    if config and config.DEBUG_DETECT_PEAKS: 
        print("Detected acc_peaks:", acc_peaks)
        print("Detected gyro_peaks:", gyro_peaks)

    return acc_peaks, gyro_peaks

# Add debug flags as a global configuration
class Config:
    def __init__(self):
        self.DEBUG_DETECT_PEAKS = False
        self.DEBUG_PLOT_PEAKS = False
        self.DEBUG_REAL_TIME_PLOTTER = False
        self.DEBUG_SYNCHRONIZATION = False
        self.DEBUG_COUNT_STROKES = False
        self.DEBUG_PBAR = False

def process_recording(args):
    user, rec, swimming_data, model, data_parameters, results_path, total_windows, config, progress_queue, position, output_dir = args
    if config and config.DEBUG_PBAR:
        print(f"[DEBUG] Starting process for {rec} at position {position}")

    # Notify progress queue to create a nested progress bar for this recording
    progress_queue.put({
        "type": "create",
        "desc": f"Processing {rec[:40]:<40}",
        "total": total_windows,
        "position": position
    })

    try:
        # Process the recording and track window progress
        window_data = extract_window_data_and_predictions(
            swimming_data, model, data_parameters, user, rec
        )
        stroke_counts, stroke_labels = count_strokes_by_style(
            window_data['normalized_windows'],
            window_data['raw_windows'],
            window_data['predicted_windows'],
            #base_prominence=0.5,
            user_number=user,
            file_name=rec,
            plot=False,
            config=config,
            window_pbar=lambda: progress_queue.put({"type": "update", "position": position}),
            output_dir=output_dir  # Pass the output directory here
            #data_parameters=data_parameters
        )
    finally:
        if config and config.DEBUG_PBAR:
            print(f"[DEBUG] Closing progress bar for {rec} at position {position}")

        # Mark progress bar as complete
        progress_queue.put({"type": "close", "desc": f"Processing {rec[:40]:<40}", "position": position})

    return {'recording': rec, 'stroke_counts': stroke_counts, 'stroke_labels': stroke_labels}

def reconstruct_original_data(updated_raw_windows, window_length=180, overlap=30):
    """
    Reconstruct the original dataset from overlapping windows
    
    Parameters:
    updated_raw_windows: List of processed windows
    window_length: Length of each window (default 180)
    overlap: Overlap between windows (default 30)
    
    Returns:
    Reconstructed numpy array representing the original dataset
    """
    if not updated_raw_windows:
        return None
    
    # Determine the total length of the reconstructed data
    num_windows = len(updated_raw_windows)
    total_length = (num_windows - 1) * (window_length - overlap) + window_length
    
    # Initialize the reconstructed array with the same number of columns as the first window
    reconstructed_data = np.zeros((total_length, updated_raw_windows[0].shape[1]))
    
    # Reconstruct the data with overlapping
    for idx, window in enumerate(updated_raw_windows):
        start_idx = idx * (window_length - overlap)
        end_idx = start_idx + window_length
        
        # For the first window, simply copy the entire window
        if idx == 0:
            reconstructed_data[start_idx:end_idx, :] = window
        else:
            # For subsequent windows, blend the overlapping regions
            overlap_region = reconstructed_data[start_idx:start_idx+overlap, :]
            new_region = window[:overlap, :]
            
            # Simple averaging in the overlap region
            blended_overlap = (overlap_region + new_region) / 2
            reconstructed_data[start_idx:start_idx+overlap, :] = blended_overlap
            
            # Copy the non-overlapping part of the new window
            reconstructed_data[start_idx+overlap:end_idx, :] = window[overlap:, :]
    
    return reconstructed_data
    
def process_and_reconstruct(raw_windows, stroke_labels, window_length=180, overlap=30):
    """
    Process raw windows and reconstruct the original dataset using row indices,
    handling overlaps by blending overlapping regions.

    Parameters:
    raw_windows: List of raw windows with timestamps and sensor data.
    stroke_labels: Global stroke labels aligned with the dataset.
    window_length: Length of each window (default: 180 samples).
    overlap: Overlap between consecutive windows (default: 30 samples).

    Returns:
    Reconstructed dataset with stroke counts integrated.
    """
    # Check if raw_windows is empty or None
    if raw_windows is None or len(raw_windows) == 0:
        return None

    # Determine the total number of samples in the original dataset
    total_samples = int(max(window[-1, 0] for window in raw_windows)) + 1  # Assuming row indices are sequential

    # Initialize reconstructed data with stroke count column
    num_features = raw_windows[0].shape[1]  # Number of features in the raw window (excluding stroke count)
    reconstructed_data = np.zeros((total_samples, num_features + 1))  # +1 for the stroke_count column
    contribution_count = np.zeros(total_samples)  # Tracks the number of contributions per row for blending

    for idx, raw_window in enumerate(raw_windows):
        # Extract row indices for the current window
        row_indices = raw_window[:, 0].astype(int)

        # Extract the corresponding stroke labels for the current window
        stroke_count_column = stroke_labels[row_indices].reshape(-1, 1)

        # Add stroke_count column to the current window
        updated_window = np.hstack((raw_window, stroke_count_column))

        # Update reconstructed data row by row
        for row_idx, row_data in zip(row_indices, updated_window):
            if contribution_count[row_idx] > 0:
                # Blend overlapping rows by averaging
                reconstructed_data[row_idx, :-1] = (
                    reconstructed_data[row_idx, :-1] * contribution_count[row_idx] + row_data[:-1]
                ) / (contribution_count[row_idx] + 1)
                reconstructed_data[row_idx, -1] = (
                    reconstructed_data[row_idx, -1] * contribution_count[row_idx] + row_data[-1]
                ) / (contribution_count[row_idx] + 1)
            else:
                # Add non-overlapping rows directly
                reconstructed_data[row_idx] = row_data

            # Increment contribution count for this row
            contribution_count[row_idx] += 1

    return reconstructed_data



def save_updated_data(reconstructed_data, user, file_name, output_dir, data_parameters, config):
    """
    Save the updated data with stroke counts to a CSV file.
    """
    # Path to where we want to save the training results
    sub_dir_path = os.path.join(output_dir, user)

    # Ensure the output directory exists
    os.makedirs(sub_dir_path, exist_ok=True)  # Create the user-specific directory

    # Define the columns
    sensor_columns = data_parameters['data_columns']  # Use provided sensor columns
    full_columns = ["row_index", "timestamp"] + sensor_columns + ["label", "stroke_count"]  # Include stroke_count

    # Convert reconstructed_data to DataFrame
    updated_df = pd.DataFrame(reconstructed_data, columns=full_columns)

    # Debugging: Display the first few rows of the updated DataFrame
    if config.DEBUG_COUNT_STROKES:
        print(f"[DEBUG] Updated DataFrame preview:\n{updated_df.head()}")

    # Save the DataFrame
    base_name, _ = os.path.splitext(file_name)  # Split into ('file_name', '.csv')
    new_file_name = f"{base_name}_updated.csv"
    save_path = os.path.join(sub_dir_path, new_file_name)
    updated_df.to_csv(save_path, index=False)
    print(f"Updated data saved to: {save_path}")


def count_strokes_by_style(normalized_windows, raw_windows, predicted_styles, 
                           base_prominence=0.5, user_number=None, file_name=None, 
                           plot=None, config=None, window_pbar=None, output_dir=None, data_parameters=None):
    """
    Count strokes using style-specific peak detection logic.
    """
    # Initialize stroke counts dictionary for each style
    stroke_counts = {label: 0 for label in label_names_abb}

    # Initialize a real-time plotter if enabled
    plotter = RealTimePlotter(user_number=user_number, file_name=file_name) if plot else None

    # Initialize global trackers for peaks and stroke labels
    global_acc_peaks = set()
    global_gyro_peaks = set()
    stroke_labels_indices = set()  # To avoid duplicate labeling

    # Initialize stroke_labels array for the entire dataset
    num_samples = (len(raw_windows) - 1) * 30 + 180  # Account for window overlap
    stroke_labels = np.zeros(num_samples)

    # Calculate global prominence for peak detection
    #global_prominence = calculate_global_prominence(normalized_windows, base_prominence, config=config)

    # Iterate through all windows
    for i, (normalized_window, raw_window) in enumerate(zip(normalized_windows, raw_windows)):
        # Determine the predicted style and its confidence
        predicted_style = np.argmax(predicted_styles[i])
        style_confidence = np.max(predicted_styles[i])

        # Skip windows with low confidence or null style (label 0)
        if predicted_style in [0, 5, 6] or style_confidence < 0.8:
            start_idx = i * 30
            end_idx = start_idx + 180
            if config.DEBUG_COUNT_STROKES:
                print(f"[DEBUG] Window {i} (Skipped): label = {predicted_style} stroke_labels[{start_idx}:{end_idx}] = {stroke_labels[start_idx:end_idx]}")
            if callable(window_pbar):
                window_pbar()
            continue

        # Preprocess sensor data for peak detection
        filtered_sensor_data = preprocess_sensor_data(normalized_window, predicted_style, window_index=i)

        # Extract raw sensor data
        raw_sensor_data = {
            "style_acc": raw_window[:, 2:5],  # ACC_0, ACC_1, ACC_2
            "style_gyro": raw_window[:, 5:8],  # GYRO_0, GYRO_1, GYRO_2
            "style_acc_peak_key": "raw_acc_magnitude",
            "style_gyro_peak_key": "raw_gyro_magnitude"
        }

        # Get style-specific peak detection parameters
        style_params = stroke_axis_params[predicted_style]
        prominence = style_params["prominence"]
        distance = style_params["distance"]

        # Perform peak detection
        acc_peaks, gyro_peaks = detect_peaks(
            filtered_sensor_data, 
            predicted_style,
            base_prominence=prominence,
            raw_sensor_data=raw_sensor_data,
            distance=distance,
            #global_prominence=global_prominence,
            config=config
        )
        # Debugging output
        if config.DEBUG_COUNT_STROKES:
            print(f"Window {i}: Detected Acc Peaks: {acc_peaks}, Gyro Peaks: {gyro_peaks}")

        valid_strokes = 0
        # Access the correct keys based on the predicted style
        acc_peak_indices = acc_peaks.get(filtered_sensor_data["style_acc_peak_key"], [])
        gyro_peak_indices = gyro_peaks.get(filtered_sensor_data["style_gyro_peak_key"], [])

        if config.DEBUG_COUNT_STROKES:
            print(f"Processing Style: {label_names_abb[predicted_style]}")
            print(f"Acc Peaks: {acc_peak_indices}")
            print(f"Gyro Peaks: {gyro_peak_indices}")

        # Match accelerometer and gyroscope peaks

        for acc_peak in acc_peak_indices:
            # Check to be sure peaks aren't in style 0, 5, 6 as some windows will have transition from rest(0), turn(5) or kick(6)
            style_label = raw_window[acc_peak, 8]
            # Check if the label is NaN
            if np.isnan(style_label):
                style_label = 0 
            if int(style_label) in [0, 5, 6]:
                continue 
            global_acc_peak = int(raw_window[acc_peak, 0])# global_row_index + acc_peak
            for gyro_peak in gyro_peak_indices:
                global_gyro_peak = int(raw_window[gyro_peak, 0])#global_row_index + gyro_peak

                # Synchronization tolerance check
                if abs(acc_peak - gyro_peak) <= 3:
                    if global_acc_peak not in global_acc_peaks and global_gyro_peak not in global_gyro_peaks:
                        valid_strokes += 1
                        global_acc_peaks.add(global_acc_peak)
                        global_gyro_peaks.add(global_gyro_peak)
                        stroke_labels[global_acc_peak] = 1  # Label the stroke
                        stroke_labels_indices.add(global_acc_peak)   
                        break              

        # Update stroke counts
        stroke_counts[label_names_abb[predicted_style]] += valid_strokes

        # Debugging output for strokes and labels
        start_idx = i * 30
        end_idx = start_idx + 180
        if config.DEBUG_COUNT_STROKES:
            print(f"[DEBUG] Window {i}: stroke_labels[{start_idx}:{end_idx}] = {stroke_labels[start_idx:end_idx]}")
            print(f"[DEBUG] Window {i}, Style: {label_names_abb[predicted_style]}, Valid Strokes: {valid_strokes}, Stoke Label Indices: {sorted(stroke_labels_indices)} numlabels: {len(stroke_labels_indices)}")

        # Update progress bar if applicable
        if callable(window_pbar):
            window_pbar()

        if config.DEBUG_COUNT_STROKES:
            print(f"[DEBUG] Window {i}, Style: {label_names_abb[predicted_style]}, "
                  f"Valid Strokes: {valid_strokes}, Acc Peaks: {acc_peak_indices}, Gyro Peaks: {gyro_peak_indices}")


        if config.DEBUG_REAL_TIME_PLOTTER:
            plotter.stroke_counts[label_names_abb[predicted_style]] += valid_strokes

        # Optional visualization
        if config.DEBUG_COUNT_STROKES_PLOTTER:
            plot_data_with_peaks(
                filtered_sensor_data,
                acc_peaks,
                gyro_peaks,
                predicted_style,
                window_index=i,
                user_number=user_number,
                file_name=file_name
            )

        if config.DEBUG_REAL_TIME_PLOTTER:
            plotter.update_plot(
                filtered_sensor_data["acc_filtered"],
                filtered_sensor_data["gyro_filtered"],
                filtered_sensor_data["acc_magnitude_filtered"],
                filtered_sensor_data["gyro_magnitude_filtered"],
                acc_peaks,
                gyro_peaks,
                predicted_style,
                style_confidence,
                i
            )

        if config.DEBUG_COUNT_STROKES:
            print(f"Window {i}: stroke_labels[{start_idx}:{end_idx}] = {stroke_labels[start_idx:end_idx]}")

    if config.DEBUG_REAL_TIME_PLOTTER:
        plotter.close_plot()


    """
    # Handle cumulative stroke labeling for overlapping regions
    updated_raw_windows = []
    for idx, raw_window in enumerate(raw_windows):
        start_idx = idx * 30
        end_idx = start_idx + 180
        stroke_count_column = stroke_labels[start_idx:end_idx].reshape(-1, 1)  # Extract relevant stroke counts
        updated_window = np.hstack((raw_window, stroke_count_column))
        updated_raw_windows.append(updated_window)

    
    reconstruced_dataset = process_and_reconstruct(raw_windows=raw_windows, stroke_labels=stroke_labels)
    # Save updated raw windows with stroke counts
    save_updated_data(
        np.array(reconstruced_dataset),
        user=str(user_number),
        file_name=file_name,
        output_dir=output_dir,
        data_parameters=data_parameters,
        config=config
    )
    """
    # Final debugging output for stroke_labels
    if config.DEBUG_COUNT_STROKES:
        print(f"[DEBUG] Final stroke_labels array:\n{stroke_labels}")

    # Return stroke_counts and stroke_labels
    return stroke_counts, stroke_labels

def monitor_progress(queue, total_tasks, config):
    """
    Monitor the progress queue and manage progress bars for multiple recordings.
    """
    progress_bars = {}  # Track active progress bars
    active_files = set()  # Track files being processed to avoid duplicates

    # Global progress bar
    global_bar = tqdm(
        total=total_tasks,
        desc="Global Progress",
        position=0,
        ncols=100,
        leave=True,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
        colour='cyan'
    )

    while True:
        message = queue.get()
        if config and config.DEBUG_PBAR:
            print(f"[DEBUG] Monitor received message: {message}")

        if message is None:  # Stop signal
            break

        msg_type = message.get("type")
        position = message.get("position")
        desc = message.get("desc")

        if msg_type == "create":
            # Create a progress bar if not already active for the file
            if desc not in active_files:
                if config and config.DEBUG_PBAR:
                    print(f"[DEBUG] Creating progress bar for {desc} at position {position}")
                progress_bars[position] = tqdm(
                    total=message["total"],
                    desc=desc,
                    position=position + 1,  # Offset from global progress bar
                    ncols=100,
                    leave=True,
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
                )
                active_files.add(desc)
            else:
                if config and config.DEBUG_PBAR:
                    print(f"[WARNING] Progress bar for {desc} already exists at position {position}!")

        elif msg_type == "update":
            # Update the progress bar if it exists
            if position in progress_bars:
                progress_bars[position].update(1)
                global_bar.update(1)  # Update global progress
            else:
                if config and config.DEBUG_PBAR:
                    print(f"[WARNING] Update received for non-existent progress bar at position {position}")

        elif msg_type == "close":
            # Close the progress bar and remove it
            if position in progress_bars:
                if config and config.DEBUG_PBAR:
                    print(f"[DEBUG] Closing progress bar for {desc} at position {position}")
                progress_bars[position].close()
                del progress_bars[position]
                active_files.discard(desc)  # Mark file as no longer active

        elif msg_type == "user_complete":
            # Update global progress for user completion
            global_bar.update(1)

    # Ensure all progress bars are closed before exiting
    global_bar.close()
    for bar in progress_bars.values():
        bar.close()



def main():
    # Global debug flags
    config = Config()
    config.DEBUG_DETECT_PEAKS = False
    config.DEBUG_PLOT_PEAKS = False
    config.DEBUG_REAL_TIME_PLOTTER = False
    config.DEBUG_SYNCHRONIZATION = False
    config.DEBUG_COUNT_STROKES = False
    config.DEBUG_COUNT_STROKES_PLOTTER = False 
    config.DEBUG_PBAR = False

    # Setup paths and data
    data_path = '/Users/juanloya/Documents/SwimmingModelPython/swim_v2/data_copy/processed_30Hz_relabeled'
    results_path = '/Users/juanloya/Documents/SwimmingModelPython/swim_v2/tutorial_save_path_epoch60_mag/'
    # Define the output directory for updated files
    output_dir = '/Users/juanloya/Documents/SwimmingModelPython/swim_v2/data_modified'
    # Load users from data parameters
    with open(os.path.join(results_path, '1', 'data_parameters.pkl'), 'rb') as f:
        data_parameters = pickle.load(f)[0]
    users = data_parameters['users']

    # Calculate total number of tasks (all recordings for all users)
    total_tasks = 0
    for user in users:
        # Temporarily load data to count recordings
        temp_swimming_data = learning_data.LearningData()
        temp_swimming_data.load_data(
            data_path=data_path,
            data_columns=data_parameters['data_columns'],
            users=[user],
            labels=data_parameters['labels']
        )
        total_tasks += len(temp_swimming_data.data_dict['original'][user].keys())

    all_user_results = {}

    # Shared progress queue for all users and recordings
    manager = multiprocessing.Manager()
    progress_queue = manager.Queue()

    # Start the monitor process
    monitor_process = multiprocessing.Process(
        target=monitor_progress,
        args=(progress_queue, total_tasks, config)  # Include user-level progress updates
    )
    monitor_process.start()

    # Main progress bar
    with tqdm(
        total=len(users),
        desc="Processing Users",
        position=0,
        ncols=100,
        colour='green',
        leave=True,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
    ) as user_pbar:
        for user in users:
            print(f"Processing user: {user}")

            # Prepare swimming data for the user
            swimming_data = learning_data.LearningData()
            swimming_data.load_data(
                data_path=data_path,
                data_columns=data_parameters['data_columns'],
                users=[user],
                labels=data_parameters['labels']
            )

            # Combine labels
            for label in data_parameters['combine_labels'].keys():
                swimming_data.combine_labels(
                    labels=data_parameters['combine_labels'][label],
                    new_label=label
                )

            # Create sliding windows
            swimming_data.sliding_window_locs(
                win_len=data_parameters['win_len'],
                slide_len=data_parameters['slide_len']
            )

            # Load user model
            model = tf.keras.models.load_model(
                os.path.join(results_path, user, 'model_best.keras'),
                compile=False
            )

            # Get recordings to process
            recordings = list(swimming_data.data_dict['original'][user].keys())

            # Prepare multiprocessing arguments for recordings
            mp_args = [
                (
                    user, rec, swimming_data, model, data_parameters, results_path,
                    len(swimming_data.window_locs['original'][user][rec][0]),
                    config, progress_queue, idx + 1, output_dir
                )
                for idx, rec in enumerate(recordings)
            ]

            # Use multiprocessing to process recordings
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                user_results = pool.map(process_recording, mp_args)

            # Process results
            #all_user_results[user] = {
            #    result['recording']: result['stroke_counts']
            #    for result in user_results
            #}
            # Process and save results
            for result in user_results:
                recording = result['recording']
                stroke_counts = result['stroke_counts']
                stroke_labels = result['stroke_labels']

                # Save stroke counts
                if user not in all_user_results:
                    all_user_results[user] = {}
                all_user_results[user][recording] = stroke_counts

                # Path to where we want to save the training results
                sub_dir_path = os.path.join(output_dir, user)
                base_name, _ = os.path.splitext(recording)  # Split into ('file_name', '.csv')
                new_file_name = f"{base_name}_updated.csv"

                # Ensure the output directory exists
                os.makedirs(sub_dir_path, exist_ok=True)  # Create the user-specific directory

                # Check if stroke_labels need padding to match the dataset
                dataset_length = swimming_data.data_dict['original'][user][recording].shape[0]
                if len(stroke_labels) < dataset_length:
                    padding_length = dataset_length - len(stroke_labels)
                    stroke_labels = np.pad(stroke_labels, (0, padding_length), 'constant', constant_values=0)

                # Update stroke labels in swimming_data
                swimming_data.data_dict['original'][user][recording]['stroke_labels'] = stroke_labels

                # Save the updated dataset with stroke labels


                updated_df = swimming_data.data_dict['original'][user][recording]
                updated_df.to_csv(
                    os.path.join(sub_dir_path, new_file_name),
                    index=False
                )
                print(f"Updated data saved to: {sub_dir_path}")

            # Save user results
            user_save_path = os.path.join(results_path, user, 'stroke_counts_results.pkl')
            with open(user_save_path, 'wb') as f:
                pickle.dump(all_user_results[user], f)

            # Update user progress bar
            user_pbar.update(1)

    # Stop the monitor process
    progress_queue.put(None)
    monitor_process.join()

    print("\nProcessing complete. Results saved for all users.")

    # Print results summary
    for user, recordings in all_user_results.items():
        print(f"Results for User {user}:")
        for rec, counts in recordings.items():
            print(f"  Recording: {rec}")
            print(f"  Stroke Counts by Style: {counts}")


if __name__ == '__main__':
    main()
