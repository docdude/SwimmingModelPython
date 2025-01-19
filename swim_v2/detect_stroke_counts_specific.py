import os
import pickle
import numpy as np
import tensorflow as tf
import learning_data
import utils
import scipy.signal  # For peak detection
import matplotlib.pyplot as plt
from collections import deque
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean as scipy_euclidean

import pywt

# Define label names for the stroke styles
label_names_abb = ['Null', 'Fr', 'Br', 'Ba', 'Bu']

# Stroke-style-specific axis and parameter mapping
stroke_axis_params = {
    1: {"acc_axes": [0, 1, 2], "gyro_axes": [0, 1, 2], "prominence": 1.5, "distance": 35},  # Freestyle
    2: {"acc_axes": [0, 1, 2], "gyro_axes": [0, 1, 2], "prominence": 1.7, "distance": 50}, # Breaststroke
    3: {"acc_axes": [0, 1, 2], "gyro_axes": [0, 1, 2], "prominence": 0.5, "distance": 45},  # Backstroke
    4: {"acc_axes": [0, 1, 2], "gyro_axes": [0, 1, 2], "prominence": 0.5, "distance": 35}, # Butterfly
}


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
    Extract window data, predictions, and sensor values
    
    Parameters:
    -----------
    swimming_data : LearningData object
        The loaded swimming data
    model : tensorflow.keras.Model
        The trained CNN model
    user : str
        User identifier
    rec : str
        Recording identifier
    
    Returns:
    --------
    dict containing window information
    """
    # Get window starts and stops
    win_starts = swimming_data.window_locs['original'][user][rec][0]
    win_stops = swimming_data.window_locs['original'][user][rec][1]
    
    # Prepare arrays to store data
    windows = np.zeros((len(win_starts), swimming_data.win_len, len(swimming_data.data_columns)))
    sensor_windows = np.zeros((len(win_starts), swimming_data.win_len, len(swimming_data.data_columns)))
    y_true_windows = np.zeros((len(windows), 5))
    y_true_windows_maj = np.zeros(len(windows))
    
    # Process each window
    for iii in range(len(win_starts)):
        win_start = win_starts[iii]
        win_stop = win_stops[iii]
        
        # Get raw sensor data for the window
        sensor_window = swimming_data.data_dict['original'][user][rec][swimming_data.data_columns].values[win_start:win_stop+1, :]
        
        # Normalize window
        window_norm = swimming_data.normalize_window(sensor_window, 
                                                     norm_type=data_parameters['window_normalization'])
        
        windows[iii] = window_norm
        sensor_windows[iii] = sensor_window
        
        # Get window labels
        win_labels = swimming_data.data_dict['original'][user][rec]['label'].values[win_start: win_stop + 1]
        win_label_cat, majority_label = swimming_data.get_window_label(win_labels, 
                                                                       label_type='majority',
                                                                       majority_thresh=0.25)
        
        y_true_windows[iii, :] = win_label_cat
        y_true_windows_maj[iii] = majority_label
    
    # Reshape for model prediction
    window_reshape = windows.reshape((windows.shape[0], windows.shape[1], windows.shape[2], 1))
    
    # Predict labels
    y_pred_windows = model.predict(window_reshape)
    # Display results
    print("Predictions shape:", y_pred_windows.shape)
    print("Sample predictions:", y_pred_windows[:5])
    return {
        'normalized_windows': windows, #normalized sensor data
        'raw_windows': sensor_windows, #raw sensor data
        'true_windows_cat': y_true_windows,
        'true_windows_maj': y_true_windows_maj,
        'predicted_windows': y_pred_windows
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

def synchronize_signals(signal1, signal2, window_size=10):
    """
    Synchronize two signals using cross-correlation with windowing.
    :param signal1: Reference signal (e.g., accelerometer magnitude).
    :param signal2: Signal to align with the reference (e.g., gyroscope magnitude).
    :param window_size: Size of the window to consider for cross-correlation.
    :return: Aligned signal2.
    """
    # Ensure signals are zero-centered
    signal1 = signal1 - np.mean(signal1)
    signal2 = signal2 - np.mean(signal2)

    # Calculate cross-correlation
    correlation = np.correlate(signal1, signal2, mode='full')
    lag = np.argmax(correlation) - (len(signal2) - 1)

    # Use windowing to limit the lag adjustment
    if abs(lag) > window_size:
        lag = np.sign(lag) * window_size

    if lag > 0:
        aligned_signal2 = np.pad(signal2, (lag, 0), mode='constant')[:len(signal1)]
    elif lag < 0:
        aligned_signal2 = signal2[-lag:]
        aligned_signal2 = np.pad(aligned_signal2, (0, len(signal1) - len(aligned_signal2)), mode='constant')
    else:
        aligned_signal2 = signal2

    return aligned_signal2

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

def synchronize_signals_dtw(signal1, signal2, style=None, debug=False):
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
    if debug:
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

def preprocess_sensor_data(window, predicted_style, debug=False, window_index=None):
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
        acc_z_pos_filtered = acc_filtered[:, 2]  #  Z Axis
        gyro_z_pos_filtered = gyro_filtered[:, 2]  # Z axis
        gyro_z_pos_synced = synchronize_signals_dtw(acc_z_pos_filtered, gyro_z_pos_filtered, predicted_style, debug)
        sensor_data.update({
            "style_acc": acc_z_pos_filtered,
            "style_gyro": gyro_z_pos_synced,
            "style_acc_peak_key": "acc_z_positive",
            "style_gyro_peak_key": "gyro_z_positive"
        })

        # Debugging visualization
        if debug:
            print(f"Debugging synchronization for Freestyle (Window {window_index})")
            debug_synchronization(
                acc_z_pos_filtered, gyro_z_pos_filtered,
                label1="Acc Y Negative Filtered (Freestyle)", 
                label2="Gyro Z Positive Filtered (Freestyle)", 
                predicted_style=predicted_style
            )

    elif predicted_style == 2:  # Breaststroke
        #acc_yz_magnitude = np.sqrt(acc_filtered[:, 1]**2 + acc_filtered[:, 2]**2)
        #acc_yz_magnitude_filtered = preprocess_signal(acc_yz_magnitude)
        acc_z_neg_filtered = -acc_filtered[:, 2]  # Negated Z
        gyro_x_pos_filtered = gyro_filtered[:, 0]  # X axis
        gyro_x_pos_synced = synchronize_signals_dtw(acc_z_neg_filtered, gyro_x_pos_filtered, predicted_style, debug)
        sensor_data.update({
            "style_acc": acc_z_neg_filtered,
            "style_gyro": gyro_x_pos_synced,
            "style_acc_peak_key": "acc_z_negative",
            "style_gyro_peak_key": "gyro_x_positive"
        })

        # Debugging visualization
        if debug:
            print(f"Debugging synchronization for Breaststroke (Window {window_index})")
            debug_synchronization(
                acc_z_neg_filtered, gyro_x_pos_filtered,
                label1="Acc Z Negative Filtered (Breaststroke)", 
                label2="Gyro X Positive Filtered (Breaststroke)",
                predicted_style=predicted_style
            )

    elif predicted_style == 3:  # Backstroke
        acc_z_neg_filtered = -acc_filtered[:, 2]  # Z axis
        gyro_y_neg_filtered = -gyro_filtered[:, 1]  # Y axis
        gyro_y_neg_synced = synchronize_signals_dtw(acc_z_neg_filtered, gyro_y_neg_filtered, predicted_style, debug)
        sensor_data.update({
            "style_acc": acc_z_neg_filtered,
            "style_gyro": gyro_y_neg_synced,
            "style_acc_peak_key": "acc_z_negative",
            "style_gyro_peak_key": "gyro_y_negative"
        })

        # Debugging visualization
        if debug:
            print(f"Debugging synchronization for Backstroke (Window {window_index})")
            debug_synchronization(
                acc_z_neg_filtered, gyro_y_neg_filtered,
                label1="Acc Z Negative Filtered (Backstroke)", 
                label2="Gyro Y Negative Filtered (Backstroke)",
                predicted_style=predicted_style
            )

    elif predicted_style == 4:  # Butterfly
        gyro_y_neg_filtered = -gyro_filtered[:, 1]  # Negated Y axis
        gyro_y_neg_synced = synchronize_signals_dtw(acc_magnitude_filtered, gyro_y_neg_filtered, predicted_style, debug)
        sensor_data.update({
            "style_acc": acc_magnitude_filtered,
            "style_gyro": gyro_y_neg_synced,
            "style_acc_peak_key": "acc_magnitude_filtered",
            "style_gyro_peak_key": "gyro_y_negative"
        })

        # Debugging visualization
        if debug:
            print(f"Debugging synchronization for Butterfly (Window {window_index})")
            debug_synchronization(
                acc_magnitude_filtered, gyro_y_neg_filtered,
                label1="Acc Magnitude (Butterfly)", 
                label2="Gyro Y Negative Filtered (Butterfly)",
                predicted_style=predicted_style
            )

    return sensor_data


def detect_peaks(filtered_sensor_data, predicted_style, base_prominence=0.5, raw_sensor_data=None, distance=5, debug=False):
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

    Returns:
    --------
    acc_peaks : dict
        Detected peaks in the accelerometer data.
    gyro_peaks : dict
        Detected peaks in the gyroscope data.
    """
    # Debug print
    if debug:
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
        if debug:
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
                print("Warning: Zero magnitude encountered in wavelet coefficients.")
                return c  # Return unchanged coefficients for problematic cases
        denoised_coeffs = [safe_threshold(c, threshold) for c in coeffs]
        return pywt.waverec(denoised_coeffs, wavelet)
    
    # Apply wavelet denoising
    acc_signal = wavelet_denoise(filtered_sensor_data["style_acc"], level=1)
    gyro_signal = wavelet_denoise(filtered_sensor_data["style_gyro"], level=1)

    if not debug: 
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
        if debug: print("\nCalculating prominence from raw accelerometer data:")
        acc_prominence = calculate_dynamic_prominence(raw_sensor_data["style_acc"], base_prominence)
        if debug: print("\nCalculating prominence from raw gyroscope data:")
        gyro_prominence = calculate_dynamic_prominence(raw_sensor_data["style_gyro"], base_prominence)
    else:
        if debug: print("\nCalculating prominence from processed accelerometer data:")
        acc_prominence = calculate_dynamic_prominence(acc_signal, base_prominence)
        if debug: print("\nCalculating prominence from processed gyroscope data:")
        gyro_prominence = calculate_dynamic_prominence(gyro_signal, base_prominence)

    # Debugging output for prominence
    if debug: 
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
    if not debug:
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
    if debug:
        print("Detected acc_peaks:", acc_peaks)
        print("Detected gyro_peaks:", gyro_peaks)
    
    return acc_peaks, gyro_peaks



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


def count_strokes_by_style(normalized_windows, raw_windows, predicted_styles, base_prominence=0.5, user_number=None, file_name=None, plot=False, debug=False):
    """
    Count strokes using style-specific peak detection logic.
    
    Parameters:
    -----------
    normalized_windows : ndarray
        Normalized sensor data windows.
    raw_windows : ndarray
        Raw sensor data windows.
    predicted_styles : ndarray
        Predicted swimming styles for each window.
    base_prominence : float
        Base prominence for peak detection.
    user_number : int, optional
        User identifier for plotting.
    file_name : str, optional
        File name for plotting.
    plot : bool, optional
        Whether to plot the results.
    
    Returns:
    --------
    dict
        Stroke counts for each style.
    """
    stroke_counts = {label: 0 for label in label_names_abb}
    plotter = RealTimePlotter(user_number=user_number, file_name=file_name) if plot else None
    global_acc_peaks = set()
    global_gyro_peaks = set()

    for i, (normalized_window, raw_window) in enumerate(zip(normalized_windows, raw_windows)):
        predicted_style = np.argmax(predicted_styles[i])
        style_confidence = np.max(predicted_styles[i])

        if predicted_style == 0 or style_confidence < 0.8:
            continue

        filtered_sensor_data = preprocess_sensor_data(normalized_window, predicted_style, debug=debug, window_index=i)

        # Use raw data for peak detection
        raw_sensor_data = {
            "style_acc": raw_window[:, :3],  # Assuming first three columns are accelerometer data
            "style_gyro": raw_window[:, 3:],  # Assuming next three columns are gyroscope data
            "style_acc_peak_key": "raw_acc_magnitude",
            "style_gyro_peak_key": "raw_gyro_magnitude"
        }
        # Style specifict peak detection parameters 
        style_params = stroke_axis_params[predicted_style]  
        prominence = style_params["prominence"]
        distance = style_params["distance"]
        # Detect peaks for the current window
        acc_peaks, gyro_peaks = detect_peaks(filtered_sensor_data, 
                                            predicted_style,    
                                            base_prominence=prominence,
                                            raw_sensor_data=raw_sensor_data, 
                                            distance=distance, 
                                            debug=debug)

        # Debugging output
        if debug:
            print(f"Window {i}: Detected Acc Peaks: {acc_peaks}, Gyro Peaks: {gyro_peaks}")

        valid_strokes = 0

        # Access the correct keys based on the predicted style
        acc_peak_indices = acc_peaks.get(filtered_sensor_data["style_acc_peak_key"], [])
        gyro_peak_indices = gyro_peaks.get(filtered_sensor_data["style_gyro_peak_key"], [])

        if debug:
            print(f"Processing Style: {label_names_abb[predicted_style]}")
            print(f"Acc Peaks: {acc_peak_indices}")
            print(f"Gyro Peaks: {gyro_peak_indices}")

        # Match accelerometer and gyroscope peaks
        for acc_peak in acc_peak_indices:
            global_acc_peak = acc_peak + i * 30  # Assuming 30 Hz sampling rate
            for gyro_peak in gyro_peak_indices:
                global_gyro_peak = gyro_peak + i * 30
                if abs(acc_peak - gyro_peak) <= 3:  # Synchronization tolerance
                    if global_acc_peak not in global_acc_peaks and global_gyro_peak not in global_gyro_peaks:
                        valid_strokes += 1
                        global_acc_peaks.add(global_acc_peak)
                        global_gyro_peaks.add(global_gyro_peak)
                        break

        # Update stroke counts
        stroke_counts[label_names_abb[predicted_style]] += valid_strokes
        if plot:
            plotter.stroke_counts[label_names_abb[predicted_style]] += valid_strokes
        
        if debug:
            print(f"Window {i}: Counted {valid_strokes} valid strokes")

        # Optional visualization
        if debug:
            plot_data_with_peaks(
                filtered_sensor_data,
                acc_peaks,
                gyro_peaks,
                predicted_style,
                window_index=i,
                user_number=user_number,
                file_name=file_name
            )

        if plot:
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

    if plot:
        plotter.close_plot()

    return stroke_counts




def main():
    #data_path = '/Users/juanloya/Documents/SwimmingModelPython/swim_v2/SyntheticSwimmingData'
    data_path = '/Users/juanloya/Documents/SwimmingModelPython/swim_v2/data/processed_30Hz_relabeled'

    results_path = '/Users/juanloya/Documents/SwimmingModelPython/swim_v2/tutorial_save_path_epoch60_mag/'
    
    #with open(os.path.join(results_path, '22/data_parameters.pkl'), 'rb') as f:
     #   data_parameters = pickle.load(f)[0]
    

    #users = data_parameters['users']
        # User whose model we want to load
    user = '0'
    experiment_save_path = os.path.join(results_path, user)

    # Get the data parameters used for loading
    with open(os.path.join(results_path, user, 'data_parameters.pkl'), 'rb') as f:
        data_parameters = pickle.load(f)[0] 
    swimming_data = learning_data.LearningData()
    swimming_data.load_data(data_path=data_path, 
                             data_columns=data_parameters['data_columns'],
                             users=[user], 
                             labels=data_parameters['labels'])
    
    for label in data_parameters['combine_labels'].keys():
        swimming_data.combine_labels(labels=data_parameters['combine_labels'][label], 
                                     new_label=label)
    
    swimming_data.sliding_window_locs(win_len=data_parameters['win_len'], 
                                      slide_len=data_parameters['slide_len'])
    
    #all_predictions = {}
    
    #for (i, user) in enumerate(users):
     #   user_number = i + 1  # Assign a user number based on loop index
      #  print(f"Working on User {user_number}: {user}. {i+1} of {len(users)}")
        #print(f"Working on user: {user}. {i+1} of {len(users)}")
    user_number = 1
    print(f"Working on User {user_number}: {user}.")   
    model = tf.keras.models.load_model(os.path.join(results_path, user, 'model_best.keras'), 
                                           compile=False)
        
    user_predictions = {}
        
    for rec in swimming_data.data_dict['original'][user].keys():
        print(f"Processing recording: {rec}")
        file_name = rec  # Use the recording name as the file name
            
        window_data = extract_window_data_and_predictions(swimming_data, model, 
                                                              data_parameters, user, rec)
            
        stroke_counts = count_strokes_by_style(
                window_data['normalized_windows'], 
                window_data['raw_windows'], 
                window_data['predicted_windows'], 
                base_prominence=0.5,
                user_number= user_number,
                file_name=rec,
                plot=False,
                debug=False
            )
            
        user_predictions[rec] = {
            'window_data': window_data,
            'stroke_counts': stroke_counts
        }
    with open(os.path.join(experiment_save_path, 'stroke_counts_results.pkl'), 'wb') as f:
        pickle.dump(user_predictions, f)
    
    print(f"\nUser: {user}")
    for rec, data in user_predictions.items():
        print(f"  Recording: {rec}")
        print("  Stroke Counts by Style:", data['stroke_counts'])

if __name__ == '__main__':
    main()
    


