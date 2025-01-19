# detect_stroke_counts_specific.py
import os
import pickle
import numpy as np
import tensorflow as tf
import learning_data
import scipy.signal  # For peak detection
import matplotlib.pyplot as plt
from collections import deque
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean as scipy_euclidean
from tqdm import tqdm

# Define label names for the stroke styles
label_names_abb = ['Null', 'Fr', 'Br', 'Ba', 'Bu']

# Stroke-style-specific axis and parameter mapping
stroke_axis_params = {
    1: {"acc_axes": [0, 1, 2], "gyro_axes": [0, 1, 2], "prominence": 1.6, "distance": 60},  # Freestyle
    2: {"acc_axes": [0, 1, 2], "gyro_axes": [0, 1, 2], "prominence": 1.6, "distance": 50},  # Breaststroke
    3: {"acc_axes": [0, 1, 2], "gyro_axes": [0, 1, 2], "prominence": 1.1, "distance": 60},  # Backstroke
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
    fig, axs = plt.subplots(2, 5, figsize=(12, 6))
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
        # Annotate each peak with its global row index
        for peak_idx in peak_indices:
            global_row_idx = sensor_data["row_idx"] + peak_idx + window_index * 30
            axs[0, 4].text(
                peak_idx,
                sensor_data["style_acc"][peak_idx],
                f"{global_row_idx}",
                fontsize=8,
                color="black",
                ha="left",
                va="bottom"
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
        # Annotate each peak with its global row index
        for peak_idx in peak_indices:
            global_row_idx = sensor_data["row_idx"] + peak_idx + window_index * 30
            axs[1, 4].text(
                peak_idx,
                sensor_data["style_gyro"][peak_idx],
                f"{global_row_idx}",
                fontsize=8,
                color="black",
                ha="left",
                va="bottom"
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

    plt.figure(figsize=(12, 6))
    plt.plot(signal1, label=label1, alpha=0.7)
    plt.plot(signal2, label=label2, alpha=0.7)
    plt.plot(aligned_signal2, label=f"{label2} (Aligned)", alpha=0.7, linestyle="--")
    plt.legend()
    plt.title("Signal Synchronization Debug")
    plt.grid(True)
    plt.show()


def extract_window_data_and_predictions(swimming_data, data_parameters, user, rec):
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
    dict containing window information including labels.
    """
    # Get window starts and stops
    win_starts, win_stops = swimming_data.window_locs['original'][user][rec]

    # Prepare arrays
    num_windows = len(win_starts)
    windows = np.zeros((num_windows, swimming_data.win_len, len(swimming_data.data_columns)))  # Initialize sensor_windows 
    dup_windows = np.zeros((num_windows, swimming_data.win_len, len(swimming_data.columns)))  # Initialize copy of data set 
    style = np.zeros((num_windows, swimming_data.win_len, 1))  # Initialize label array
    sensor_windows = np.zeros_like(dup_windows)

    # Process each window
    for i, (start, stop) in enumerate(zip(win_starts, win_stops)):
        # Raw sensor data
        raw_data = swimming_data.data_dict['original'][user][rec][swimming_data.data_columns].iloc[start:stop+1].values
        dup_data = swimming_data.data_dict['original'][user][rec][swimming_data.columns].iloc[start:stop+1].values
        # Extract labels from raw data
        labels = swimming_data.data_dict['original'][user][rec]['label'].iloc[start:stop+1].values  # Get labels corresponding to the each row within each window

        sensor_windows[i] = dup_data  # Add timestamps + sensor data + label columns + row index
        # Normalized data
        windows[i] = swimming_data.normalize_window(raw_data, norm_type=data_parameters['window_normalization'])
        style[i] = labels[:, np.newaxis]  # Reshape labels to be (180, 1)



    return {
        'recording': rec,
        'normalized_windows': windows,
        'raw_windows': sensor_windows,
        'labels': style  # Return the labels
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

def synchronize_signals_dtw1(signal1, signal2, style=None):
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
        window_size = 20 # Default window size

    # Create a partial function with the style parameter
    #def style_distance(u, v):
    #    return custom_distance_function(u, v, style)

    # Compute the DTW distance and path with a Sakoe-Chiba band
    distance, path = fastdtw(signal1, signal2, dist=custom_euclidean, radius=window_size)

    # Create an aligned version of signal2 based on the DTW path
    aligned_signal2 = np.zeros_like(signal1)
    
    for i, j in path:
        aligned_signal2[i] = signal2[j]

    # Optional: Print debugging information
    if DEBUG_SYNCHRONIZATION:
        print(f"Style {style} DTW Synchronization:")
        print(f"Window Size: {window_size}")
        print(f"Total DTW Distance: {distance}")

    return aligned_signal2

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
    config : dict, optional
        Additional configuration options (e.g., debug flags).

    Returns:
    --------
    aligned_signal2 : ndarray
        Aligned version of signal2.
    """

    # Ensure signals are NumPy arrays and flatten them to 1-D
    signal1 = np.asarray(signal1).ravel()
    signal2 = np.asarray(signal2).ravel()

    # Check for NaN or Inf values
    if np.any(np.isnan(signal1)) or np.any(np.isnan(signal2)):
        raise ValueError("Input signals contain NaN values.")
    if np.any(np.isinf(signal1)) or np.any(np.isinf(signal2)):
        raise ValueError("Input signals contain infinite values.")

    # Determine window size for DTW based on style
    if style == 4:  # Butterfly
        window_size = 10  # Smaller window for precise alignment
    elif style == 2:  # Breaststroke
        window_size = 15  # Moderate window size
    else:
        window_size = 20  # Default window size

    # Define a style-specific distance function
    def style_specific_distance(u, v, style):
        """
        Compute style-specific distance between two points.

        Parameters:
        -----------
        u : float
            Value from signal1.
        v : float
            Value from signal2.
        style : int
            Swimming style identifier.

        Returns:
        --------
        float
            Computed distance.
        """
        # Default scaling factors (style-specific adjustments)
        if style == 1:  # Freestyle
            scale_acc = 0.5  # Prioritize accelerometer (e.g., Y-axis)
            scale_gyro = 1.5 # Less weight on gyroscope
        elif style == 2:  # Breaststroke
            scale_acc = 0.9
            scale_gyro = 1.1
        elif style == 3:  # Backstroke
            scale_acc = 1.5
            scale_gyro = 0.2
        elif style == 4:  # Butterfly
            scale_acc = 1.2
            scale_gyro = 0.8
        else:
            scale_acc = 1.0
            scale_gyro = 1.0  # Default scaling factors

        # Weighted Euclidean distance
        return np.sqrt((scale_acc * u) ** 2 + (scale_gyro * v) ** 2)

    # Use the style-specific distance function for DTW
    distance, path = fastdtw(
        signal1,
        signal2,
        dist=lambda u, v: style_specific_distance(u, v, style),
        radius=window_size
    )

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

def preprocess_sensor_data(normalized_window, label):
    """
    Preprocess accelerometer and gyroscope data for peak detection,
    including style-specific preprocessing logic and visualization.

    Parameters:
    -----------
    window : ndarray
        Sensor data window containing accelerometer and gyroscope data.
    window_index : int or None
        Current window index for labeling in debug plots.

    Returns:
    --------
    list of dicts
        Each dict contains processed sensor data for a valid row.
    """
    processed_data = []

    acc_data = normalized_window[:, :3]  # Assuming ACC_0, ACC_1, ACC_2 are the first three columns
    gyro_data = normalized_window[:, 3:6]  # Assuming GYRO_0, GYRO_1, GYRO_2 are the next three columns

    # Calculate raw magnitudes
    acc_magnitude = calculate_magnitude(acc_data)
    gyro_magnitude = calculate_magnitude(gyro_data)

    # Filter and smooth individual axes
    acc_filtered = acc_data #np.array([preprocess_signal(acc_data[:, j]) for j in range(3)]).T
    gyro_filtered = gyro_data #np.array([preprocess_signal(gyro_data[:, j]) for j in range(3)]).T

    # Filter and smooth magnitudes
    acc_magnitude_filtered = acc_magnitude#preprocess_signal(acc_magnitude)
    gyro_magnitude_filtered = gyro_magnitude #preprocess_signal(gyro_magnitude)
    
    # Identify segments of consistent style
    current_style = label[0]
    start_idx = 0

    for i in range(1, len(label)):
        if label[i] != current_style or i == len(label) - 1:
            # Process the segment with consistent style
            end_idx = i if label[i] != current_style else i + 1
            segment_acc_filtered = acc_filtered[start_idx:end_idx]
            segment_gyro_filtered = gyro_filtered[start_idx:end_idx]
            segment_acc_magnitude_filtered = acc_magnitude_filtered[start_idx:end_idx]
            segment_gyro_magnitude_filtered = gyro_magnitude_filtered[start_idx:end_idx]
            row_idx = start_idx

            # Style-specific processing
            if current_style in [1, 2, 3, 4]:
                sensor_data = {
                    "acc_filtered": segment_acc_filtered,
                    "gyro_filtered": segment_gyro_filtered,
                    "acc_magnitude_filtered": segment_acc_magnitude_filtered,
                    "gyro_magnitude_filtered": segment_gyro_magnitude_filtered,
                    "style": current_style,
                    "style_acc": None,
                    "style_gyro": None,
                    "style_acc_peak_key": None,
                    "style_gyro_peak_key": None,
                    "row_idx": row_idx
                }

                if current_style == 1:  # Freestyle
                    acc_y_neg_filtered = -segment_acc_filtered[:, 1]  # Y Axis
                    gyro_z_pos_filtered = segment_gyro_filtered[:, 2]  # Z axis
                    gyro_z_pos_synced = synchronize_signals_dtw(acc_y_neg_filtered, gyro_z_pos_filtered, current_style)
                    sensor_data.update({
                        "style_acc": acc_y_neg_filtered,
                        "style_gyro": gyro_z_pos_synced,
                        "style_acc_peak_key": "acc_y_negative",
                        "style_gyro_peak_key": "gyro_z_positive"
                    })

                    # Debugging visualization
                    if DEBUG_SYNCHRONIZATION:
                        print(f"Debugging synchronization for Freestyle (Window {i})")
                        debug_synchronization(
                            acc_y_neg_filtered, gyro_z_pos_filtered,
                            label1="Acc Y Negative Filtered (Freestyle)", 
                            label2="Gyro Z Positive Filtered (Freestyle)", 
                            predicted_style=current_style
                        )

                elif current_style == 2:  # Breaststroke
                    gyro_z_neg_filtered = -segment_gyro_filtered[:, 2]  # Z axis
                    gyro_z_neg_synced = synchronize_signals_dtw(segment_acc_magnitude_filtered, gyro_z_neg_filtered, current_style)
                    sensor_data.update({
                        "style_acc": segment_acc_magnitude_filtered,
                        "style_gyro": gyro_z_neg_synced,
                        "style_acc_peak_key": "acc_magnitude",
                        "style_gyro_peak_key": "gyro_z_negative"
                    })
                    # Debugging visualization
                    if DEBUG_SYNCHRONIZATION:
                        print(f"Debugging synchronization for Breaststroke (Window {i})")
                        debug_synchronization(
                            acc_magnitude_filtered, gyro_z_neg_filtered,
                            label1="Acc Magnitude Filtered (Breaststroke)", 
                            label2="Gyro Z Negative Filtered (Breaststroke)",
                            predicted_style=current_style
                        )

                elif current_style == 3:  # Backstroke
                    acc_z_pos_filtered = segment_acc_filtered[:, 2]  # Z axis
                    gyro_y_pos_filtered = segment_gyro_filtered[:, 1]  # Y axis
                    gyro_y_pos_synced = synchronize_signals_dtw(acc_z_pos_filtered, gyro_y_pos_filtered, current_style)
                    sensor_data.update({
                        "style_acc": acc_z_pos_filtered,
                        "style_gyro": gyro_y_pos_synced,
                        "style_acc_peak_key": "acc_z_positive",
                        "style_gyro_peak_key": "gyro_y_positive"
                    })
                    # Debugging visualization
                    if DEBUG_SYNCHRONIZATION:
                        print(f"Debugging synchronization for Backstroke (Window {i})")
                        debug_synchronization(
                            acc_z_pos_filtered, gyro_y_pos_filtered,
                            label1="Acc Z Positive Filtered (Backstroke)", 
                            label2="Gyro Y Positive Filtered (Backstroke)",
                            predicted_style=current_style
                        )

                elif current_style == 4:  # Butterfly
                    gyro_y_neg_filtered = -segment_gyro_filtered[:, 1]  # Negated Y axis
                    gyro_y_neg_synced = synchronize_signals_dtw(segment_acc_magnitude_filtered, gyro_y_neg_filtered, current_style)
                    sensor_data.update({
                        "style_acc": segment_acc_magnitude_filtered,
                        "style_gyro": gyro_y_neg_synced,
                        "style_acc_peak_key": "acc_magnitude_filtered",
                        "style_gyro_peak_key": "gyro_y_negative"
                    })
                    # Debugging visualization
                    if DEBUG_SYNCHRONIZATION:
                        print(f"Debugging synchronization for Butterfly (Window {i})")
                        debug_synchronization(
                            acc_magnitude_filtered, gyro_y_neg_filtered,
                            label1="Acc Magnitude (Butterfly)", 
                            label2="Gyro Y Negative Filtered (Butterfly)",
                            predicted_style=current_style
                        )

                processed_data.append(sensor_data)

            # Update for the next segment
            current_style = label[i]
            start_idx = i

    return processed_data


def detect_peaks(filtered_sensor_data_list, base_prominence=0.5, raw_sensor_data=None, distance=5, window_idx=0):
    """
    Detect peaks for accelerometer and gyroscope data based on the processed sensor data.

    Parameters:
    -----------
    filtered_sensor_data_list : list of dicts
        List containing processed sensor data for valid rows.
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

    acc_peaks = {}
    gyro_peaks = {}

    def calculate_dynamic_prominence(signal, base_prominence=0.5, style=None):
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

        # Style-specific tuning
        if style == 1:  # Freestyle
            range_scale = 0.01
            std_scale = 0.08
        elif style == 2:  # Breaststroke
            range_scale = 0.02
            std_scale = 0.1
        elif style == 3:  # Backstroke
            range_scale = 0.01
            std_scale = 0.09
        elif style == 4:  # Butterfly
            range_scale = 0.025
            std_scale = 0.12
        else:
            range_scale = 0.01
            std_scale = 0.08

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
        if DEBUG_DETECT_PEAKS: 
            print(f"Signal analysis:")
            print(f"  Range: {signal_range:.4f}")
            print(f"  Standard Deviation: {local_std:.4f}")
            print(f"  Log(Range + 1): {np.log1p(signal_range):.4f}")
            print(f"  Sqrt(STD): {np.sqrt(local_std):.4f}")
            print(f"Calculated prominence: {dynamic_prominence:.4f}")

        return dynamic_prominence

    # Iterate through each processed sensor data entry
    for sensor_data in filtered_sensor_data_list:
        if sensor_data is None:
            continue

        # Get the style from the sensor data
        style = int(sensor_data["style"])

        # Skip unwanted styles
        if style not in [1, 2, 3, 4]:
            continue

        # Get style-specific peak detection parameters
        style_params = stroke_axis_params[style]
        base_prominence = style_params["prominence"]
        distance = style_params["distance"]

        acc_signal = sensor_data["style_acc"]
        gyro_signal = sensor_data["style_gyro"]

        # Calculate dynamic prominence using raw data if available
        if raw_sensor_data is not None:
            if DEBUG_DETECT_PEAKS:  
                print("\nCalculating prominence from raw accelerometer data:")
            acc_prominence = calculate_dynamic_prominence(raw_sensor_data["style_acc"], base_prominence, style)
            if DEBUG_DETECT_PEAKS:  
                print("\nCalculating prominence from raw gyroscope data:")
            gyro_prominence = calculate_dynamic_prominence(raw_sensor_data["style_gyro"], base_prominence, style)
        else:
            if DEBUG_DETECT_PEAKS:  
                print("\nCalculating prominence from processed accelerometer data:")
            acc_prominence = calculate_dynamic_prominence(acc_signal, base_prominence, style)
            if DEBUG_DETECT_PEAKS:  
                print("\nCalculating prominence from processed gyroscope data:")
            gyro_prominence = calculate_dynamic_prominence(gyro_signal, base_prominence, style)

        # Debugging output for prominence
        if DEBUG_DETECT_PEAKS:  
            print(f"Calculated Acc Prominence: {acc_prominence}")
            print(f"Calculated Gyro Prominence: {gyro_prominence}")

        # Detect peaks in accelerometer data
        acc_peaks[sensor_data["style_acc_peak_key"]] = scipy.signal.find_peaks(
            acc_signal,
            prominence=acc_prominence,
            #width=2,
            distance=distance
        )[0]

        # Detect peaks in gyroscope data
        gyro_peaks[sensor_data["style_gyro_peak_key"]] = scipy.signal.find_peaks(
            gyro_signal,
            prominence=gyro_prominence,
            #width=2,
            distance=distance
        )[0]

        # Visualization of the detected peaks
        if DEBUG_PLOT_PEAKS:
            plt.figure(figsize=(12, 6))

            # Accelerometer Plot
            plt.subplot(2, 1, 1)
            plt.plot(acc_signal, label='Accelerometer Signal', color='blue')

            # Plot detected accelerometer peaks
            acc_peak_indices = acc_peaks[sensor_data["style_acc_peak_key"]]
            plt.plot(acc_peak_indices, acc_signal[acc_peak_indices], ".", label='Detected Peaks', color='red')

            # Annotate each detected accelerometer peak with its global row_idx
            for peak_idx in acc_peak_indices:
                global_row_idx = sensor_data["row_idx"] + peak_idx + window_idx * 30
                plt.text(peak_idx, acc_signal[peak_idx], f"{global_row_idx}", color='black', fontsize=8, ha='left', va='bottom')

            plt.title(f"Window: {window_idx} Accelerometer Signal with Detected Peaks")
            plt.legend()

            # Gyroscope Plot
            plt.subplot(2, 1, 2)
            plt.plot(gyro_signal, label='Gyroscope Signal', color='orange')

            # Plot detected gyroscope peaks
            gyro_peak_indices = gyro_peaks[sensor_data["style_gyro_peak_key"]]
            plt.plot(gyro_peak_indices, gyro_signal[gyro_peak_indices], ".", label='Detected Peaks', color='red')

            # Annotate each detected gyroscope peak with its global row_idx
            for peak_idx in gyro_peak_indices:
                global_row_idx = sensor_data["row_idx"] + peak_idx + window_idx * 30
                plt.text(peak_idx, gyro_signal[peak_idx], f"{global_row_idx}", color='black', fontsize=8, ha='left', va='bottom')

            plt.title(f"Window: {window_idx} Gyroscope Signal with Detected Peaks")
            plt.legend()

            plt.tight_layout()
            plt.show()

    # Debug print
    if DEBUG_DETECT_PEAKS:
        print("Detected acc_peaks:", acc_peaks)
        print("Detected gyro_peaks:", gyro_peaks)

    return acc_peaks, gyro_peaks



# Update the count_strokes_by_style function to calculate global prominence
def count_strokes_by_style(normalized_windows, raw_windows, labels, user_number=None, file_name=None):
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

    # Initialize stroke counts dictionary for each style
    stroke_counts = {label: 0 for label in label_names_abb}

    # Initialize global trackers for peaks and stroke labels
    global_acc_peaks = set()
    global_gyro_peaks = set()
    stroke_labels_indices = set()  # To avoid duplicate labeling

    # Initialize stroke_labels array for the entire dataset
    num_samples = (len(raw_windows) - 1) * 30 + 180  # Account for window overlap
    stroke_labels = np.zeros(num_samples)

            # Iterate through all windows
    for i, (normalized_window, raw_window, label) in enumerate(tqdm(
            zip(normalized_windows, raw_windows, labels),
            total=len(normalized_windows), 
            desc="Processing Windows", 
            leave=False  # Removes the bar after completion
        )):

        # Preprocess sensor data for peak detection
        filtered_sensor_data_list = preprocess_sensor_data(normalized_window, label)

        # Extract raw sensor data
        raw_sensor_data = {
            "style_acc": raw_window[:, 2:5],  # ACC_0, ACC_1, ACC_2
            "style_gyro": raw_window[:, 5:8],  # GYRO_0, GYRO_1, GYRO_2
        }

        # Perform peak detection
        acc_peaks, gyro_peaks = detect_peaks(
            filtered_sensor_data_list,
            base_prominence=0.5,
            raw_sensor_data=raw_sensor_data,
            distance=5,
            window_idx=i
        )

        # Iterate through each processed sensor data entry
        for sensor_data in filtered_sensor_data_list:
            style = int(sensor_data["style"])
            row_idx = sensor_data["row_idx"] + i * 30 # add sliding length offset 
      
            # Debugging output
            if DEBUG_COUNT_STROKES:
                print(f"[DEBUG] Window {i}: Detected Acc Peaks: {acc_peaks}, Gyro Peaks: {gyro_peaks}")

            valid_strokes = 0
            # Access the correct keys based on the sensor data
            acc_peak_indices = acc_peaks.get(sensor_data["style_acc_peak_key"], [])
            gyro_peak_indices = gyro_peaks.get(sensor_data["style_gyro_peak_key"], [])

            if DEBUG_COUNT_STROKES:
                print(f"Processing Style: {label_names_abb[style]}")
                print(f"Acc Peaks: {acc_peak_indices}")
                print(f"Gyro Peaks: {gyro_peak_indices}")

            # Match accelerometer and gyroscope peaks
            for acc_peak in acc_peak_indices:
                global_acc_peak = int(acc_peak + row_idx) #int(raw_window[acc_peak, 0])  # Global row index for accelerometer
                for gyro_peak in gyro_peak_indices:
                    global_gyro_peak = int(gyro_peak + row_idx) #int(raw_window[gyro_peak, 0])  # Global row index for gyroscope

                    # Synchronization tolerance check
                    if abs(acc_peak - gyro_peak) <= 5:
                        if global_acc_peak not in global_acc_peaks and global_gyro_peak not in global_gyro_peaks:
                            valid_strokes += 1
                            global_acc_peaks.add(global_acc_peak)
                            global_gyro_peaks.add(global_gyro_peak)
                            stroke_labels[global_acc_peak] = 1  # Label the stroke
                            stroke_labels_indices.add(global_acc_peak)
                            break

            # Update stroke counts
            stroke_counts[label_names_abb[style]] += valid_strokes

            # Debugging output for strokes and labels
            start_idx = i * 30
            end_idx = start_idx + 180
            if DEBUG_COUNT_STROKES:
                #print(f"[DEBUG] Window {i}: stroke_labels[{start_idx}:{end_idx}] = {stroke_labels[start_idx:end_idx]}")
                print(f"[DEBUG] Window {i}, Style: {label_names_abb[style]}, Valid Strokes: {valid_strokes}, Stoke Label Indices: {sorted(stroke_labels_indices)} numlabels: {len(stroke_labels_indices)}")



            # Optional visualization
            if DEBUG_COUNT_STROKES_PLOTTER:
                plot_data_with_peaks(
                    sensor_data,
                    acc_peaks,
                    gyro_peaks,
                    style,
                    window_index=i,
                    user_number=user_number,
                    file_name=file_name
                )


            if DEBUG_COUNT_STROKES:
                print(f"[DEBUG] Window {i}, Style: {label_names_abb[style]}, "
                    f"Valid Strokes: {valid_strokes}, Acc Peaks: {acc_peak_indices}, Gyro Peaks: {gyro_peak_indices}")

            if DEBUG_COUNT_STROKES:
                print(f"[DEBUG] Window {i}: stroke_labels[{start_idx}:{end_idx}] = {stroke_labels[start_idx:end_idx]}")

        # Final debugging output for stroke_labels
        #if DEBUG_COUNT_STROKES:
        #    print(f"[DEBUG] Final stroke_labels array:\n{stroke_labels}")

    # Return stroke_counts and stroke_labels
    return stroke_counts, stroke_labels

# Other existing code remains unchanged...

def main():
        # Global debug flags can be set here
    global DEBUG_DETECT_PEAKS, DEBUG_PLOT_PEAKS, DEBUG_REAL_TIME_PLOTTER, DEBUG_SYNCHRONIZATION, DEBUG_COUNT_STROKES, DEBUG_COUNT_STROKES_PLOTTER
    
    # Turn on/off specific debug outputs
    DEBUG_DETECT_PEAKS = False
    DEBUG_PLOT_PEAKS = False
    DEBUG_REAL_TIME_PLOTTER = False
    DEBUG_SYNCHRONIZATION = False
    DEBUG_COUNT_STROKES = False
    DEBUG_COUNT_STROKES_PLOTTER = False
    data_path = '/Users/juanloya/Documents/SwimmingModelPython/swim_v2/data_copy/processed_30Hz_relabeled'

    #data_path = '/Users/juanloya/Documents/SwimmingModelPython/swim_v2/SwimStyleData2_orig'
    results_path = '/Users/juanloya/Documents/SwimmingModelPython/swim_v2/tutorial_save_path_epoch60_mag/'
    # Define the output directory for updated files
    output_dir = '/Users/juanloya/Documents/SwimmingModelPython/swim_v2/data_modified'

    user = '20'
    experiment_save_path = os.path.join(results_path, user)

    with open(os.path.join(results_path, user, 'data_parameters.pkl'), 'rb') as f:
        data_parameters = pickle.load(f)[0] 
    swimming_data = learning_data.LearningData()
    swimming_data.load_data(data_path=data_path, 
                             data_columns=data_parameters['data_columns'],
                             users=[user], 
                             labels=data_parameters['labels'])
    
    
    swimming_data.sliding_window_locs(win_len=data_parameters['win_len'], 
                                      slide_len=data_parameters['slide_len'])
    
    user_number = 1
    print(f"Working on User {user_number}: {user}.")   
        
    user_predictions = {}
        
    # Get the list of recordings to process
    recordings = list(swimming_data.data_dict['original'][user].keys())
    
    # Use tqdm to create a progress bar for the recordings
    for rec in tqdm(recordings, 
                    desc=f"Processing Recordings for User {user}", 
                    total=len(recordings),  # Specify total number of recordings
                    unit="recording",       # Unit of progress
                    colour='green',         # Optional: color of progress bar
                    ncols=100,              # Optional: width of progress bar
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):
        print(f"Processing recording: {rec}")
        file_name = rec  # Use the recording name as the file name
            
        window_data = extract_window_data_and_predictions(swimming_data,
                                                              data_parameters, user, rec)
            
        stroke_counts, stroke_labels = count_strokes_by_style(
                window_data['normalized_windows'], 
                window_data['raw_windows'], 
                window_data['labels'], 
                user_number=user,
                file_name=rec
            )
            
        user_predictions[rec] = {
            'window_data': window_data,
            'stroke_counts': stroke_counts,
            'stroke_labels': stroke_labels
        }
        # Path to where we want to save the training results
        sub_dir_path = os.path.join(output_dir, user)
        base_name, _ = os.path.splitext(rec)  # Split into ('file_name', '.csv')
        new_file_name = f"{base_name}_updated.csv"

        # Ensure the output directory exists
        os.makedirs(sub_dir_path, exist_ok=True)  # Create the user-specific directory

        # Check if stroke_labels need padding to match the dataset
        dataset_length = swimming_data.data_dict['original'][user][rec].shape[0]
        if len(stroke_labels) < dataset_length:
            padding_length = dataset_length - len(stroke_labels)
            stroke_labels = np.pad(stroke_labels, (0, padding_length), 'constant', constant_values=0)

        # Update stroke labels in swimming_data
        swimming_data.data_dict['original'][user][rec]['stroke_labels'] = stroke_labels

        # Save the updated dataset with stroke labels
        updated_df = swimming_data.data_dict['original'][user][rec]
        updated_df.to_csv(
            os.path.join(sub_dir_path, new_file_name),
            index=False
        )
        print(f"Updated data saved to: {sub_dir_path}")
        
    with open(os.path.join(experiment_save_path, 'stroke_counts_results.pkl'), 'wb') as f:
        pickle.dump(user_predictions, f)
    
    print(f"\nUser: {user}")
    for rec, data in user_predictions.items():
        print(f"  Recording: {rec}")
        print("  Stroke Counts by Style:", data['stroke_counts'])

if __name__ == '__main__':
    main()
