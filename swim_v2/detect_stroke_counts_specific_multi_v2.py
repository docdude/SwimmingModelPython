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

    elif predicted_style == 4:  # Butterfly
        gyro_y_neg_filtered = -gyro_filtered[:, 1]  # Negated Y axis
        gyro_y_neg_synced = synchronize_signals_dtw(acc_magnitude_filtered, gyro_y_neg_filtered, predicted_style)
        sensor_data.update({
            "style_acc": acc_magnitude_filtered,
            "style_gyro": gyro_y_neg_synced,
            "style_acc_peak_key": "acc_magnitude_filtered",
            "style_gyro_peak_key": "gyro_y_negative"
        })

    return sensor_data


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
        """
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
        """
            # Style-specific tuning
        if style == 1:  # Freestyle
            range_scale = 0.01
            std_scale = 0.08
        elif style == 2:  # Breaststroke
            range_scale = 0.02
            std_scale = 0.1
        elif style == 3:  # Backstroke
            range_scale = 0.015
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
        if config and config.DEBUG_DETECT_PEAKS: 
            print(f"Signal analysis:")
            print(f"  Range: {signal_range:.4f}")
            print(f"  Standard Deviation: {local_std:.4f}")
            print(f"  Log(Range + 1): {np.log1p(signal_range):.4f}")
            print(f"  Sqrt(STD): {np.sqrt(local_std):.4f}")
            print(f"Calculated prominence: {dynamic_prominence:.4f}")

        return dynamic_prominence


    acc_signal =filtered_sensor_data["style_acc"]
    gyro_signal = filtered_sensor_data["style_gyro"]

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
            config=config,
            window_pbar=lambda: progress_queue.put({"type": "update", "position": position})
        )
    finally:
        if config and config.DEBUG_PBAR:
            print(f"[DEBUG] Closing progress bar for {rec} at position {position}")

        # Mark progress bar as complete
        progress_queue.put({"type": "close", "desc": f"Processing {rec[:40]:<40}", "position": position})

    return {'recording': rec, 'stroke_counts': stroke_counts, 'stroke_labels': stroke_labels}


def count_strokes_by_style(normalized_windows, raw_windows, predicted_styles, config=None, window_pbar=None):
    """
    Count strokes using style-specific peak detection logic.
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
    for i, (normalized_window, raw_window) in enumerate(zip(normalized_windows, raw_windows)):
        # Determine the predicted style and its confidence
        predicted_style = np.argmax(predicted_styles[i])
        style_confidence = np.max(predicted_styles[i])

        # Skip windows with low confidence or null style (label 0)
        if predicted_style in [0, 5, 6]: # or style_confidence < 0.8:
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

        if config.DEBUG_COUNT_STROKES:
            print(f"Window {i}: stroke_labels[{start_idx}:{end_idx}] = {stroke_labels[start_idx:end_idx]}")

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
    config.DEBUG_SYNCHRONIZATION = False
    config.DEBUG_COUNT_STROKES = False
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
            user = '20'
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
