import os
import pickle
import numpy as np
import tensorflow as tf
import learning_data
import utils
import scipy.signal  # For peak detection
import matplotlib.pyplot as plt
from collections import deque

# Define label names for the stroke styles
label_names_abb = ['Null', 'Fr', 'Br', 'Ba', 'Bu']

# Stroke-style-specific axis and parameter mapping
stroke_axis_params = {
    1: {"acc_axes": [0, 1, 2], "gyro_axes": [0, 1, 2], "prominence": 1.0, "distance": 10},  # Freestyle
    2: {"acc_axes": [0, 1, 2], "gyro_axes": [0, 1, 2], "prominence": 0.8, "distance": 45}, # Breaststroke
    3: {"acc_axes": [0, 1, 2], "gyro_axes": [0, 1, 2], "prominence": 1.0, "distance": 40},  # Backstroke
    4: {"acc_axes": [ 1], "gyro_axes": [1], "prominence": 1.5, "distance": 35}, # Butterfly
}

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

    return {
        'normalized_windows': windows,
        'sensor_windows': sensor_windows,
        'true_windows_cat': y_true_windows,
        'true_windows_maj': y_true_windows_maj,
        'predicted_windows': y_pred_windows
    }

def plot_data_with_peaks(
    acc_data, gyro_data, acc_magnitude, gyro_magnitude, acc_peaks, gyro_peaks, predicted_style, window_index, user_number=None, file_name=None
):
    """
    Plot accelerometer and gyroscope data with detected peaks, including positive and negative peaks.
    """
    num_axes = 3
    fig, axs = plt.subplots(2, num_axes + 1, figsize=(15, 8))

    # Plot accelerometer axes
    for i, axis_label in enumerate(["X", "Y", "Z"]):
        axs[0, i].plot(acc_data[:, i], label=f'Acc {axis_label}')
        # Positive peaks
        if "positive" in acc_peaks and i in acc_peaks["positive"]:
            axs[0, i].plot(acc_peaks["positive"][i], acc_data[acc_peaks["positive"][i], i], 'ro', label=f'Acc +Peaks {axis_label}')
        # Negative peaks
        if "negative" in acc_peaks and i in acc_peaks["negative"]:
            axs[0, i].plot(acc_peaks["negative"][i], acc_data[acc_peaks["negative"][i], i], 'bo', label=f'Acc -Peaks {axis_label}')
        axs[0, i].set_title(f'Accelerometer {axis_label}')
        axs[0, i].legend()
        axs[0, i].grid(True)

    # Plot accelerometer magnitude
    axs[0, num_axes].plot(acc_magnitude, label='Acc Magnitude')
    if isinstance(acc_peaks["positive"], np.ndarray):
        axs[0, num_axes].plot(acc_peaks["positive"], acc_magnitude[acc_peaks["positive"]], 'ro', label='Acc Magnitude Peaks')
    axs[0, num_axes].set_title('Accelerometer Magnitude')
    axs[0, num_axes].legend()
    axs[0, num_axes].grid(True)

    # Plot gyroscope axes
    for i, axis_label in enumerate(["X", "Y", "Z"]):
        axs[1, i].plot(gyro_data[:, i], label=f'Gyro {axis_label}')
        # Positive peaks
        if "positive" in gyro_peaks and i in gyro_peaks["positive"]:
            axs[1, i].plot(gyro_peaks["positive"][i], gyro_data[gyro_peaks["positive"][i], i], 'ro', label=f'Gyro +Peaks {axis_label}')
        # Negative peaks
        if "negative" in gyro_peaks and i in gyro_peaks["negative"]:
            axs[1, i].plot(gyro_peaks["negative"][i], gyro_data[gyro_peaks["negative"][i], i], 'bo', label=f'Gyro -Peaks {axis_label}')
        axs[1, i].set_title(f'Gyroscope {axis_label}')
        axs[1, i].legend()
        axs[1, i].grid(True)

    # Plot gyroscope magnitude
    axs[1, num_axes].plot(gyro_magnitude, label='Gyro Magnitude')
    if isinstance(gyro_peaks["positive"], np.ndarray):
        axs[1, num_axes].plot(gyro_peaks["positive"], gyro_magnitude[gyro_peaks["positive"]], 'ro', label='Gyro Magnitude Peaks')
    axs[1, num_axes].set_title('Gyroscope Magnitude')
    axs[1, num_axes].legend()
    axs[1, num_axes].grid(True)

    fig.suptitle(
        f'User: {user_number}, File: {file_name}, Window: {window_index} (Style: {label_names_abb[predicted_style]})',
        fontsize=16,
    )
    plt.tight_layout()
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

        for ax in self.ax[:2]:  # Clear individual and magnitude plots
            ax.clear()
            ax.grid(True)

        style_color = self.style_colors[predicted_style]

        # Plot accelerometer magnitude
        self.ax[0].plot(acc_magnitude, color=style_color, label='Acc Magnitude')
        if len(acc_peaks) > 0:
            self.ax[0].plot(acc_peaks, acc_magnitude[acc_peaks], 'ro', markersize=5, label='Acc Peaks')
        self.ax[0].set_title(f'Accelerometer Magnitude (Style: {label_names_abb[predicted_style]}, Confidence: {style_confidence:.2f})')

        # Plot gyroscope magnitude
        self.ax[1].plot(gyro_magnitude, color=style_color, label='Gyro Magnitude')
        if len(gyro_peaks) > 0:
            self.ax[1].plot(gyro_peaks, gyro_magnitude[gyro_peaks], 'ro', markersize=5, label='Gyro Peaks')
        self.ax[1].set_title('Gyroscope Magnitude')

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


def preprocess_sensor_data(window):
    """
    Preprocess accelerometer and gyroscope data for peak detection.
    Applies high-pass filtering and smoothing to all axes and magnitudes.
    """
    acc_data = window[:, :3]
    gyro_data = window[:, 3:6]

    # Calculate magnitude
    acc_magnitude = calculate_magnitude(acc_data)
    gyro_magnitude = calculate_magnitude(gyro_data)

    # Apply high-pass filtering and smoothing
    acc_filtered = np.array([preprocess_signal(acc_data[:, j]) for j in range(3)]).T
    gyro_filtered = np.array([preprocess_signal(gyro_data[:, j]) for j in range(3)]).T
    acc_magnitude_filtered = preprocess_signal(acc_magnitude)
    gyro_magnitude_filtered = preprocess_signal(gyro_magnitude)

    # Synchronize signals
    gyro_synced = synchronize_signals(acc_magnitude_filtered, gyro_magnitude_filtered)

    return {
        "acc": acc_filtered,
        "gyro": gyro_filtered,
        "acc_magnitude": acc_magnitude_filtered,
        "gyro_magnitude": gyro_synced,
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

def synchronize_signals(signal1, signal2):
    """
    Synchronize two signals using cross-correlation.
    :param signal1: Reference signal (e.g., accelerometer magnitude).
    :param signal2: Signal to align with the reference (e.g., gyroscope magnitude).
    :return: Aligned signal2.
    """
    correlation = np.correlate(signal1 - np.mean(signal1), signal2 - np.mean(signal2), mode='full')
    lag = np.argmax(correlation) - (len(signal2) - 1)

    if lag > 0:
        aligned_signal2 = np.pad(signal2, (lag, 0), mode='constant')[:len(signal1)]
    elif lag < 0:
        aligned_signal2 = signal2[-lag:]
        aligned_signal2 = np.pad(aligned_signal2, (0, len(signal1) - len(aligned_signal2)), mode='constant')
    else:
        aligned_signal2 = signal2

    return aligned_signal2

def preprocess_signal(signal, cutoff=0.3, fs=30, savgol_window=5, savgol_poly=3):
    filtered_signal = butter_highpass_filter(signal, cutoff, fs)
    smoothed_signal = smooth_data_with_savgol(filtered_signal, savgol_window, savgol_poly)
    return smoothed_signal

def detect_peaks(sensor_data, predicted_style, base_prominence=0.5):
    """
    Detect peaks for accelerometer and gyroscope data based on the predicted stroke style.
    Considers both magnitude and axis-specific data.
    """
    params = stroke_axis_params.get(predicted_style, {})
    acc_axes = params.get("acc_axes", [])
    gyro_axes = params.get("gyro_axes", [])
    prominence = params.get("prominence", base_prominence)
    distance = params.get("distance", 10)

    acc_peaks = {"positive": {}, "negative": {}}
    gyro_peaks = {"positive": {}, "negative": {}}

    # Axis-specific peak detection
    for axis in acc_axes:
        # Positive peaks
        acc_peaks["positive"][axis], _ = scipy.signal.find_peaks(
            sensor_data["acc"][:, axis],
            prominence=prominence,
            distance=distance,
        )
        # Negative peaks
        acc_peaks["negative"][axis], _ = scipy.signal.find_peaks(
            -sensor_data["acc"][:, axis],  # Flip signal for negative peak detection
            prominence=prominence,
            distance=distance,
        )

    for axis in gyro_axes:
        # Positive peaks
        gyro_peaks["positive"][axis], _ = scipy.signal.find_peaks(
            sensor_data["gyro"][:, axis],
            prominence=prominence,
            distance=distance,
        )
        # Negative peaks
        gyro_peaks["negative"][axis], _ = scipy.signal.find_peaks(
            -sensor_data["gyro"][:, axis],  # Flip signal for negative peak detection
            prominence=prominence,
            distance=distance,
        )

    # Magnitude-based peak detection
    mag_acc_peaks, _ = scipy.signal.find_peaks(
        sensor_data["acc_magnitude"],
        prominence=prominence,
        distance=distance,
    )
    mag_gyro_peaks, _ = scipy.signal.find_peaks(
        sensor_data["gyro_magnitude"],
        prominence=prominence,
        distance=distance,
    )

    return {
        "acc": acc_peaks,
        "gyro": gyro_peaks,
        "acc_magnitude": mag_acc_peaks,
        "gyro_magnitude": mag_gyro_peaks,
    }


def detect_peaks_with_advanced_threshold(sensor_data, predicted_style, base_prominence=0.5, min_peaks=30, fs=30):
    """
    Enhanced peak detection with adaptive dynamic prominence and relaxed constraints.
    """
    local_mean = np.mean(sensor_data)
    local_std = np.std(sensor_data)
    signal_range = np.ptp(sensor_data)

    dynamic_prominence = base_prominence * max(local_std / (local_mean + 1e-6), 1) * signal_range

    # Enhanced peak detection logic
    style_params = {
        0: {'prominence_mult': 0, 'distance': None},  # Null/Turn
        1: {'prominence_mult': 0.2, 'distance': 10},  # Front Crawl
        2: {'prominence_mult': 0.8, 'distance': 45},  # Breaststroke
        3: {'prominence_mult': 1.0, 'distance': 40},  # Backstroke
        4: {'prominence_mult': 1.5, 'distance': 35}   # Butterfly
    }

    style_mult = style_params.get(predicted_style, {}).get('prominence_mult', 0.5)
    min_distance = style_params.get(predicted_style, {}).get('distance', 10)
    adjusted_prominence = max(dynamic_prominence * style_mult, base_prominence)

    peaks, properties = scipy.signal.find_peaks(
        sensor_data,
        prominence=adjusted_prominence,
        distance=min_distance
    )

    if len(peaks) < min_peaks:
        peaks, properties = scipy.signal.find_peaks(
            sensor_data,
            prominence=adjusted_prominence * 0.5,
            distance=min_distance
        )

    return peaks, properties, adjusted_prominence

def count_strokes_by_style(sensor_windows, predicted_styles, base_prominence=0.5, user_number=None, file_name=None, plot=False):
    """
    Count strokes using refined preprocessing, signal synchronization, and peak detection.
    """
    stroke_counts = {label: 0 for label in label_names_abb}
    plotter = RealTimePlotter(user_number=user_number, file_name=file_name) if plot else None
    global_acc_peaks = set()
    global_gyro_peaks = set()

    for i, window in enumerate(sensor_windows):
        # Calculate magnitudes
        acc_magnitude = calculate_magnitude(window[:, :3])
        gyro_magnitude = calculate_magnitude(window[:, 3:6])

        # Synchronize gyroscope signal with accelerometer
        gyro_magnitude_synced = synchronize_signals(acc_magnitude, gyro_magnitude)

        # Debugging synchronization
        if i >= 0:  # Debug only for the first window (or any specific window you choose)
            print(f"Debugging synchronization for window {i}")
            debug_synchronization(acc_magnitude, gyro_magnitude)

        # Preprocess signals (filter + smooth)
        acc_magnitude_preprocessed = preprocess_signal(acc_magnitude)
        gyro_magnitude_preprocessed = preprocess_signal(gyro_magnitude_synced)

        # Predict stroke style and confidence
        predicted_style = np.argmax(predicted_styles[i])
        style_confidence = np.max(predicted_styles[i])

        if predicted_style != 0 and style_confidence > 0.8:
            # Detect peaks in preprocessed signals
            acc_peaks, acc_props, acc_prominence = detect_peaks_with_advanced_threshold(
                acc_magnitude, predicted_style, base_prominence)
            gyro_peaks, gyro_props, gyro_prominence = detect_peaks_with_advanced_threshold(
                gyro_magnitude_synced, predicted_style, base_prominence)

            # Match peaks and count strokes
            valid_strokes = 0
            for acc_peak in acc_peaks:
                global_acc_peak = acc_peak + i * 30
                for gyro_peak in gyro_peaks:
                    global_gyro_peak = gyro_peak + i * 30
                    if abs(acc_peak - gyro_peak) <= 5:  # Synchronization tolerance
                        if global_acc_peak not in global_acc_peaks and global_gyro_peak not in global_gyro_peaks:
                            valid_strokes += 1
                            global_acc_peaks.add(global_acc_peak)
                            global_gyro_peaks.add(global_gyro_peak)
                            break

            # Update stroke counts
            stroke_counts[label_names_abb[predicted_style]] += valid_strokes
            if plot:
                plotter.stroke_counts[label_names_abb[predicted_style]] += valid_strokes

            print(f"Window {i}: Counted {valid_strokes} valid strokes")
            # Visualize the data and detected peaks for problematic windows
            if valid_strokes >= 1:  # Replace with your expected count condition
                plot_data_with_peaks(acc_magnitude, gyro_magnitude_synced, acc_peaks, gyro_peaks, predicted_style, i, user_number, file_name)

        if plot:
            plotter.update_plot(acc_magnitude, gyro_magnitude, predicted_style, acc_peaks, gyro_peaks, style_confidence, i)

    if plot:
        plotter.close_plot()

    return stroke_counts

def count_strokes_by_style1(sensor_windows, predicted_styles, base_prominence=0.75, user_number=None, file_name=None, plot=False):
    stroke_counts = {label: 0 for label in label_names_abb}
    global_acc_peaks = set()
    global_gyro_peaks = set()
    plotter = RealTimePlotter(user_number=user_number, file_name=file_name) if plot else None

    for i, window in enumerate(sensor_windows):
        predicted_style = np.argmax(predicted_styles[i])
        style_confidence = np.max(predicted_styles[i])

        if predicted_style == 0 or style_confidence < 0.8:
            continue

        sensor_data = preprocess_sensor_data(window)
        peaks = detect_peaks(sensor_data, predicted_style, base_prominence)

        valid_strokes = 0
        for acc_peak in peaks["acc_magnitude"]:
            global_acc_peak = acc_peak + i * 30
            for gyro_peak in peaks["gyro_magnitude"]:
                global_gyro_peak = gyro_peak + i * 30

                if abs(acc_peak - gyro_peak) <= 5:
                    if global_acc_peak not in global_acc_peaks and global_gyro_peak not in global_gyro_peaks:
                        valid_strokes += 1
                        global_acc_peaks.add(global_acc_peak)
                        global_gyro_peaks.add(global_gyro_peak)

        # Update stroke counts
        stroke_counts[label_names_abb[predicted_style]] += valid_strokes
        if plot:
            plotter.stroke_counts[label_names_abb[predicted_style]] += valid_strokes

        print(f"Window {i}: Counted {valid_strokes} valid strokes")
        if valid_strokes >= 1:  # Adjust threshold as needed
            plot_data_with_peaks(
                sensor_data["acc"],
                sensor_data["gyro"],
                sensor_data["acc_magnitude"],
                sensor_data["gyro_magnitude"],
                peaks["acc"],
                peaks["gyro"],
                predicted_style,
                i,
                user_number,
                file_name
            )


        if plot:
            plotter.update_plot(
                sensor_data["acc"],
                sensor_data["gyro"],
                sensor_data["acc_magnitude"],
                sensor_data["gyro_magnitude"],
                peaks["acc_magnitude"],
                peaks["gyro_magnitude"],
                predicted_style,
                style_confidence,
                i
            )

    if plot:
        plotter.close_plot()

    return stroke_counts

def main():
    data_path = '/Users/juanloya/Documents/SwimmingModelPython/swim_v2/data/processed_30Hz_relabeled'
    results_path = '/Users/juanloya/Documents/SwimmingModelPython/swim_v2/tutorial_save_path_epoch60/'
    
    #with open(os.path.join(results_path, '22/data_parameters.pkl'), 'rb') as f:
     #   data_parameters = pickle.load(f)[0]
    
    #users = data_parameters['users']
        # User whose model we want to load
    user = '3'

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
            
        stroke_counts = count_strokes_by_style1(
                window_data['normalized_windows'], 
                window_data['predicted_windows'], 
                base_prominence=0.5,
                user_number= user_number,
                file_name=rec,
                plot=False
            )
            
        user_predictions[rec] = {
            'window_data': window_data,
            'stroke_counts': stroke_counts
        }
    
    with open('stroke_counts_results.pkl', 'wb') as f:
        pickle.dump(user_predictions, f)
    
    print(f"\nUser: {user}")
    for rec, data in user_predictions.items():
        print(f"  Recording: {rec}")
        print("  Stroke Counts by Style:", data['stroke_counts'])

if __name__ == '__main__':
    main()