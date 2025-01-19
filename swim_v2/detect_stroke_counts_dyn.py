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
    1: {  # Freestyle
        "acc_axes": [2],  # Accelerometer Z-axis (Up/Down)
        "gyro_axes": [0, 2],  # Gyroscope X (Pitch) and Z (Yaw)
        "prominence": 0.5,
        "distance": 8,
    },
    2: {  # Breaststroke
        "acc_axes": [2],  # Accelerometer Z-axis
        "gyro_axes": [0],  # Gyroscope X (Pitch)
        "prominence": 0.6,
        "distance": 10,
    },
    3: {  # Backstroke
        "acc_axes": [2],  # Accelerometer Z-axis
        "gyro_axes": [0, 2],  # Gyroscope X (Pitch) and Z (Yaw)
        "prominence": 0.5,
        "distance": 8,
    },
    4: {  # Butterfly
        "acc_axes": [2],  # Accelerometer Z-axis
        "gyro_axes": [0],  # Gyroscope X (Pitch)
        "prominence": 0.8,
        "distance": 12,
    }
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

def detect_peaks_with_advanced_threshold(sensor_data, predicted_style, base_prominence=0.5, window_size=180):
    # Enhanced peak detection logic
    style_params = {
        0: {'prominence_mult': 0, 'distance': None},  # Null/Turn
        1: {'prominence_mult': 0.2, 'distance': 10},  # Front Crawl
        2: {'prominence_mult': 0.8, 'distance': 45},  # Breaststroke
        3: {'prominence_mult': 1.0, 'distance': 40},  # Backstroke
        4: {'prominence_mult': 1.5, 'distance': 35}   # Butterfly
    }
    
    local_mean = np.mean(sensor_data)
    local_std = np.std(sensor_data, ddof=1)
    signal_range = np.ptp(sensor_data)
    #print(signal_range)
    #print(local_mean, local_std)
    if local_mean > 0:
        dynamic_prominence = base_prominence * (local_std / local_mean) * signal_range
    else:
        dynamic_prominence = base_prominence * signal_range
    
    style_mult = style_params[predicted_style]['prominence_mult']
    min_distance = style_params[predicted_style]['distance']
    
    adjusted_prominence = dynamic_prominence * style_mult
    print(adjusted_prominence)
    peaks, properties = scipy.signal.find_peaks(
        sensor_data,
        prominence=adjusted_prominence,
        distance=min_distance,
        width=1
    )
    
    return peaks, properties, adjusted_prominence

def plot_data_with_peaks(acc_magnitude, gyro_magnitude, acc_peaks_dict, gyro_peaks_dict, predicted_style, window_index, user_number=None, file_name=None):
    """
    Plot accelerometer and gyroscope magnitudes with detected peaks.
    
    Parameters:
    ----------
    acc_magnitude : array-like
        Smoothed accelerometer magnitude.
    gyro_magnitude : array-like
        Smoothed gyroscope magnitude.
    acc_peaks_dict : dict
        Detected peaks for accelerometer axes (key: axis, value: list of peak indices).
    gyro_peaks_dict : dict
        Detected peaks for gyroscope axes (key: axis, value: list of peak indices).
    predicted_style : int
        Predicted stroke style.
    window_index : int
        Window index for the current segment.
    user_number : str or int, optional
        Identifier for the user.
    file_name : str, optional
        Name of the file being processed.
    """
    plt.figure(figsize=(12, 6))
    
    # Plot accelerometer magnitude
    plt.subplot(2, 1, 1)
    plt.plot(acc_magnitude, label='Accelerometer Magnitude')
    for axis, peaks in acc_peaks_dict.items():
        plt.plot(peaks, acc_magnitude[peaks], 'ro', label=f'Acc Peaks (Axis {axis})')
    plt.title(f'User: {user_number}, File: {file_name}  '
              f'Accelerometer Magnitude - Window {window_index} '
              f'(Style: {label_names_abb[predicted_style]})')
    plt.legend()

    # Plot gyroscope magnitude
    plt.subplot(2, 1, 2)
    plt.plot(gyro_magnitude, label='Gyroscope Magnitude')
    for axis, peaks in gyro_peaks_dict.items():
        plt.plot(peaks, gyro_magnitude[peaks], 'ro', label=f'Gyro Peaks (Axis {axis})')
    plt.title(f'Window {window_index} - Gyroscope')
    plt.legend()
    
    plt.tight_layout()
    plt.show()



class RealTimePlotter:
    def __init__(self, window_size=180, history_size=540, user_number=None, file_name=None):
        self.window_size = window_size
        self.history_size = history_size
        self.user_number = user_number
        self.file_name = file_name
        self.fig = plt.figure(figsize=(15, 10))
        self.gs = self.fig.add_gridspec(3, 1, height_ratios=[2, 2, 1])
        self.ax = [
            self.fig.add_subplot(self.gs[0]),
            self.fig.add_subplot(self.gs[1]),
            self.fig.add_subplot(self.gs[2])
        ]
        self.history_acc = deque(maxlen=history_size)
        self.history_gyro = deque(maxlen=history_size)
        self.stroke_counts = {label: 0 for label in label_names_abb}
        self.style_colors = {
            0: 'gray',    # Null/Turn
            1: 'blue',    # Front Crawl
            2: 'green',   # Breaststroke
            3: 'red',     # Backstroke
            4: 'purple'   # Butterfly
        }
        self.setup_plot()

    def setup_plot(self):
        for ax in self.ax[:2]:
            ax.set_xlim(0, self.history_size)
            ax.set_ylim(0, 5)
            ax.grid(True)
        
        self.ax[2].set_xlim(-1, len(label_names_abb))
        self.ax[2].set_ylim(0, 10)
        self.ax[2].grid(True)
        self.ax[2].set_xticks(range(len(label_names_abb)))
        self.ax[2].set_xticklabels(label_names_abb)
        
        self.fig.tight_layout()
        plt.ion()

    def update_plot(self, acc_data, gyro_data, peaks, predicted_style, style_confidence, window_index):
        """
        Update the real-time plot with accelerometer and gyroscope data.
        """
        self.history_acc.extend(acc_data)
        self.history_gyro.extend(gyro_data)

        acc_hist = np.array(self.history_acc)
        gyro_hist = np.array(self.history_gyro)

        # Clear and reset the axes
        for ax in self.ax:
            ax.clear()
            ax.grid(True)

        style_color = self.style_colors[predicted_style]

        # Plot accelerometer history
        self.ax[0].plot(acc_hist, color=style_color, alpha=0.7, label="Accelerometer Data")
        for axis, axis_peaks in peaks["acc"].items():
            adjusted_acc_peaks = [p + len(acc_hist) - len(acc_data) for p in axis_peaks if p < len(acc_hist)]
            self.ax[0].plot(adjusted_acc_peaks, acc_hist[adjusted_acc_peaks], 'ro', markersize=5, label=f"Acc Axis {axis} Peaks")
        self.ax[0].set_title(f"User: {self.user_number}, File: {self.file_name} - Accelerometer (Window {window_index})\n"
                             f"Style: {label_names_abb[predicted_style]}, Confidence: {style_confidence:.2f}")
        self.ax[0].legend()

        # Plot gyroscope history
        self.ax[1].plot(gyro_hist, color=style_color, alpha=0.7, label="Gyroscope Data")
        for axis, axis_peaks in peaks["gyro"].items():
            adjusted_gyro_peaks = [p + len(gyro_hist) - len(gyro_data) for p in axis_peaks if p < len(gyro_hist)]
            self.ax[1].plot(adjusted_gyro_peaks, gyro_hist[adjusted_gyro_peaks], 'ro', markersize=5, label=f"Gyro Axis {axis} Peaks")
        self.ax[1].set_title(f"Gyroscope Data (Window {window_index})")
        self.ax[1].legend()

        # Update cumulative stroke counts
        x = np.arange(len(label_names_abb))
        counts = [self.stroke_counts[label] for label in label_names_abb]
        bars = self.ax[2].bar(x, counts, color=[self.style_colors[i] for i in range(len(label_names_abb))])
        for bar, count in zip(bars, counts):
            self.ax[2].text(bar.get_x() + bar.get_width() / 2., bar.get_height(), f'{int(count)}', ha='center', va='bottom')

        self.ax[2].set_title("Cumulative Stroke Counts")
        self.ax[2].set_xticks(x)
        self.ax[2].set_xticklabels(label_names_abb)
        max_count = max(counts) if counts else 10
        self.ax[2].set_ylim(0, max_count * 1.1)

        plt.pause(0.01)

    def close_plot(self):
        plt.ioff()
        plt.close(self.fig)


def smooth_data(data, window_size=2):
    """Smooth the data using a moving average."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

def debug_synchronization(signal1, signal2):
    """Debug synchronization by plotting signals before and after alignment."""
    aligned_signal2 = synchronize_signals(signal1, signal2)

    plt.figure(figsize=(10, 6))
    plt.plot(signal1, label="Signal 1 (Reference)", alpha=0.7)
    plt.plot(signal2, label="Signal 2 (Original)", alpha=0.7)
    plt.plot(aligned_signal2, label="Signal 2 (Aligned)", alpha=0.7, linestyle="--")
    plt.legend()
    plt.title("Signal Synchronization Debug")
    plt.show()

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
    """Synchronize two signals using cross-correlation."""
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

def preprocess_signal(signal, cutoff=0.5, fs=30, savgol_window=7, savgol_poly=2):
    """Preprocess the signal by applying Butterworth filtering and Savitzky-Golay smoothing."""
    filtered_signal = butter_highpass_filter(signal, cutoff=cutoff, fs=fs)
    smoothed_signal = smooth_data_with_savgol(filtered_signal, window_size=savgol_window, poly_order=savgol_poly)
    return smoothed_signal

def detect_peaks_dynamic(sensor_data, predicted_style, base_prominence, fs):
    """Enhanced peak detection dynamically adjusting for stroke styles."""
    params = stroke_axis_params.get(predicted_style, {})
    acc_axes = params.get("acc_axes", [])
    gyro_axes = params.get("gyro_axes", [])
    prominence = params.get("prominence", base_prominence)
    distance = params.get("distance", 10)

    peaks = {"acc": {}, "gyro": {}}

    for axis in acc_axes:
        peaks["acc"][axis], _ = scipy.signal.find_peaks(
            sensor_data["acc"][:, axis],
            prominence=prominence,
            distance=distance,
        )
    
    for axis in gyro_axes:
        peaks["gyro"][axis], _ = scipy.signal.find_peaks(
            sensor_data["gyro"][:, axis],
            prominence=prominence,
            distance=distance,
        )

    return peaks

def count_strokes_by_style(sensor_windows, predicted_styles, base_prominence=0.75, user_number=None, file_name=None, plot=False):
    stroke_counts = {label: 0 for label in label_names_abb}
    plotter = RealTimePlotter(user_number=user_number, file_name=file_name) if plot else None

    global_acc_peaks = set()
    global_gyro_peaks = set()

    for i, window in enumerate(sensor_windows):
        # Extract accelerometer and gyroscope data
        acc_data = window[:, :3]  # X, Y, Z accelerometer axes
        gyro_data = window[:, 3:6]  # X, Y, Z gyroscope axes

        # Preprocess accelerometer data
        acc_filtered = np.array([preprocess_signal(acc_data[:, j]) for j in range(3)]).T
        acc_magnitude = calculate_magnitude(acc_filtered)

        # Preprocess gyroscope data
        gyro_filtered = np.array([preprocess_signal(gyro_data[:, j]) for j in range(3)]).T
        gyro_magnitude = calculate_magnitude(gyro_filtered)

        # Synchronize gyroscope with accelerometer
        gyro_synced = np.array([
            synchronize_signals(acc_filtered[:, j], gyro_filtered[:, j]) for j in range(3)
        ]).T

        # Predict stroke style and confidence
        predicted_style = np.argmax(predicted_styles[i])
        style_confidence = np.max(predicted_styles[i])

        if predicted_style != 0 and style_confidence > 0.8:
            # Select relevant axes for current stroke type
            stroke_params = stroke_axis_params.get(predicted_style, {})
            acc_axes = stroke_params.get("acc_axes", [])
            gyro_axes = stroke_params.get("gyro_axes", [])

            # Create sensor data dictionary for `detect_peaks_dynamic`
            processed_sensor_data = {
                "acc": acc_filtered,
                "gyro": gyro_synced,
            }

            # Perform dynamic peak detection
            peaks = detect_peaks_dynamic(processed_sensor_data, predicted_style, base_prominence, fs=30)

            valid_strokes = 0

            # Match peaks for valid strokes
            for acc_axis in acc_axes:
                acc_peaks = peaks["acc"].get(acc_axis, [])
                for gyro_axis in gyro_axes:
                    gyro_peaks = peaks["gyro"].get(gyro_axis, [])
                    for acc_peak in acc_peaks:
                        global_acc_peak = acc_peak + i * 30  # Global index for accelerometer peak
                        for gyro_peak in gyro_peaks:
                            global_gyro_peak = gyro_peak + i * 30  # Global index for gyroscope peak
                            if abs(acc_peak - gyro_peak) <= 5:  # Synchronization tolerance
                                if global_acc_peak not in global_acc_peaks and global_gyro_peak not in global_gyro_peaks:
                                    valid_strokes += 1
                                    global_acc_peaks.add(global_acc_peak)
                                    global_gyro_peaks.add(global_gyro_peak)
                                    break

            # Update stroke counts
            stroke_counts[label_names_abb[predicted_style]] += valid_strokes

            # Debugging output
            print(f"Window {i}: Counted {valid_strokes} valid strokes for style {label_names_abb[predicted_style]}")

            # Plotting
            if valid_strokes >= 1:
                acc_magnitude_smooth = smooth_data_with_savgol(acc_magnitude)
                gyro_magnitude_smooth = smooth_data_with_savgol(calculate_magnitude(gyro_synced))
                plot_data_with_peaks(
                    acc_magnitude_smooth,
                    gyro_magnitude_smooth,
                    {axis: peaks["acc"].get(axis, []) for axis in acc_axes},
                    {axis: peaks["gyro"].get(axis, []) for axis in gyro_axes},
                    predicted_style,
                    i,
                    user_number=user_number,
                    file_name=file_name
                )

            if plot:
                plotter.update_plot(acc_magnitude, gyro_magnitude, predicted_style, acc_peaks, gyro_peaks, style_confidence, i)

    if plot:
        plotter.close_plot()

    return stroke_counts


def main():
    data_path = 'data/processed_30Hz_relabeled'
    results_path = 'tutorial_save_path_epoch60/'
    
    #with open(os.path.join(results_path, '22/data_parameters.pkl'), 'rb') as f:
     #   data_parameters = pickle.load(f)[0]
    
    #users = data_parameters['users']
        # User whose model we want to load
    user = '1'

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