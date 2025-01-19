import os
import pickle
import numpy as np
import tensorflow as tf
import learning_data
import utils
import scipy.signal
import matplotlib.pyplot as plt
from collections import deque

# Define label names for the stroke styles
label_names_abb = ['Null', 'Fr', 'Br', 'Ba', 'Bu']

def extract_window_data_and_predictions(swimming_data, model, data_parameters, user, rec):
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
    windows = windows.reshape((windows.shape[0], windows.shape[1], windows.shape[2], 1))
    
    # Predict labels
    y_pred_windows = model.predict(windows)
    
    return {
        'normalized_windows': windows,
        'sensor_windows': sensor_windows,
        'true_windows_cat': y_true_windows,
        'true_windows_maj': y_true_windows_maj,
        'predicted_windows': y_pred_windows
    }

def detect_peaks_with_advanced_threshold(sensor_data, predicted_style, base_prominence=0.5):
    # Get style-specific parameters dynamically
    style_params = determine_style_params(predicted_style)
    style_prominence_mult = style_params['prominence_mult']
    min_distance = style_params['min_dist']
    
    # Calculate local statistics
    local_mean = np.mean(sensor_data)
    local_std = np.std(sensor_data)
    signal_range = np.ptp(sensor_data)
    
    # Calculate dynamic prominence
    if local_mean > 0:
        dynamic_prominence = base_prominence * (local_std / local_mean) * signal_range
    else:
        dynamic_prominence = base_prominence * signal_range
    
    # Adjust prominence with style-specific multiplier
    adjusted_prominence = dynamic_prominence * style_prominence_mult
    
    # More stringent prominence bounds
    adjusted_prominence = max(adjusted_prominence, base_prominence * 0.5)  # Lower minimum
    adjusted_prominence = min(adjusted_prominence, base_prominence * 1.5)  # Lower maximum
    
    # Detect peaks
    peaks, properties = scipy.signal.find_peaks(
        sensor_data,
        prominence=adjusted_prominence,
        distance=min_distance,
        width=(2, 10)  # Add width constraint
    )
    
    return peaks, properties, adjusted_prominence

def determine_style_params(predicted_style):
    # Adjusted parameters for each style
    style_params = {
        0: {'min_dist': None, 'prominence_mult': 0},    # Null/Turn
        1: {'min_dist': 30, 'prominence_mult': 1.2},    # Freestyle
        2: {'min_dist': 45, 'prominence_mult': 1.0},    # Breaststroke
        3: {'min_dist': 40, 'prominence_mult': 1.1},    # Backstroke
        4: {'min_dist': 35, 'prominence_mult': 1.3}     # Butterfly
    }
    return style_params.get(predicted_style, {'min_dist': 30, 'prominence_mult': 1.0})

def plot_data_with_peaks(acc_magnitude, gyro_magnitude, acc_peaks, gyro_peaks, predicted_style, window_index, user_number=None, file_name=None):

    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(acc_magnitude, label='Accelerometer Magnitude')
    plt.plot(acc_peaks, acc_magnitude[acc_peaks], 'ro', label='Detected Peaks')
    plt.title(f'User: {user_number}, File: {file_name}  '
                         f'Accelerometer Magnitude - Window {window_index} '
                         f'(Style: {label_names_abb[predicted_style]})')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(gyro_magnitude, label='Gyroscope Magnitude')
    plt.plot(gyro_peaks, gyro_magnitude[gyro_peaks], 'ro', label='Detected Peaks')
    plt.title(f'Window {window_index} - Gyroscope')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    plt.waitforbuttonpress()

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
        for ax in self.ax[:2]:
            ax.set_xlim(0, self.history_size)
            ax.grid(True)
            ax.set_ylim(0, 5)
        
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
        
    def update_plot(self, acc_data, gyro_data, predicted_style, acc_peaks, gyro_peaks, 
                 style_confidence, window_index):
        self.history_acc.extend(acc_data)
        self.history_gyro.extend(gyro_data)
    
        acc_hist = np.array(self.history_acc)
        gyro_hist = np.array(self.history_gyro)
    
        for ax in self.ax:
            ax.clear()
            ax.grid(True)
    
        style_color = self.style_colors[predicted_style]
    
        # Plot main signals
        self.ax[0].plot(acc_hist, color=style_color, alpha=0.7)
        self.ax[1].plot(gyro_hist, color=style_color, alpha=0.7)
    
        # Plot peaks with correct positioning
        if len(acc_peaks) > 0:
            current_window_start = len(acc_hist) - len(acc_data)
            adjusted_acc_peaks = [p + current_window_start for p in acc_peaks]
            valid_peaks = [p for p in adjusted_acc_peaks if p < len(acc_hist)]
            if valid_peaks:
                self.ax[0].plot(valid_peaks, acc_hist[valid_peaks], 'ro', markersize=8, label='Acc Peaks')
    
        if len(gyro_peaks) > 0:
            current_window_start = len(gyro_hist) - len(gyro_data)
            adjusted_gyro_peaks = [p + current_window_start for p in gyro_peaks]
            valid_peaks = [p for p in adjusted_gyro_peaks if p < len(gyro_hist)]
            if valid_peaks:
                self.ax[1].plot(valid_peaks, gyro_hist[valid_peaks], 'ro', markersize=8, label='Gyro Peaks')

        # Update titles with user number and file name
        self.ax[0].set_title(f'User: {self.user_number}, File: {self.file_name}  '
                         f'Accelerometer Magnitude - Window {window_index} '
                         f'(Style: {label_names_abb[predicted_style]}, Confidence: {style_confidence:.2f})')
        self.ax[1].set_title('Gyroscope Magnitude')
    
        x = np.arange(len(label_names_abb))
        counts = [self.stroke_counts[label] for label in label_names_abb]
        bars = self.ax[2].bar(x, counts, color=[self.style_colors[i] for i in range(len(label_names_abb))])
    
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            self.ax[2].text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(count)}',
                        ha='center', va='bottom')
    
        self.ax[2].set_title('Cumulative Stroke Counts')
        self.ax[2].set_xticks(x)
        self.ax[2].set_xticklabels(label_names_abb)
    
        max_count = max(counts) if counts else 10
        self.ax[2].set_ylim(0, max_count * 1.1)
    
        for ax in self.ax[:2]:
            ax.set_xlim(0, self.history_size)
            ax.set_ylim(0, max(np.max(acc_hist), np.max(gyro_hist)) * 1.1)
    
        plt.pause(0.01)
        
    def close_plot(self):
        plt.ioff()
        plt.close(self.fig)

def calculate_magnitude(data):
    """Calculate the magnitude from the sensor data."""
    return np.sqrt(np.sum(data**2, axis=1))

def calculate_derivative(magnitude):
    """Calculate the derivative of the magnitude."""
    return np.diff(magnitude, prepend=magnitude[0])

def smooth_data(data, window_size=5):
    """Smooth the data using a moving average."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

def find_peaks_in_rapid_changes(magnitude, rapid_changes, prominence=0.5, min_distance=20):
    """Find peaks within regions of rapid changes."""
    peaks, _ = scipy.signal.find_peaks(magnitude, prominence=prominence, distance=min_distance)
    # Filter peaks to only those that are within rapid change indices
    filtered_peaks = [p for p in peaks if p in rapid_changes]
    return filtered_peaks


def detect_strokes_with_features(sensor_windows, predicted_styles, base_prominence=0.2, user_number=None, file_name=None, plot=False):
    stroke_counts = {label: 0 for label in label_names_abb}
    plotter = RealTimePlotter(user_number=user_number, file_name=file_name) if plot else None

    # Initialize a set to track global peak indices
    global_acc_peaks = set()
    global_gyro_peaks = set()

    for i, window in enumerate(sensor_windows):
        # Calculate features
        acc_magnitude = calculate_magnitude(window[:, :3])
        gyro_magnitude = calculate_magnitude(window[:, 3:6])
        acc_derivative = calculate_derivative(acc_magnitude)
        gyro_derivative = calculate_derivative(gyro_magnitude)
        
        # Smooth the magnitude data
        acc_magnitude_smooth = smooth_data(acc_magnitude)
        gyro_magnitude_smooth = smooth_data(gyro_magnitude)
        
        predicted_style = np.argmax(predicted_styles[i])
        style_confidence = np.max(predicted_styles[i])
         
        # Initialize peaks as empty arrays
        acc_peaks = np.array([])
        gyro_peaks = np.array([])

        if predicted_style != 0 and style_confidence > 0.9:
            # Identify rapid changes
            acc_rapid_changes = np.where(np.abs(acc_derivative) > np.std(acc_derivative))[0]
            gyro_rapid_changes = np.where(np.abs(gyro_derivative) > np.std(gyro_derivative))[0]
            
            # Find peaks in rapid change regions
            acc_peaks = find_peaks_in_rapid_changes(acc_magnitude_smooth, acc_rapid_changes, prominence=base_prominence)
            gyro_peaks = find_peaks_in_rapid_changes(gyro_magnitude_smooth, gyro_rapid_changes, prominence=base_prominence)
            
            # Filter peaks based on additional criteria
            acc_peaks = [p for p in acc_peaks if acc_magnitude_smooth[p] > np.percentile(acc_magnitude_smooth, 80)]
            gyro_peaks = [p for p in gyro_peaks if gyro_magnitude_smooth[p] > np.percentile(gyro_magnitude_smooth, 80)]
            
            print(f"Window {i}: Detected {len(acc_peaks)} acc peaks, {len(gyro_peaks)} gyro peaks")
            
            # Match peaks and count strokes with a minimum time threshold
            valid_strokes = 0
            used_acc_peaks = set()
            used_gyro_peaks = set()
            last_stroke_time = -40  # Initialize to a negative value
            
            for acc_peak in acc_peaks:
                global_acc_peak = acc_peak + i * 30  # Calculate global index
                for gyro_peak in gyro_peaks:
                    global_gyro_peak = gyro_peak + i * 30  # Calculate global index
                    if abs(acc_peak - gyro_peak) <= 2:  # Matching peaks
                        # Check if the time since the last stroke is sufficient and if it's a new peak
                        if (acc_peak - last_stroke_time) > 40 and global_acc_peak not in global_acc_peaks and global_gyro_peak not in global_gyro_peaks:
                            valid_strokes += 1
                            global_acc_peaks.add(global_acc_peak)
                            global_gyro_peaks.add(global_gyro_peak)
                            last_stroke_time = acc_peak  # Update the last stroke time
                            break
            
            # Update stroke counts
            stroke_counts[label_names_abb[predicted_style]] += valid_strokes
            if plot:
                plotter.stroke_counts[label_names_abb[predicted_style]] += valid_strokes
            
            print(f"Window {i}: Counted {valid_strokes} valid strokes")
            # Visualize the data and detected peaks for problematic windows
            if valid_strokes >= 1:  # Replace with your expected count condition
                plot_data_with_peaks(acc_magnitude_smooth, gyro_magnitude_smooth, acc_peaks, gyro_peaks, predicted_style,  i, user_number, file_name)       

        if plot:
            plotter.update_plot(acc_magnitude, gyro_magnitude, predicted_style, 
                                acc_peaks, gyro_peaks, style_confidence, i)
    
    if plot:
        plotter.close_plot()
    
    return stroke_counts

def main():
    data_path = 'data/processed_30Hz_relabeled'
    results_path = 'tutorial_save_path_epoch60/'
    
    # User whose model we want to load
    user = '18'

    # Get the data parameters used for loading
    with open(os.path.join(results_path, user, 'data_parameters.pkl'), 'rb') as f:
        data_parameters = pickle.load(f)[0] 
    
    #users = data_parameters['users']
    
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
    
    # Process only the first user for testing
    #user = users[0]
    user_number = 1
    print(f"Working on User {user_number}: {user}.")
    
    model = tf.keras.models.load_model(os.path.join(results_path, user, 'model_best.keras'), 
                                       compile=False)
    
    user_predictions = {}
    
    for rec in swimming_data.data_dict['original'][user].keys():
        print(f"Processing recording: {rec}")
        file_name = rec
        
        window_data = extract_window_data_and_predictions(swimming_data, model, 
                                                          data_parameters, user, rec)
        
        stroke_counts = detect_strokes_with_features(
            window_data['sensor_windows'], 
            window_data['predicted_windows'], 
            base_prominence=0.5,
            user_number=user_number,
            file_name=rec,
            plot=False  # Set to False if you don't want to plot
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
