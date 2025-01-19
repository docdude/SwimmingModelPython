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
    
    # Further adjust prominence based on signal characteristics
    adjusted_prominence = max(adjusted_prominence, base_prominence * 0.75)  # Ensure a minimum prominence
    adjusted_prominence = min(adjusted_prominence, base_prominence * 1.5)  # Cap the maximum prominence
    
    # Detect peaks
    peaks, properties = scipy.signal.find_peaks(
        sensor_data,
        prominence=adjusted_prominence,
        distance=min_distance,
        width=3
    )
    
    return peaks, properties, adjusted_prominence

def determine_style_params(predicted_style):
    # Example dynamic parameters for each style
    style_params = {
        0: {'min_dist': None, 'prominence_mult': 0},    # Null/Turn
        1: {'min_dist': 30, 'prominence_mult': 1.2},    # Freestyle
        2: {'min_dist': 45, 'prominence_mult': 0.8},    # Breaststroke
        3: {'min_dist': 40, 'prominence_mult': 1.0},    # Backstroke
        4: {'min_dist': 35, 'prominence_mult': 1.5}     # Butterfly
    }
    return style_params.get(predicted_style, {'min_dist': 30, 'prominence_mult': 1.0})

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
        self.ax[0].plot(acc_hist, color=style_color, alpha=0.7)
        self.ax[1].plot(gyro_hist, color=style_color, alpha=0.7)
        
        # Plot accelerometer peaks
        if len(acc_peaks) > 0:
            adjusted_peaks = acc_peaks + len(acc_hist) - len(acc_data)
            self.ax[0].plot(adjusted_peaks, acc_hist[adjusted_peaks], 'ro', markersize=5, label='Acc Peaks')
                
        # Plot gyroscope peaks
        if len(gyro_peaks) > 0:
            adjusted_peaks = gyro_peaks + len(gyro_hist) - len(gyro_data)
            self.ax[1].plot(adjusted_peaks, gyro_hist[adjusted_peaks], 'ro', markersize=5, label='Gyro Peaks')
        
        if style_confidence > 0.85:
            self.update_stroke_counts(predicted_style, acc_peaks, gyro_peaks)
        
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

def count_strokes_by_style(sensor_windows, predicted_styles, base_prominence=0.5, user_number=None, file_name=None):
    stroke_counts = {label: 0 for label in label_names_abb}
    plotter = RealTimePlotter(user_number=user_number, file_name=file_name)

    # Initialize a set to track global peak indices
    global_acc_peaks = set()
    global_gyro_peaks = set()

    for i, window in enumerate(sensor_windows):
        acc_magnitude = np.sqrt(np.sum(window[:, :3]**2, axis=1))
        gyro_magnitude = np.sqrt(np.sum(window[:, 3:6]**2, axis=1))
        
        predicted_style = np.argmax(predicted_styles[i])
        style_confidence = np.max(predicted_styles[i])
        
        acc_peaks, acc_props, acc_prominence = detect_peaks_with_advanced_threshold(
            acc_magnitude, predicted_style, base_prominence)
        gyro_peaks, gyro_props, gyro_prominence = detect_peaks_with_advanced_threshold(
            gyro_magnitude, predicted_style, base_prominence)
        
     #   plotter.update_plot(acc_magnitude, gyro_magnitude, predicted_style, 
      #                      acc_peaks, gyro_peaks, style_confidence, i)
        
        if predicted_style != 0 and style_confidence > 0.85:
            plotter.update_stroke_counts(predicted_style, acc_peaks, gyro_peaks)
    
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
    
    all_predictions = {}
    
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
            
            stroke_counts = count_strokes_by_style(
                window_data['sensor_windows'], 
                window_data['predicted_windows'], 
                base_prominence=0.5,
                user_number=user_number,
                file_name=rec
            )
            
            user_predictions[rec] = {
                'window_data': window_data,
                'stroke_counts': stroke_counts
            }
        
    all_predictions[user] = user_predictions
    
    with open('stroke_counts_results.pkl', 'wb') as f:
        pickle.dump(all_predictions, f)
    
    for user, recordings in all_predictions.items():
        print(f"\nUser: {user}")
        for rec, data in recordings.items():
            print(f"  Recording: {rec}")
            print("  Stroke Counts by Style:", data['stroke_counts'])

if __name__ == '__main__':
    main()
