import os
import pickle
import numpy as np
import tensorflow as tf
import learning_data
import utils
import scipy.signal
import matplotlib.pyplot as plt
from collections import deque
from scipy import signal
from scipy.stats import pearsonr

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


import numpy as np
import matplotlib.pyplot as plt

class StrokeStyleDetector:
    def __init__(self):
        # Define dominant axes and correlations for each stroke style
        self.style_characteristics = {
            # ... (existing code)
        }
        
        # Floating elimination parameters
        self.tf_lb = -0.2  # Lower bound for floating
        self.tf_ub = 0.2   # Upper bound for floating
        self.tf_time = 40  # Minimum time for floating (in samples)
        
        # Feature point detection parameters
        self.tb_x_lb = -0.5  # Lower bound for X-axis
        self.tb_x_ub = 0.5   # Upper bound for X-axis
        self.tb_y_lb = -0.8  # Lower bound for Y-axis
        self.tb_y_ub = 0.8   # Upper bound for Y-axis
        self.tb_z_lb = -0.8  # Lower bound for Z-axis
        self.tb_z_ub = 0.8   # Upper bound for Z-axis
        
        self.tia = 0.6       # Threshold for regular fluctuations
        self.tsg = 0.5       # Threshold for similar segments
        self.tamp = 0.3      # Threshold for amplitude difference

    def smooth_data(self, data, window_size=5):
        """Smooth the data using a moving average."""
        return np.convolve(data, np.ones(window_size)/window_size, mode='same')

    def eliminate_floating_data(self, data):
        """Eliminate floating data from sensor readings."""
        valid_mask = np.ones(data.shape[0], dtype=bool)
        n_samples = data.shape[0]
        floating_start = None
        
        for i in range(1, n_samples - 1):
            if floating_start is None:
                if (np.all(data[i-1, :] > self.tf_ub) and np.all(data[i, :] < self.tf_ub)) or \
                   (np.all(data[i-1, :] < self.tf_lb) and np.all(data[i, :] > self.tf_lb)):
                    floating_start = i
            else:
                if (np.all(data[i, :] < self.tf_ub) and np.all(data[i+1, :] > self.tf_ub)) or \
                   (np.all(data[i, :] > self.tf_lb) and np.all(data[i+1, :] < self.tf_lb)):
                    floating_end = i
                    if (floating_end - floating_start) > self.tf_time:
                        valid_mask[floating_start:floating_end] = False
                    floating_start = None

        cleaned_data = data[valid_mask]
        return cleaned_data

    def detect_feature_points_for_axis(self, axis_data, lb, ub):
        """Detect feature points for a single axis."""
        feature_points = []
        for i in range(1, len(axis_data)):
            if (axis_data[i-1] < ub and axis_data[i] >= ub) or (axis_data[i-1] > lb and axis_data[i] <= lb):
                feature_points.append(i)
        return feature_points

    def detect_feature_points(self, data, style):
        """Detect feature points in sensor data."""
        acc_x_points = self.detect_feature_points_for_axis(data[:, 0], self.tb_x_lb, self.tb_x_ub)
        acc_y_points = self.detect_feature_points_for_axis(data[:, 1], self.tb_y_lb, self.tb_y_ub)
        acc_z_points = self.detect_feature_points_for_axis(data[:, 2], self.tb_z_lb, self.tb_z_ub)
        gyro_x_points = self.detect_feature_points_for_axis(data[:, 3], self.tb_x_lb, self.tb_x_ub)
        gyro_y_points = self.detect_feature_points_for_axis(data[:, 4], self.tb_y_lb, self.tb_y_ub)
        gyro_z_points = self.detect_feature_points_for_axis(data[:, 5], self.tb_z_lb, self.tb_z_ub)

        # Print the number of feature points detected
        print(f"Detected feature points: Acc X: {len(acc_x_points)}, Acc Y: {len(acc_y_points)}, Acc Z: {len(acc_z_points)}")
        print(f"Detected feature points: Gyro X: {len(gyro_x_points)}, Gyro Y: {len(gyro_y_points)}, Gyro Z: {len(gyro_z_points)}")

        # Combine feature points
        fx = acc_x_points
        fy = acc_y_points if style in ['Fr', 'Ba', 'Bu'] else gyro_y_points
        fz = acc_z_points if style in ['Fr', 'Ba', 'Bu'] else gyro_z_points
        return fx, fy, fz

    def analyze_strokes(self, sensor_data, fx, fy, fz):
        """
        Analyze strokes based on feature points using the detailed method from the paper.
    
        Parameters:
        - sensor_data: Full sensor data array
        - fx, fy, fz: Feature points for different axes
    
        Returns:
        - Number of strokes detected
        """
        def cal_amp(data, tfi, tfj):
            """Calculate amplitude value within a segment."""
            if tfj <= tfi or tfi >= len(data) or tfj > len(data):
                return float('inf')
            
            p_prime = (tfi + tfj) // 2
            x1 = np.max(data[tfi:p_prime]) - np.min(data[tfi:p_prime])
            x2 = np.max(data[p_prime:tfj]) - np.min(data[p_prime:tfj])
            return abs(x1 - x2)

        def rho_ia(data, tfi, tfj):
            """Calculate correlation coefficient for a segment."""
            if tfj <= tfi or tfi >= len(data) or tfj > len(data):
                return 0
            
            segment = data[tfi:tfj]
            mid_point = len(segment) // 2
            s0 = segment[:mid_point]
            s1 = segment[mid_point:mid_point + len(s0)]
        
            if len(s0) < 3 or len(s1) < 3:  # Minimum length for meaningful correlation
                return 0
            
            return np.corrcoef(s0, s1)[0, 1]

        def rho_sg(data, tfi, tfj, tfk, Td):
            """Calculate correlation coefficient between two nearby segments."""
            if tfj <= tfi or tfk <= tfj or tfi >= len(data) or tfk > len(data):
                return 0
            
            if (tfj - tfi) < 5 or (tfk - tfj) < 5:
                return 0
            
            segment1 = data[tfi:tfj]
            segment2 = data[tfj:tfk]
        
            segment1 = (segment1 - np.mean(segment1)) / (np.std(segment1) + 1e-6)
            segment2 = (segment2 - np.mean(segment2)) / (np.std(segment2) + 1e-6)
        
            target_length = Td
            t1 = np.linspace(0, 1, len(segment1))
            t2 = np.linspace(0, 1, len(segment2))
            t_interp = np.linspace(0, 1, target_length)
        
            segment1_interp = np.interp(t_interp, t1, segment1)
            segment2_interp = np.interp(t_interp, t2, segment2)
        
            return np.corrcoef(segment1_interp, segment2_interp)[0, 1]

        # Parameters for stroke detection
        Tia = 0.5   # Threshold for intra-segment correlation
        Tsg = 0.5   # Threshold for inter-segment correlation
        Td = 100    # Predefined segment size
        Tamp = 1.5  # Amplitude threshold
        min_feature_distance = 10  # Minimum distance between feature points

        # Add segment size constraints
        min_segment_size = 10  # Minimum segment size in samples
        max_segment_size = 60  # Maximum segment size in samples

        all_features = sorted(set(fx + fy + fz))
        filtered_features = []
    
        for feat in all_features:
            if not filtered_features or feat - filtered_features[-1] >= min_feature_distance:
                filtered_features.append(feat)
    
        stroke_count = 0
        valid_segments = []
    
        # Modified stroke detection logic
        for axis in range(min(3, sensor_data.shape[1])):
            axis_data = sensor_data[:, axis]
        
            for i in range(len(filtered_features) - 2):
                tfi = filtered_features[i]
                tfj = filtered_features[i + 1]
                tfk = filtered_features[i + 2]
            
                # Check segment size constraints
                if (tfj - tfi < min_segment_size or 
                    tfk - tfj < min_segment_size or 
                    tfj - tfi > max_segment_size or 
                    tfk - tfj > max_segment_size):
                    continue
            
                # Calculate correlations and amplitude
                ia_corr = abs(rho_ia(axis_data, tfi, tfj))  # Use absolute correlation
                sg_corr = abs(rho_sg(axis_data, tfi, tfj, tfk, Td))  # Use absolute correlation
                amp_diff = cal_amp(axis_data, tfi, tfj)
            
                # Modified stroke detection conditions
                if (ia_corr > Tia and 
                    sg_corr > Tsg and 
                    amp_diff <= Tamp and
                    (tfi, tfj) not in valid_segments):

                    # Additional validation check
                    segment_length = tfj - tfi
                    print(f"Checking segment: {tfi}-{tfj}-{tfk}, ia_corr: {ia_corr:.3f}, sg_corr: {sg_corr:.3f}, amp_diff: {amp_diff:.3f}")

                    if segment_length >= min_segment_size and segment_length <= max_segment_size:
                        valid_segments.append((tfi, tfj))
                        stroke_count += 1
                        print(f"Valid stroke detected at segment {tfi}-{tfj}")

        # Normalize stroke count based on window size
        max_expected_strokes = 2  # Adjusted maximum expected strokes per window
        return min(stroke_count, max_expected_strokes)

    def plot_raw_data_with_feature_points(self, sensor_data, fx, fy, fz):
        plt.figure(figsize=(15, 10))
        plt.plot(sensor_data[:, 0], label='Acc X')
        plt.plot(sensor_data[:, 1], label='Acc Y')
        plt.plot(sensor_data[:, 2], label='Acc Z')
    
        # Highlight feature points
        plt.scatter(fx, sensor_data[fx, 0], color='red', label='Acc X Feature Points')
        plt.scatter(fy, sensor_data[fy, 1], color='green', label='Acc Y Feature Points')
        plt.scatter(fz, sensor_data[fz, 2], color='blue', label='Acc Z Feature Points')
    
        plt.title('Raw Sensor Data with Feature Points')
        plt.xlabel('Sample Index')
        plt.ylabel('Sensor Value')
        plt.legend()
        plt.grid()
        plt.show() 

    def plot_detected_strokes(self, sensor_data, fx, fy, fz, style_name):
        """
        Visualize the detected feature points and potential strokes.
    
        Parameters:
        - sensor_data: Full sensor data array
        - fx, fy, fz: Feature points for different axes
        - style_name: Swimming style name
        """
    
        # Create a figure with subplots for each axis
        fig, axs = plt.subplots(3, 2, figsize=(15, 10))
        fig.suptitle(f'Stroke Detection Analysis - {style_name} Style', fontsize=16)
    
        # Axes labels
        axis_labels = ['Acc X', 'Acc Y', 'Acc Z', 'Gyro X', 'Gyro Y', 'Gyro Z']
    
        # Plot each axis
        for i in range(6):
            row = i // 2
            col = i % 2
        
            # Select the current axis data
            axis_data = sensor_data[:, i]
        
            # Select feature points for the current axis
            if i == 0:
                feature_points = [p for p in fx if p < len(axis_data)]
            elif i == 1:
                feature_points = [p for p in fy if p < len(axis_data)]
            elif i == 2:
                feature_points = [p for p in fz if p < len(axis_data)]
            else:
                # For gyroscope axes, use the same logic as accelerometer
                feature_points = [p for p in fx if p < len(axis_data)] if i == 3 else \
                                 [p for p in fy if p < len(axis_data)] if i == 4 else \
                                 [p for p in fz if p < len(axis_data)]
        
            # Plot the axis data
            axs[row, col].plot(axis_data, label=axis_labels[i], alpha=0.7)
        
            # Highlight feature points
            if feature_points:
                axs[row, col].scatter(feature_points, axis_data[feature_points], 
                                       color='red', label='Feature Points')
        
            axs[row, col].set_title(f'{axis_labels[i]} - Feature Points')
            axs[row, col].legend()
            axs[row, col].grid(True)
    
        plt.tight_layout()
        plt.show()



def detect_strokes_with_style_characteristics(
    sensor_windows, 
    predicted_styles, 
    user_number=None, 
    file_name=None, 
    plot=False, 
    max_plots=5  # Limit number of plots
):
    stroke_counts = {label: 0 for label in label_names_abb}
    plotter = RealTimePlotter(user_number=user_number, file_name=file_name) if plot else None
    detector = StrokeStyleDetector()

    plots_generated = 0

    for i, window in enumerate(sensor_windows):
        # Split window data into accelerometer and gyroscope
        sensor_data = window
        
        # Eliminate floating data
        sensor_data_cleaned = detector.eliminate_floating_data(sensor_data)
        
        predicted_style = np.argmax(predicted_styles[i])
        style_confidence = np.max(predicted_styles[i])
        
        if predicted_style != 0 and style_confidence > 0.9:
            style_name = label_names_abb[predicted_style]
            
            # Detect feature points
            fx, fy, fz = detector.detect_feature_points(sensor_data_cleaned, style_name)
            
            # Analyze strokes
            valid_strokes = detector.analyze_strokes(sensor_data_cleaned, fx, fy, fz)
            
            # Update stroke counts
            stroke_counts[style_name] += valid_strokes
            if plot:
                plotter.stroke_counts[style_name] += valid_strokes
            
            print(f"Window {i}: Counted {valid_strokes} valid strokes for {style_name}")
            
            # Visualize detected strokes
            if valid_strokes > 1:
                detector.plot_detected_strokes(sensor_data_cleaned, fx, fy, fz, style_name)
                detector.plot_raw_data_with_feature_points(sensor_data_cleaned, fx, fy, fz)
        
        if plot:
            fx = fy = fz = []
            plotter.update_plot(sensor_data[:, :3], sensor_data[:, 3:6], predicted_style, 
                         fx, fy, fz, style_confidence, i)
    
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
        
        stroke_counts = detect_strokes_with_style_characteristics(
            window_data['sensor_windows'], 
            window_data['predicted_windows'], 
            user_number=user_number,
            file_name=file_name,
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
