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
    1: {"width": 3, "prominence": 1.5, "distance": 55},  # Freestyle
    2: {"width": 3, "prominence": 1.6, "distance": 50},  # Breaststroke
    3: {"width": 3, "prominence": 1.1, "distance": 60},  # Backstroke
    4: {"width": 3, "prominence": 1.6, "distance": 45},  # Butterfly
}

def plot_data_with_peaks(sensor_data, acc_peaks, gyro_peaks, predicted_style, row_index, user_number=None, file_name=None):
    """
    Plot accelerometer and gyroscope data with detected peaks.
    """
    # Debug print
    print(f"Plotting peaks for style: {predicted_style} at Row: {row_index}")
    print("Received acc_peaks:", acc_peaks)
    print("Received gyro_peaks:", gyro_peaks)

    # Create figure
    fig, axs = plt.subplots(2, 2, figsize=(12, 6))
    fig.suptitle(
        f"User: {user_number}, File: {file_name}, Row: {row_index} (Style: {label_names_abb[predicted_style]})",
        fontsize=16,
    )

    # Plot raw accelerometer axes
    axs[0, 0].plot(sensor_data["acc_data"], label='Acc X', color="blue")
    axs[0, 0].set_title("Accelerometer X")
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    axs[0, 1].plot(sensor_data["acc_data"], label='Acc Y', color="green")
    axs[0, 1].set_title("Accelerometer Y")
    axs[0, 1].grid(True)
    axs[0, 1].legend()

    # Plot style-specific accelerometer data
    acc_label = sensor_data["style_acc_peak_key"]
    axs[0, 1].plot(sensor_data["style_acc"], label=acc_label, color="purple")
    if acc_label in acc_peaks:
        peak_indices = acc_peaks[acc_label]
        axs[0, 1].plot(
            peak_indices,
            sensor_data["style_acc"][peak_indices],
            "ro",
            label=f"{acc_label} Peaks"
        )
    axs[0, 1].set_title(f"Style-Specific {acc_label}")
    axs[0, 1].grid(True)
    axs[0, 1].legend(loc='best')

    # Plot raw gyroscope axes
    axs[1, 0].plot(sensor_data["gyro_data"], label='Gyro X', color="orange")
    axs[1, 0].set_title("Gyroscope X")
    axs[1, 0].grid(True)
    axs[1, 0].legend()

    # Plot style-specific gyroscope data
    gyro_label = sensor_data["style_gyro_peak_key"]
    axs[1, 1].plot(sensor_data["style_gyro"], label=gyro_label, color="red")
    if gyro_label in gyro_peaks:
        peak_indices = gyro_peaks[gyro_label]
        axs[1, 1].plot(
            peak_indices,
            sensor_data["style_gyro"][peak_indices],
            "ro",
            label=f"{gyro_label} Peaks"
        )
    axs[1, 1].set_title(f"Style-Specific {gyro_label}")
    axs[1, 1].grid(True)
    axs[1, 1].legend(loc='best')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def calculate_magnitude(data):
    """Calculate the magnitude from the sensor data."""
    return np.sqrt(np.sum(data**2, axis=1))

def preprocess_sensor_data_row(raw_row, style):
    """
    Preprocess a single row of accelerometer and gyroscope data for peak detection.

    Parameters:
    -----------
    raw_row : ndarray
        Sensor data row containing accelerometer and gyroscope data.
    label : int
        Label for the current row.

    Returns:
    --------
    dict
        Processed sensor data for the row.
    """
    acc_data = raw_row[2:5]  # Assuming ACC_0, ACC_1, ACC_2 are the first three columns
    gyro_data = raw_row[5:8]  # Assuming GYRO_0, GYRO_1, GYRO_2 are the next three columns

    # Calculate raw magnitudes
    acc_magnitude = calculate_magnitude(acc_data[np.newaxis, :])[0]
    gyro_magnitude = calculate_magnitude(gyro_data[np.newaxis, :])[0]

    # Style-specific processing
    sensor_data = {
        "acc_data": acc_data,
        "gyro_data": gyro_data,
        "acc_magnitude": acc_magnitude,
        "gyro_magnitude": gyro_magnitude,
        "style": style,
        "style_acc": None,
        "style_gyro": None,
        "style_acc_peak_key": None,
        "style_gyro_peak_key": None,
        "row_idx": raw_row[0]  # Assuming the first column is the row index
    }

    if style == 1:  # Freestyle
        acc_y_neg_data = -acc_data[1]  # Y Axis
        gyro_z_pos_data = gyro_data[2]  # Z axis
        sensor_data.update({
            "style_acc": acc_y_neg_data,
            "style_gyro": gyro_z_pos_data,
            "style_acc_peak_key": "acc_y_negative",
            "style_gyro_peak_key": "gyro_z_positive"
        })

    elif style == 2:  # Breaststroke
        gyro_z_neg_data = -gyro_data[2]  # Z axis
        sensor_data.update({
            "style_acc": acc_magnitude,
            "style_gyro": gyro_z_neg_data,
            "style_acc_peak_key": "acc_magnitude",
            "style_gyro_peak_key": "gyro_z_negative"
        })

    elif style == 3:  # Backstroke
        acc_z_pos_data = acc_data[2]  # Z axis
        gyro_y_pos_data = gyro_data[1]  # Y axis
        sensor_data.update({
            "style_acc": acc_z_pos_data,
            "style_gyro": gyro_y_pos_data,
            "style_acc_peak_key": "acc_z_positive",
            "style_gyro_peak_key": "gyro_y_positive"
        })

    elif style == 4:  # Butterfly
        gyro_y_neg_data = -gyro_data[1]  # Negated Y axis
        sensor_data.update({
            "style_acc": acc_magnitude,
            "style_gyro": gyro_y_neg_data,
            "style_acc_peak_key": "acc_magnitude_data",
            "style_gyro_peak_key": "gyro_y_negative"
        })

    return sensor_data


def detect_peaks(sensor_data_list, base_prominence=0.5, distance=5):
    """
    Detect peaks for accelerometer and gyroscope data based on the processed sensor data.

    Parameters:
    -----------
    sensor_data_list : list of dicts
        List containing processed sensor data for valid rows.
    base_prominence : float
        Base prominence value for peak detection.
    distance : int
        Minimum distance between peaks.

    Returns:
    --------
    acc_peaks : dict
        Detected peaks in the accelerometer data.
    gyro_peaks : dict
        Detected peaks in the gyroscope data.
    """

    acc_peaks = {}
    gyro_peaks = {}

        # Aggregate data by style
    style_data = {1: {"acc": [], "gyro": []},
                  2: {"acc": [], "gyro": []},
                  3: {"acc": [], "gyro": []},
                  4: {"acc": [], "gyro": []}}
    
    for sensor_data in sensor_data_list:
        style = int(sensor_data["style"])
        if style in style_data:
            style_data[style]["acc"].append(sensor_data["style_acc"])
            style_data[style]["gyro"].append(sensor_data["style_gyro"])

    def calculate_dynamic_prominence(signal, base_prominence=0.5, style=None):
        """
        Calculate dynamic prominence with refined adjustments for swimming styles.

        Parameters:
        -----------
        signal : ndarray
            Input signal for peak detection.
        base_prominence : float
            Base prominence value.
        style : int, optional
            Swimming style identifier for style-specific tuning.

        Returns:
        --------
        float
            Dynamic prominence for peak detection.
        """
        # Calculate metrics
        signal_range = np.ptp(signal)  # Peak-to-peak range
        local_std = np.std(signal, ddof=1)
        iqr = np.subtract(*np.percentile(signal, [75, 25]))

        # Style-specific scaling factors
        if style == 1:  # Freestyle
            range_scale = 0.01
            std_scale = 0.08
            iqr_scale = 0.05
        elif style == 2:  # Breaststroke
            range_scale = 0.03
            std_scale = 0.08
            iqr_scale = 0.06
        elif style == 3:  # Backstroke
            range_scale = 0.025
            std_scale = 0.06
            iqr_scale = 0.07
        elif style == 4:  # Butterfly
            range_scale = 0.03
            std_scale = 0.1
            iqr_scale = 0.08
        else:
            range_scale = 0.02
            std_scale = 0.08
            iqr_scale = 0.05

        # Calculate dynamic prominence
        dynamic_prominence = (
            base_prominence +
            (range_scale * np.log1p(signal_range)) +
            (std_scale * np.sqrt(local_std)) +
            (iqr_scale * np.sqrt(iqr))
        )

        # Clipping bounds based on style
        lower_bound = max(0.5, 0.1 * signal_range)
        upper_bound = min(7.0, 0.6 * signal_range)
        dynamic_prominence = np.clip(dynamic_prominence, lower_bound, upper_bound)

        # Debug output
        if DEBUG_DETECT_PEAKS:
            print(f"Signal analysis({label_names_abb[style]}):")
            print(f"  Range: {signal_range:.4f}")
            print(f"  Standard Deviation: {local_std:.4f}")
            print(f"  Interquartile Range: {iqr:.4f}")
            print(f"  Log(Range + 1): {np.log1p(signal_range):.4f}")
            print(f"  Sqrt(STD): {np.sqrt(local_std):.4f}")
            print(f"  Sqrt(IQR): {np.sqrt(iqr):.4f}")
            print(f"Base prominence: {base_prominence:.4f}")            
            print(f"Calculated prominence: {dynamic_prominence:.4f}")

        return dynamic_prominence

    # Process each style's aggregated data
    for style, data in style_data.items():
        if not data["acc"] or not data["gyro"]:
            continue  # Skip if there's no data for this style

        # Get style-specific peak detection parameters
        style_params = stroke_axis_params[style]
        base_prominence = style_params["prominence"]
        distance = style_params["distance"]
        width = style_params["width"]
        
        # Convert lists to arrays
        acc_signal = np.array(data["acc"])
        gyro_signal = np.array(data["gyro"])

        if DEBUG_DETECT_PEAKS:  
            print("\nCalculating prominence from processed accelerometer data:")
        acc_prominence = calculate_dynamic_prominence(acc_signal, base_prominence, style)
        if DEBUG_DETECT_PEAKS:  
            print("\nCalculating prominence from processed gyroscope data:")
        gyro_prominence = calculate_dynamic_prominence(gyro_signal, base_prominence, style)

        # Debugging output for prominence
        if DEBUG_DETECT_PEAKS:  
            print(f"[DEBUG] Calculated Acc Prominence: {acc_prominence}")
            print(f"[DEBUG] Calculated Gyro Prominence: {gyro_prominence}")

        # Detect peaks
        acc_peaks_key = f"style_{style}_acc"
        gyro_peaks_key = f"style_{style}_gyro"

        # Detect peaks in accelerometer data
        acc_peaks[acc_peaks_key] = scipy.signal.find_peaks(
            acc_signal,
            prominence=acc_prominence,
            #width=width,
            distance=distance
        )[0]

        # Detect peaks in gyroscope data
        gyro_peaks[gyro_peaks_key] = scipy.signal.find_peaks(
            gyro_signal,
            prominence=gyro_prominence,
            #width=width,
            distance=distance
        )[0]

        # Visualization of the detected peaks
        if DEBUG_PLOT_PEAKS:
            plt.figure(figsize=(12, 6))

            # Accelerometer Plot
            plt.subplot(2, 1, 1)
            plt.plot(acc_signal, label='Accelerometer Signal', color='blue')

            # Plot detected accelerometer peaks
            acc_peak_indices = acc_peaks[acc_peaks_key]
            plt.plot(acc_peak_indices, acc_signal[acc_peak_indices], ".", label='Detected Peaks', color='red')

            # Annotate each detected accelerometer peak with its row_idx
            for peak_idx in acc_peak_indices:
                row_idx = sensor_data["row_idx"]
                plt.text(peak_idx, acc_signal[peak_idx], f"{row_idx}", color='black', fontsize=8, ha='left', va='bottom')

            plt.title(f"Row: {sensor_data['row_idx']} Accelerometer Signal with Detected Peaks")
            plt.legend()

            # Gyroscope Plot
            plt.subplot(2, 1, 2)
            plt.plot(gyro_signal, label='Gyroscope Signal', color='orange')

            # Plot detected gyroscope peaks
            gyro_peak_indices = gyro_peaks[gyro_peaks_key]
            plt.plot(gyro_peak_indices, gyro_signal[gyro_peak_indices], ".", label='Detected Peaks', color='red')

            # Annotate each detected gyroscope peak with its row_idx
            for peak_idx in gyro_peak_indices:
                row_idx = sensor_data["row_idx"]
                plt.text(peak_idx, gyro_signal[peak_idx], f"{row_idx}", color='black', fontsize=8, ha='left', va='bottom')

            plt.title(f"Row: {sensor_data['row_idx']} Gyroscope Signal with Detected Peaks")
            plt.legend()

            plt.tight_layout()
            plt.show()

    # Debug print
    if DEBUG_DETECT_PEAKS:
        print("Detected acc_peaks:", acc_peaks)
        print("Detected gyro_peaks:", gyro_peaks)

    return acc_peaks, gyro_peaks


def count_strokes_by_style_row(swimming_data, user_number=None, file_name=None):
    """
    Count strokes using style-specific peak detection logic on a row-by-row basis.

    Parameters:
    -----------
    swimming_data : LearningData object
        The loaded swimming data.
    user_number : int, optional
        User identifier for plotting.
    file_name : str, optional
        File name for plotting.

    Returns:
    --------
    dict
        Stroke counts, stroke labels for each style.
    """
    stroke_counts = {label: 0 for label in label_names_abb}
    stroke_labels = np.zeros(len(swimming_data.data_dict['original'][user_number][file_name]), dtype=int)

    # Initialize lists to accumulate style-specific data
    processed_data = []

    # Process only the specified file
    df = swimming_data.data_dict['original'][user_number][file_name]

    for index, row in tqdm(df.iterrows(),
                            desc="Processing Recording",
                            leave=False):
                               
        style = row['label']
        # Skip -1, 0, 5, 6 which are unknown, rest, turns or kicks
        if style not in [1, 2, 3, 4]:
            continue

        sensor_data = preprocess_sensor_data_row(row.values, style)

        # Debugging output for the current row
        if DEBUG_COUNT_STROKES:
            print(f"[DEBUG] Processing Row {index}: Label = {style}, Sensor Data = {sensor_data}")

        processed_data.append(sensor_data)

    # Now process the accumulated data for peak detection
    acc_peaks, gyro_peaks = detect_peaks(processed_data, base_prominence=0.5, distance=5)       

    # Track unique peaks to avoid double-counting
    unique_peaks = set()

    # Count strokes based on detected peaks
    for sensor_data in processed_data:
        style = int(sensor_data["style"])
        row_idx = int(sensor_data["row_idx"])
        acc_peak_indices = acc_peaks.get(f"style_{style}_acc", [])
        gyro_peak_indices = gyro_peaks.get(f"style_{style}_gyro", [])

        valid_strokes = 0
        if style == 1:  # Freestyle
            for peak in gyro_peak_indices:
                if peak not in unique_peaks:
                    valid_strokes += 1
                    unique_peaks.add(peak)
                    stroke_labels[peak + row_idx] = 1

        elif style == 2:  # Breaststroke
            for peak in acc_peak_indices:
                if peak not in unique_peaks:
                    valid_strokes += 1
                    unique_peaks.add(peak)
                    stroke_labels[peak + row_idx] = 1

        elif style == 3:  # Backstroke
            for peak in gyro_peak_indices:
                if peak not in unique_peaks:
                    valid_strokes += 1
                    unique_peaks.add(peak)
                    stroke_labels[peak + row_idx] = 1

        elif style == 4:  # Butterfly
            for peak in acc_peak_indices:
                if peak not in unique_peaks:
                    valid_strokes += 1
                    unique_peaks.add(peak)
                    stroke_labels[peak + row_idx] = 1

        # Debugging output for detected peaks
        if DEBUG_COUNT_STROKES:
            print(f"[DEBUG] Row {index}, Style: {label_names_abb[style]}, "
                  f"Acc Peaks: {acc_peak_indices}, Gyro Peaks: {gyro_peak_indices}")

        stroke_counts[label_names_abb[style]] += valid_strokes

        # Debugging output for stroke counts
        if DEBUG_COUNT_STROKES:
            print(f"[DEBUG] Row {row_idx}, Valid Strokes: {valid_strokes}, "
                  f"Total Count for {label_names_abb[style]}: {stroke_counts[label_names_abb[style]]}")

        # Optional visualization
        if DEBUG_COUNT_STROKES_PLOTTER:
            plot_data_with_peaks(
                sensor_data,
                acc_peaks,
                gyro_peaks,
                style,
                row_index=row_idx,  # Use row index instead of window index
                user_number=user_number,
                file_name=file_name
            )

    return stroke_counts, stroke_labels


# Update the main function to use the new row-by-row counting function
def main():
    # Global debug flags can be set here
    global DEBUG_DETECT_PEAKS, DEBUG_PLOT_PEAKS, DEBUG_REAL_TIME_PLOTTER, DEBUG_SYNCHRONIZATION, DEBUG_COUNT_STROKES, DEBUG_COUNT_STROKES_PLOTTER
    
    # Turn on/off specific debug outputs
    DEBUG_DETECT_PEAKS = True
    DEBUG_PLOT_PEAKS = False
    DEBUG_REAL_TIME_PLOTTER = False
    DEBUG_SYNCHRONIZATION = False
    DEBUG_COUNT_STROKES = False
    DEBUG_COUNT_STROKES_PLOTTER = False

    data_path = '/Users/juanloya/Documents/SwimmingModelPython/swim_v2/data_copy/processed_30Hz_relabeled'
    results_path = '/Users/juanloya/Documents/SwimmingModelPython/swim_v2/tutorial_save_path_epoch60_mag/'
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
    
    swimming_data.preprocess_filtering(
        butter_params={'cutoff': 0.3, 'fs': 30, 'order': 4},
        savgol_params={'window_size': 5, 'poly_order': 3}, 
        apply_butter=True, apply_savgol=True
    )    
    
    user_number = 1
    print(f"Working on User {user_number}: {user}.")   
        
    user_predictions = {}
        
    recordings = list(swimming_data.data_dict['original'][user].keys())
    
    for rec in tqdm(recordings, 
                    desc=f"Processing Recordings for User {user}", 
                    total=len(recordings),  
                    unit="recording",       
                    colour='green',         
                    ncols=100,              
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):
        print(f"Processing recording: {rec}")
            
        stroke_counts, stroke_labels = count_strokes_by_style_row(
                swimming_data=swimming_data, 
                user_number=user,
                file_name=rec
            )
            
        user_predictions[rec] = {
            'stroke_counts': stroke_counts,
            'stroke_labels': stroke_labels
        }
        sub_dir_path = os.path.join(output_dir, user)
        base_name, _ = os.path.splitext(rec)
        new_file_name = f"{base_name}_updated.csv"

        os.makedirs(sub_dir_path, exist_ok=True)

        dataset_length = swimming_data.data_dict['original'][user][rec].shape[0]
        if len(stroke_labels) < dataset_length:
            padding_length = dataset_length - len(stroke_labels)
            stroke_labels = np.pad(stroke_labels, (0, padding_length), 'constant', constant_values=0)

        swimming_data.data_dict['original'][user][rec]['stroke_labels'] = stroke_labels

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