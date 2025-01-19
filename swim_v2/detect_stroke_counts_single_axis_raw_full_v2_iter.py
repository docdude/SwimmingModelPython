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
    1: {"width": 3, "prominence": 1.5, "distance": 47},  # Freestyle
    2: {"width": 3, "prominence": 1.5, "distance": 48},  # Breaststroke
    3: {"width": 3, "prominence": 0.9, "distance": 45},  # Backstroke
    4: {"width": 3, "prominence": 1.3, "distance": 36},  # Butterfly
}

def plot_data_with_peaks(
    acc_data,
    gyro_data,
    acc_signals,
    gyro_signals,
    acc_peaks_indices,
    gyro_peaks_indices,
    row_indices,
    style,
    acc_prominence,
    acc_distance,
    gyro_prominence,
    gyro_distance,
    user_number=None,
    file_name=None,
    samples_per_plot=1000
):
    """
    Visualize accelerometer and gyroscope signals, ensuring equal samples are plotted for style-specific signals.

    Parameters:
    -----------
    acc_data : ndarray
        Complete accelerometer data (X, Y, Z axes).
    gyro_data : ndarray
        Complete gyroscope data (X, Y, Z axes).
    acc_signals : ndarray
        Style-specific accelerometer signal used for peak detection.
    gyro_signals : ndarray
        Style-specific gyroscope signal used for peak detection.
    acc_peaks_indices : list
        Indices of detected peaks in the accelerometer signal.
    gyro_peaks_indices : list
        Indices of detected peaks in the gyroscope signal.
    row_indices : ndarray
        Global row indices corresponding to each signal sample.
    style : int
        Swimming style identifier.
    user_number : int, optional
        User identifier for context.
    file_name : str, optional
        File name for context.
    samples_per_plot : int, optional
        Number of samples to visualize per plot.
    """
    label_name = label_names_abb[style]
    total_samples = len(acc_signals)
    num_plots = (total_samples + samples_per_plot - 1) // samples_per_plot  # Total plots needed

    for plot_idx in range(num_plots):
        # Determine the range of samples for this plot
        start_idx = plot_idx * samples_per_plot
        end_idx = min((plot_idx + 1) * samples_per_plot, total_samples)
        plot_range = slice(start_idx, end_idx)

        # Extract peaks within this range
        acc_peaks_in_chunk = [p for p in acc_peaks_indices if start_idx <= p < end_idx]
        gyro_peaks_in_chunk = [p for p in gyro_peaks_indices if start_idx <= p < end_idx]

        # Adjust peaks for the current plot range
        adjusted_acc_peaks = [p - start_idx for p in acc_peaks_in_chunk]
        adjusted_gyro_peaks = [p - start_idx for p in gyro_peaks_in_chunk]

        # Page 1: Raw accelerometer and gyroscope data (X, Y, Z axes)
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(
            f"User: {user_number}, File: {file_name}, Style: {label_name}, Plot: {plot_idx + 1} (Raw Axes)",
            fontsize=16,
        )

        for i, axis_label in enumerate(["X", "Y", "Z"]):
            axs[0, i].plot(acc_data[plot_range, i], label=f"Acc {axis_label}", color=["blue", "green", "purple"][i])
            axs[0, i].set_title(f"Accelerometer {axis_label}")
            axs[0, i].grid(True)
            axs[0, i].legend()

            axs[1, i].plot(gyro_data[plot_range, i], label=f"Gyro {axis_label}", color=["orange", "red", "brown"][i])
            axs[1, i].set_title(f"Gyroscope {axis_label}")
            axs[1, i].grid(True)
            axs[1, i].legend()

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

        # Page 2: Style-specific accelerometer and gyroscope signals
        fig, axs = plt.subplots(2, 1, figsize=(15, 10))
        fig.suptitle(
            f"User: {user_number}, File: {file_name}, Style: {label_name}, Plot: {plot_idx + 1} (Style-Specific Axes)",
            fontsize=16,
        )

        # Style-specific accelerometer plot
        axs[0].plot(acc_signals[plot_range], label="Style-Specific Accelerometer", color="darkblue")
        axs[0].plot(
            adjusted_acc_peaks,
            acc_signals[acc_peaks_in_chunk],
            "ro",
            label="Detected Peaks",
        )
        axs[0].set_title("Style-Specific Accelerometer Signal")
        axs[0].legend(
            loc='best',
            title=f"Prominence: {acc_prominence:.2f}\nDistance: {acc_distance:.2f}"
        )
        axs[0].grid(True)

        # Annotate accelerometer peaks and distances
        for i in range(len(adjusted_acc_peaks)):
            peak_start = adjusted_acc_peaks[i]
            axs[0].annotate(
                str(row_indices[acc_peaks_in_chunk[i]]),
                (peak_start, acc_signals[acc_peaks_in_chunk[i]]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=8,
                color="black",
            )
            # Add distance lines for all but the last peak
            if i < len(adjusted_acc_peaks) - 1:
                peak_end = adjusted_acc_peaks[i + 1]
                distance = peak_end - peak_start
                axs[0].annotate(
                    f"{distance} samples",
                    ((peak_start + peak_end) // 2,
                     (acc_signals[acc_peaks_in_chunk[i]] + acc_signals[acc_peaks_in_chunk[i + 1]]) / 2),
                    textcoords="offset points",
                    xytext=(0, -15),
                    ha="center",
                    fontsize=8,
                    color="blue",
                )
                axs[0].plot([peak_start, peak_end], [acc_signals[acc_peaks_in_chunk[i]], acc_signals[acc_peaks_in_chunk[i + 1]]], "g--")

        # Style-specific gyroscope plot
        axs[1].plot(gyro_signals[plot_range], label="Style-Specific Gyroscope", color="darkred")
        axs[1].plot(
            adjusted_gyro_peaks,
            gyro_signals[gyro_peaks_in_chunk],
            "ro",
            label="Detected Peaks",
        )
        axs[1].set_title("Style-Specific Gyroscope Signal")
        axs[1].legend(
            loc='best',
            title=f"Prominence: {gyro_prominence:.2f}\nDistance: {gyro_distance:.2f}"
        )
        axs[1].grid(True)

        # Annotate gyroscope peaks and distances
        for i in range(len(adjusted_gyro_peaks)):
            peak_start = adjusted_gyro_peaks[i]
            axs[1].annotate(
                str(row_indices[gyro_peaks_in_chunk[i]]),
                (peak_start, gyro_signals[gyro_peaks_in_chunk[i]]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=8,
                color="black",
            )
            # Add distance lines for all but the last peak
            if i < len(adjusted_gyro_peaks) - 1:
                peak_end = adjusted_gyro_peaks[i + 1]
                distance = peak_end - peak_start
                axs[1].annotate(
                    f"{distance} samples",
                    ((peak_start + peak_end) // 2,
                     (gyro_signals[gyro_peaks_in_chunk[i]] + gyro_signals[gyro_peaks_in_chunk[i + 1]]) / 2),
                    textcoords="offset points",
                    xytext=(0, -15),
                    ha="center",
                    fontsize=8,
                    color="red",
                )
                axs[1].plot([peak_start, peak_end], [gyro_signals[gyro_peaks_in_chunk[i]], gyro_signals[gyro_peaks_in_chunk[i + 1]]], "g--")

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
        gyro_y_neg_data = -gyro_data[1]  # Z axis
        sensor_data.update({
            "style_acc": acc_y_neg_data,
            "style_gyro": gyro_y_neg_data,
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
        acc_z_neg_data = -acc_data[2]  # Z axis
        gyro_y_neg_data = -gyro_data[1]  # Y axis
        sensor_data.update({
            "style_acc": acc_z_neg_data,
            "style_gyro": gyro_y_neg_data,
            "style_acc_peak_key": "acc_z_negative",
            "style_gyro_peak_key": "gyro_y_negative"
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


def calculate_dynamic_distance(signal, fs=30, style=None, base_distance=40, min_freq=0.3, max_freq=2.0):
    """
    Calculate dynamic distance for peak detection based on signal periodicity.

    Parameters:
    -----------
    signal : ndarray
        Input signal for periodicity analysis.
    fs : float
        Sampling frequency of the signal.
    style : int, optional
        Swim style identifier for style-specific adjustments.
    base_distance : float
        Default fallback distance if periodicity cannot be estimated.
    min_freq : float
        Minimum frequency for valid stroke detection (Hz).
    max_freq : float
        Maximum frequency for valid stroke detection (Hz).

    Returns:
    --------
    float
        Calculated dynamic distance.
    """
    periodicity = estimate_periodicity_fft(signal, fs, style, min_freq, max_freq)
    if periodicity is None:
        return base_distance

    # Apply style-specific scaling
    style_scaling = {
        1: 0.80,  # Freestyle
        2: 0.73,  # Breaststroke
        3: 0.75,  # Backstroke
        4: 0.70,  # Butterfly
    }

    style_min_distance = {
        1: 0, # Freestyle
        2: 0, # Breaststoke
        3: 0, # Backstroke
        4: 0, # Butterfly
    }

    scale_factor = style_scaling.get(style, 1.0)
    min_distance = style_min_distance.get(style, 0.0)

    dynamic_distance = periodicity * scale_factor + min_distance

    # Constrain the distance 
    distance = np.clip(dynamic_distance, base_distance, 100)
    # Debugging output for periodicity
    if DEBUG_DETECT_PEAKS:
        print(f"Distance: {distance} samples")
        print(f"Periodicity: {periodicity:.2f} samples")
    return distance


def estimate_periodicity_fft(signal, fs=30, style=None, min_freq=0.3, max_freq=2.0):
    """
    Estimate periodicity of a signal using FFT, filtered for stroke-relevant frequencies.

    Parameters:
    -----------
    signal : ndarray
        The input signal to estimate periodicity.
    fs : float
        Sampling frequency of the signal.
    style : int, optional
        Swim style identifier for style-specific tuning.
    min_freq : float
        Minimum frequency to consider as relevant for periodicity (in Hz).
    max_freq : float
        Maximum frequency to consider as relevant for periodicity (in Hz).

    Returns:
    --------
    float
        Estimated periodicity (peak-to-peak distance in samples).
    """
    # Normalize the signal
    signal = (signal - np.mean(signal)) / np.std(signal)

    # Compute FFT
    freqs = np.fft.rfftfreq(len(signal), d=1/fs)  # Frequency bins
    fft_magnitude = np.abs(np.fft.rfft(signal))   # Magnitude of FFT

    # Filter for relevant frequencies
    valid_freqs = (freqs >= min_freq) & (freqs <= max_freq)
    filtered_magnitude = fft_magnitude[valid_freqs]
    filtered_freqs = freqs[valid_freqs]

    # Identify dominant frequency
    if len(filtered_magnitude) == 0 or np.max(filtered_magnitude) == 0:
        return None  # No valid periodicity detected

    dominant_freq = filtered_freqs[np.argmax(filtered_magnitude)]

    # Convert frequency to periodicity
    periodicity = fs / dominant_freq

    if DEBUG_FFT_DISTANCE:
        plt.plot(filtered_freqs, filtered_magnitude)
        plt.title(f"FFT Spectrum ({label_names_abb[style]})")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.show()

    return periodicity


def count_strokes_and_detect_peaks(swimming_data, user_number=None, file_name=None, debug=False):
    """
    Count strokes and detect peaks using style-specific peak detection logic.

    Parameters:
    -----------
    swimming_data : LearningData object
        The loaded swimming data.
    user_number : int, optional
        User identifier for plotting.
    file_name : str, optional
        File name for plotting.
    debug : bool, optional
        Enable debug output.

    Returns:
    --------
    dict
        Stroke counts, stroke labels for each style.
    """
    stroke_counts = {label: 0 for label in label_names_abb}
    stroke_labels = np.zeros(len(swimming_data.data_dict['original'][user_number][file_name]), dtype=int)

    # Initialize lists to accumulate style-specific data
    processed_data = {style: [] for style in range(1, 5)}

    # Process only the specified file
    df = swimming_data.data_dict['original'][user_number][file_name]

    # Group data by style
    for index, row in df.iterrows():
        style = row['label']
        # Skip -1, 0, 5, 6 which are unknown, rest, turns or kicks
        if style in processed_data:
            sensor_data = preprocess_sensor_data_row(row.values, style)
            processed_data[style].append((sensor_data, index))  # Store sensor data with row index

    # Now process each style's accumulated data for peak detection and counting
    for style, data in processed_data.items():
        if not data:
            continue

        # Extract accelerometer and gyroscope signals (style-specific)
        style_acc_signals = np.array([d[0]['style_acc'] for d in data])
        style_gyro_signals = np.array([d[0]['style_gyro'] for d in data])

        # Extract complete accelerometer and gyroscope signals (all axes)
        acc_data = np.array([d[0]['acc_data'] for d in data])  # Shape: (N, 3) for X, Y, Z
        gyro_data = np.array([d[0]['gyro_data'] for d in data])  # Shape: (N, 3) for X, Y, Z

        row_indices = np.array([d[1] for d in data])  # Store row indices

                # Debug output for extracted data
        if DEBUG_DETECT_PEAKS:
            print(f"[DEBUG] Style: {label_names_abb[style]}")
            print(f"[DEBUG] Shape of acc_data: {acc_data.shape}")
            print(f"[DEBUG] Shape of gyro_data: {gyro_data.shape}")
            print(f"[DEBUG] Shape of style_acc_signals: {style_acc_signals.shape}")
            print(f"[DEBUG] Shape of style_gyro_signals: {style_gyro_signals.shape}")


        # Get style-specific peak detection parameters
        style_params = stroke_axis_params[style]
        base_prominence = style_params["prominence"]
        distance = style_params["distance"]

        # Calculate dynamic prominence for accelerometer and gyroscope signals
        if DEBUG_DETECT_PEAKS:  
            print("\nCalculating prominence and distance from processed accelerometer data:")
        acc_prominence = calculate_dynamic_prominence(style_acc_signals, base_prominence, style)
        acc_distance = calculate_dynamic_distance(style_acc_signals, style=style, base_distance=distance)
        if DEBUG_DETECT_PEAKS:  
            print("\nCalculating prominence and distance from processed gyroscope data:")
        gyro_prominence = calculate_dynamic_prominence(style_gyro_signals, base_prominence, style)
        gyro_distance = calculate_dynamic_distance(style_gyro_signals, style=style, base_distance=distance)



        # Detect peaks
        acc_peaks_indices = scipy.signal.find_peaks(
            style_acc_signals,
            prominence=acc_prominence,
            distance=acc_distance
        )[0]

        gyro_peaks_indices = scipy.signal.find_peaks(
            style_gyro_signals,
            prominence=gyro_prominence,
            distance=gyro_distance
        )[0]

        # Debugging output
        if DEBUG_COUNT_STROKES:
            print(f"[DEBUG] Style: {label_names_abb[style]}")
            sorted_list = sorted(acc_peaks_indices)
            max_width = len(str(max(sorted_list)))
            formatted_list = f"[{' '.join(f'{idx:>{max_width}}' for idx in sorted_list)} ]"
            # Print the formatted list
            print(f"[DEBUG] Detected Accelerometer Peaks: {formatted_list}")
            sorted_list = sorted(gyro_peaks_indices)
            max_width = len(str(max(sorted_list)))
            formatted_list = f"[{' '.join(f'{idx:>{max_width}}' for idx in sorted_list)} ]"
            # Print the formatted list
            print(f"[DEBUG] Detected Gyroscope Peaks    : {formatted_list}")

        # Track unique peaks to avoid double-counting
        unique_peaks = set()
        detected_acc_peaks_row_indices = set()
        detected_gyro_peaks_row_indices = set()

        if style in [2]:
            # Count strokes based on detected peaks
            for peak in acc_peaks_indices:
                row_idx = row_indices[peak]
                if peak not in unique_peaks:
                    stroke_labels[row_idx] = 1  # Mark the detected peak
                    stroke_counts[label_names_abb[style]] += 1
                    unique_peaks.add(peak)
                    detected_acc_peaks_row_indices.add(row_idx)
        elif style in [1, 3, 4]:
            for peak in gyro_peaks_indices:
                row_idx = row_indices[peak]
                if peak not in unique_peaks:
                    stroke_labels[row_idx] = 1  # Mark the detected peak
                    stroke_counts[label_names_abb[style]] += 1
                    unique_peaks.add(peak)
                    detected_gyro_peaks_row_indices.add(row_idx)

        # Debugging output
        if DEBUG_COUNT_STROKES:
            print(f"[DEBUG] Style by row index: {label_names_abb[style]}")

            if style in [2]:
                sorted_list = sorted(detected_acc_peaks_row_indices)
                max_width = len(str(max(sorted_list)))
                formatted_list = f"[{' '.join(f'{idx:>{max_width}}' for idx in sorted_list)} ]"

                # Print the formatted list
                print(f"[DEBUG] Detected Accelerometer Peaks: {formatted_list}")
            if style in [1,3,4]:
                sorted_list = sorted(detected_gyro_peaks_row_indices)
                max_width = len(str(max(sorted_list)))
                formatted_list = f"[{' '.join(f'{idx:>{max_width}}' for idx in sorted_list)} ]"
                # Print the formatted list
                print(f"[DEBUG] Detected Gyroscop Peaks     : {formatted_list}")

            print(f"[DEBUG] Stroke Counts: {stroke_counts[label_names_abb[style]]}")

        # Visualization of the detected peaks
        if DEBUG_PLOT_PEAKS:
            fig, axs = plt.subplots(3, 1, figsize=(14, 10))
            fig.suptitle(f"Style: {label_names_abb[style]} - Visualization", fontsize=16)

            # Accelerometer Plot
            axs[0].plot(style_acc_signals, label='Accelerometer Signal', color='blue')
            axs[0].plot(acc_peaks_indices, style_acc_signals[acc_peaks_indices], ".", label='Detected Peaks', color='red')

            # Annotate accelerometer peaks with indices
            for peak_idx in acc_peaks_indices:
                row_idx = row_indices[peak_idx]
                axs[0].annotate(
                    f"{row_idx}",
                    (peak_idx, style_acc_signals[peak_idx]),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    fontsize=8,
                    color="black",
                )
            axs[0].set_title("Accelerometer Signal")
            axs[0].legend(
                loc='best',
                title=f"Prominence: {acc_prominence:.2f}\nDistance: {acc_distance:.2f}"
            )

            # Gyroscope Plot
            axs[1].plot(style_gyro_signals, label='Gyroscope Signal', color='orange')
            axs[1].plot(gyro_peaks_indices, style_gyro_signals[gyro_peaks_indices], ".", label='Detected Peaks', color='red')

            # Annotate gyroscope peaks with indices
            for peak_idx in gyro_peaks_indices:
                row_idx = row_indices[peak_idx]
                axs[1].annotate(
                    f"{row_idx}",
                    (peak_idx, style_gyro_signals[peak_idx]),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    fontsize=8,
                    color="black",
                )
            axs[1].set_title("Gyroscope Signal")
            axs[1].legend(
                loc='best',
                title=f"Prominence: {gyro_prominence:.2f}\nDistance: {gyro_distance:.2f}"
            )

            # Heatmap Visualization
            axs[2].imshow(
                [style_acc_signals, style_gyro_signals],
                aspect='auto',
                cmap='viridis',
                interpolation='nearest'
            )
            axs[2].set_yticks([0, 1])
            axs[2].set_yticklabels(["Accelerometer", "Gyroscope"])
            axs[2].set_title("Heatmap of Signals")

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()


        # Optional visualization for stroke counting
        # Call plotting function only once for the style and file
        if DEBUG_COUNT_STROKES_PLOTTER:
            plot_data_with_peaks(
                acc_data=acc_data,
                gyro_data=gyro_data,
                acc_signals=style_acc_signals,
                gyro_signals=style_gyro_signals,
                acc_peaks_indices=acc_peaks_indices,
                gyro_peaks_indices=gyro_peaks_indices,
                row_indices=row_indices,
                style=style,
                acc_prominence=acc_prominence,
                acc_distance=acc_distance,
                gyro_prominence=gyro_prominence,
                gyro_distance=gyro_distance,
                user_number=user_number,
                file_name=file_name,
                samples_per_plot=1000
            )



    return stroke_counts, stroke_labels



# Update the main function to use the new row-by-row counting function
def main():
    # Global debug flags can be set here
    global DEBUG_DETECT_PEAKS, DEBUG_PLOT_PEAKS, DEBUG_FFT_DISTANCE, DEBUG_SYNCHRONIZATION, DEBUG_COUNT_STROKES, DEBUG_COUNT_STROKES_PLOTTER
    
    # Turn on/off specific debug outputs
    DEBUG_DETECT_PEAKS = False
    DEBUG_PLOT_PEAKS = False
    DEBUG_FFT_DISTANCE = False
    DEBUG_SYNCHRONIZATION = False
    DEBUG_COUNT_STROKES = False
    DEBUG_COUNT_STROKES_PLOTTER = False

    #data_path = '/Users/juanloya/Documents/SwimmingModelPython/swim_v2/SwimStyleData2_orig'

    data_path = '/Users/juanloya/Documents/SwimmingModelPython/swim_v2/data_copy/processed_30Hz_relabeled'
    results_path = '/Users/juanloya/Documents/SwimmingModelPython/swim_v2/tutorial_save_path_epoch60_mag/'
    output_dir = '/Users/juanloya/Documents/SwimmingModelPython/swim_v2/data_modified_users'


    with open(os.path.join(results_path, '1', 'data_parameters.pkl'), 'rb') as f:
        data_parameters = pickle.load(f)[0] 
    users = data_parameters['users']

    for user in users:
        experiment_save_path = os.path.join(results_path, user)

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
        

        print(f"Working on User: {user}.")   
            
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
                
            stroke_counts, stroke_labels = count_strokes_and_detect_peaks(
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