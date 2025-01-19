import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

def visualize_refined_peak_detection(signal, timestamps, prominence, distance, style_name="Style"):
    """
    Visualize signal with scipy.find_peaks prominence and distance parameters.

    Parameters:
    -----------
    signal : ndarray
        The acceleration or sensor data signal.
    timestamps : ndarray
        Timestamps corresponding to the signal.
    prominence : float
        Prominence parameter for peak detection.
    distance : int
        Minimum distance parameter for peak detection (in samples).
    style_name : str
        Name of the swimming style for title labeling.
    """
    # Detect peaks using find_peaks
    peaks, properties = find_peaks(signal, prominence=prominence, distance=distance)

    # Calculate distances in samples
    sample_distances = np.diff(peaks)

    # Create the plot
    plt.figure(figsize=(15, 8))
    plt.plot(timestamps, signal, label=f"{style_name} Signal", alpha=0.7)

    # Highlight detected peaks
    plt.scatter(timestamps[peaks], signal[peaks], color='red', label="Detected Peaks", zorder=5)

    # Plot prominence range for each peak
    for i, peak in enumerate(peaks):
        baseline = signal[peak] - properties['prominences'][i]
        plt.vlines(x=timestamps[peak], ymin=baseline, ymax=signal[peak],
                   color='green', linestyle='dotted', alpha=0.6, label="Prominence Range" if i == 0 else "")

    # Plot sample-based distances between peaks
    for i in range(len(sample_distances)):
        if sample_distances[i] < distance:
            color = 'red'  # Invalid distance
        else:
            color = 'blue'  # Valid distance
        plt.hlines(y=(signal[peaks[i]] + signal[peaks[i + 1]]) / 2,
                   xmin=timestamps[peaks[i]], xmax=timestamps[peaks[i + 1]],
                   color=color, linestyle='dashed', alpha=0.7,
                   label="Distance (Samples)" if i == 0 else "")
        plt.text((timestamps[peaks[i]] + timestamps[peaks[i + 1]]) / 2,
                 (signal[peaks[i]] + signal[peaks[i + 1]]) / 2 + 0.5,
                 f"{sample_distances[i]} samples", color=color, fontsize=8, ha='center')

    # Add dynamic prominence reference
    dynamic_prominence = np.median(properties['prominences'])
    plt.axhline(y=dynamic_prominence, color='orange', linestyle='--', label="Dynamic Prominence Reference", alpha=0.7)

    # Add labels and legend
    plt.title(f"{style_name} Signal with Peak Detection (Prominence={prominence}, Distance={distance})")
    plt.xlabel("Timestamps")
    plt.ylabel("Signal Value")
    plt.legend()
    plt.grid()
    plt.show()

def calculate_dynamic_prominence(signal, base_prominence=0.5, style=1):
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
        range_scale = 0.015
        std_scale = 0.1
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

    return dynamic_prominence

def visualize_stroke_labels(csv_file, swim_style, user_dir, file_name):
    """
    Visualize sensor data with stroke labels, using style-specific acceleration peaks.

    Parameters:
    -----------
    csv_file : str
        Path to the CSV file.
    swim_style : str
        Swim style (e.g., 'freestyle', 'breaststroke', 'backstroke', 'butterfly').
    """
    # Load the data
    data = pd.read_csv(csv_file)

    # Extract relevant columns
    timestamp = data["timestamp"]
    stroke_count = data["stroke_labels"]

    # Check for unexpected stroke_count values
    unexpected_values = data[~data["stroke_labels"].isin([0, 1])]
    if not unexpected_values.empty:
        print(f"Unexpected stroke_labels values found in {csv_file}:")
        print(unexpected_values)

    # Set style-specific parameters
    if swim_style == "freestyle":
        data["Swim_Acc"] = data["ACC_1"]  # Freestyle: ACC Y-axis (negative peaks)
        style_id = 1
        base_prominence = 1.5
        min_distance = 35  # Freestyle minimum distance (timestamp units)
    elif swim_style == "breaststroke":
        data["Swim_Acc"] = np.sqrt(data["ACC_0"]**2 + data["ACC_1"]**2 + data["ACC_2"]**2)  # Breaststroke: Acceleration magnitude
        style_id = 2
        base_prominence = 1.6
        min_distance = 50
    elif swim_style == "backstroke":
        data["Swim_Acc"] = -data["ACC_2"]  # Backstroke: ACC Z-axis (negative peaks)
        style_id = 3
        base_prominence = 1.2
        min_distance = 50
    elif swim_style == "butterfly":
        data["Swim_Acc"] = np.sqrt(data["ACC_0"]**2 + data["ACC_1"]**2 + data["ACC_2"]**2)  # Butterfly: Acceleration magnitude
        style_id = 4
        base_prominence = 1.6
        min_distance = 45
    else:
        print(f"Unknown swim style: {swim_style}")
        return

    # Calculate dynamic prominence for the swim style
    dynamic_prominence = calculate_dynamic_prominence(data["Swim_Acc"], base_prominence=base_prominence, style=style_id)
    prominence_line = [dynamic_prominence] * len(data)

    # Count total strokes detected
    total_strokes = int(stroke_count.sum())

    # Plot the acceleration signal
    plt.figure(figsize=(15, 8))
    plt.plot(timestamp, data["Swim_Acc"], label=f"{swim_style.capitalize()} Acc Signal", alpha=0.7)

    # Overlay stroke counts with small dots
    stroke_indices = stroke_count[stroke_count == 1].index
    plt.scatter(
        timestamp[stroke_count == 1], 
        data["Swim_Acc"][stroke_count == 1], 
        color='red', 
        label="Detected Strokes (1)", 
        marker=".", 
        s=40
    )

    # Add dynamic prominence line for the current style
    plt.plot(timestamp, prominence_line, '--', label="Dynamic Prominence Line", alpha=0.6)

    # Add reference line for the minimum distance
    plt.axhline(y=np.mean(data["Swim_Acc"]), color='gray', linestyle='--', alpha=0.5, label=f"Min Distance Threshold = {min_distance}")

    # Distance calculation and plotting
    if len(stroke_indices) > 1:
        distances = np.diff(timestamp[stroke_indices].values)
        for i in range(len(distances)):
            x1, x2 = timestamp[stroke_indices[i]], timestamp[stroke_indices[i + 1]]
            y = (data["Swim_Acc"][stroke_indices[i]] + data["Swim_Acc"][stroke_indices[i + 1]]) / 2

            # Color-code based on whether the distance meets the threshold
            if distances[i] < min_distance:
                color = 'red'  # Invalid distance
            else:
                color = 'blue'  # Valid distance

            # Draw a horizontal bar between consecutive strokes
            plt.plot([x1, x2], [y, y], color=color, linestyle='--', alpha=0.6)

            # Annotate the distance value
            plt.text((x1 + x2) / 2, y + 0.05, f"{distances[i]:.1f}", color=color, fontsize=8, ha='center')

    # Overlay unexpected stroke counts if present
    if not unexpected_values.empty:
        plt.scatter(
            unexpected_values["timestamp"], 
            unexpected_values["Swim_Acc"], 
            color='orange', 
            label="Unexpected Stroke Count", 
            marker="x", 
            s=200
        )

    # Add plot title with total strokes
    plt.title(f"User: {user_dir} {file_name} Sensor Data with Stroke Counts (Total Strokes: {total_strokes})")
    plt.xlabel("Timestamp")
    plt.ylabel("Swim Acceleration Signal")
    plt.legend()
    plt.grid()

    output_file = f"{user_dir}_{file_name}_plot.png"  # Customize the file name
    plt.savefig(output_file, format='png', dpi=300)
    plt.show()

def iterate_and_visualize(root_directory):
    """
    Iterate through each user subdirectory in the root directory and visualize updated CSV files.

    Parameters:
    -----------
    root_directory : str
        Path to the root directory containing user subdirectories.
    """
    # Iterate through each user subdirectory
    for user_dir in os.listdir(root_directory):
        user_dir = '20'
        user_path = os.path.join(root_directory, user_dir)
        if not os.path.isdir(user_path):
            continue

        print(f"Processing user directory: {user_dir}")
        
        # Iterate through CSV files in the user directory
        for file_name in os.listdir(user_path):
            if file_name.endswith("_updated.csv"):  # Process only updated CSV files
                csv_file_path = os.path.join(user_path, file_name)
                
                # Determine swim style from the file name (e.g., "Breaststroke", "Freestyle")
                if "Breaststroke" in file_name:
                    swim_style = "breaststroke"
                elif "Freestyle" in file_name:
                    swim_style = "freestyle"
                elif "Backstroke" in file_name:
                    swim_style = "backstroke"
                elif "Butterfly" in file_name:
                    swim_style = "butterfly"
                else:
                    swim_style = "unknown"
                
                print(f"Visualizing file: {file_name}, Style: {swim_style}")
                
                # Visualize the CSV file
                visualize_stroke_labels(csv_file_path, swim_style, user_dir, file_name)


# Run the script
root_directory = "data_modified"  # Replace with the actual path to your root directory
iterate_and_visualize(root_directory)
