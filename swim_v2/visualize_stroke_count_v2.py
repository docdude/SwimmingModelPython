import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks


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

def visualize_stroke_labels_with_annotations(csv_file, swim_style, user_dir, file_name):
    """
    Visualize sensor data with stroke labels overlayed, normalized stroke-specific values,
    and additional annotations for stroke indices.

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
    stroke_labels = data["stroke_labels"]
    # Check for unexpected stroke_count values
    unexpected_values = data[~data["stroke_labels"].isin([0, 1])]
    if not unexpected_values.empty:
        print(f"Unexpected stroke_labels values found in {csv_file}:")
        print(unexpected_values)

    # Set style-specific parameters
    if swim_style == "freestyle":
        data["Swim_Acc"] = -data["ACC_1"]  # Freestyle: ACC Y-axis (negative peaks)
        style_id = 1
        base_prominence = 1.5
        min_distance = 35  # Freestyle minimum distance (timestamp units)
    elif swim_style == "breaststroke":
        data["Swim_Acc"] = np.sqrt(data["ACC_0"]**2 + data["ACC_1"]**2 + data["ACC_2"]**2)  # Breaststroke: Acceleration magnitude
        style_id = 2
        base_prominence = 1.6
        min_distance = 50
    elif swim_style == "backstroke":
        data["Swim_Acc"] = data["ACC_2"]  # Backstroke: ACC Z-axis (positive peaks)
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
    total_strokes = int(stroke_labels.sum())
    # Normalize the signal
    normalized_signal = (data["Swim_Acc"] - np.mean(data["Swim_Acc"])) / np.std(data["Swim_Acc"])

    # Get indices of stroke labels
    stroke_indices = np.where(stroke_labels == 1)[0]

    # Calculate distances between stroke labels in samples
    stroke_distances = np.diff(stroke_indices)

    # Plot the normalized signal
    plt.figure(figsize=(15, 8))
    plt.plot(normalized_signal, label=f"Normalized {swim_style.capitalize()} Signal", alpha=0.7)
    
    # Overlay stroke labels with dots and annotate indices
    for idx in stroke_indices:
        plt.scatter(idx, normalized_signal[idx], color='purple', label="Stroke Label (1)", zorder=5)
        plt.text(
            idx, normalized_signal[idx], f"{idx}",
            color='black', fontsize=8, ha='left', va='bottom'
        )

    # Add prominence line
    plt.axhline(y=dynamic_prominence, color='green', linestyle='--', label=f"Prominence = {dynamic_prominence}")
    # Add dynamic prominence line for the current style
    #plt.plot(timestamp, prominence_line, '--', label="Dynamic Prominence Line", alpha=0.6)

    # Annotate distances between stroke labels
    for i, (start, end) in enumerate(zip(stroke_indices[:-1], stroke_indices[1:])):
        mid_point = (start + end) // 2
        plt.text(
            mid_point, normalized_signal[mid_point],
            f"{stroke_distances[i]}",
            color='blue', fontsize=8, ha='center', zorder=10
        )
        # Add horizontal line connecting the strokes
        plt.hlines(
            y=-normalized_signal[mid_point],
            xmin=start,
            xmax=end,
            color='blue',
            linestyle='dotted',
            linewidth=1,
            zorder=2
        )
    # Labels and legend
    plt.title(f"User: {user_dir} {file_name} Sensor Data with Stroke Counts (Total Strokes: {total_strokes})")
    plt.xlabel("Samples")
    plt.ylabel("Normalized Signal Value")
   # plt.legend()
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
                #visualize_stroke_labels(csv_file_path, swim_style, user_dir, file_name)
                visualize_stroke_labels_with_annotations(csv_file_path, swim_style, user_dir, file_name)



# Run the script
root_directory = "data_modified"  # Replace with the actual path to your root directory
iterate_and_visualize(root_directory)
