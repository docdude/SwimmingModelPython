import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import sys

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

    print(f"Signal analysis:")
    print(f"  Range: {signal_range:.4f}")
    print(f"  Standard Deviation: {local_std:.4f}")
    print(f"  Interquartile Range: {iqr:.4f}")
    print(f"  Log(Range + 1): {np.log1p(signal_range):.4f}")
    print(f"  Sqrt(STD): {np.sqrt(local_std):.4f}")
    print(f"  Sqrt(IQR): {np.sqrt(iqr):.4f}")
    print(f"Base prominence: {base_prominence:.4f}")            
    print(f"Calculated prominence: {dynamic_prominence:.4f}")

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
        data["Swim_Axis"] = -data["GYRO_1"]  # Freestyle: Gyro Y-Axis (Negative Peaks)
        style_id = 1
        base_prominence = 2.0
        min_distance = 35  # Freestyle minimum distance (timestamp units)
    elif swim_style == "breaststroke":
        data["Swim_Axis"] = np.sqrt(data["ACC_0"]**2 + data["ACC_1"]**2 + data["ACC_2"]**2)  # Breaststroke: Acceleration magnitude
        style_id = 2
        base_prominence = 1.6
        min_distance = 50
    elif swim_style == "backstroke":
        data["Swim_Axis"] = -data["GYRO_1"]  # Backstroke: Gyro Y-Axis (Negative peaks)
        style_id = 3
        base_prominence = 1.2
        min_distance = 50
    elif swim_style == "butterfly":
        data["Swim_Axis"] = -data["GYRO_1"]  # Butterfly: Gyro Y-Axis (Negative peaks)
        style_id = 4
        base_prominence = 1.6
        min_distance = 45
    else:
        print(f"Unknown swim style: {swim_style}")
        return

    # Calculate dynamic prominence for the swim style
    dynamic_prominence = calculate_dynamic_prominence(data["Swim_Axis"], base_prominence=base_prominence, style=style_id)
    prominence_line = [dynamic_prominence] * len(data)

    # Count total strokes detected
    total_strokes = int(stroke_labels.sum())
    # Normalize the signal
    normalized_signal = (data["Swim_Axis"] - np.mean(data["Swim_Axis"])) / np.std(data["Swim_Axis"])

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



def iterate_and_visualize(root_directory, user_dir=None):

    """
    Iterate through each user subdirectory in the root directory and visualize updated CSV files.

    Parameters:
    -----------
    root_directory : str
        Path to the root directory containing user subdirectories.
    """
    # If user_dir is specified, narrow the iteration
    user_dirs = [str(user_dir)] if user_dir is not None else os.listdir(root_directory)

    for user_dir in user_dirs:
        user_path = os.path.join(root_directory, user_dir)
        if not os.path.isdir(user_path):
            print(f"Skipping: {user_dir} (not a directory)")
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
def main():
            # # Access the command-line arguments
    if len(sys.argv) > 1:
        user_dir = sys.argv[1]
        print(f"Processing user_dir: {user_dir} ")
    else:
        print("Missing user!")
        return

    root_directory = "/Users/juanloya/Documents/SwimmingModelPython/swim_v2/data_modified_users"  # Replace with the actual path to your root directory
    iterate_and_visualize(root_directory, user_dir=user_dir)

if __name__ == '__main__':
    main()