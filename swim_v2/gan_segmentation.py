import os
import pandas as pd
import numpy as np

# Parameters
WIN_LEN = 180  # Window length
SLIDE_LEN = 30  # Overlap length
LABEL_COLUMN = "label"  # Column indicating swim style or transition
TIMESTAMP_COLUMN = "timestamp"  # Timestamp column
SENSOR_COLUMNS = ["ACC_0", "ACC_1", "ACC_2", "GYRO_0", "GYRO_1", "GYRO_2"]  # Sensor data columns
OUTPUT_DIR = "gan_dataset_with_timestamps"  # Directory to save segmented windows
MAX_FILES = 100  # Maximum number of files per swim style

# Label-to-name mapping
LABEL_NAMES = {
    1: "freestyle",
    2: "butterfly",
    3: "backstroke",
    4: "breaststroke",
}

# Create the output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_and_save_swim_style_windows(data, win_len, slide_len, sensor_columns, label_column, timestamp_column, output_dir, max_files):
    """
    Extract and save windows grouped by swim style, embedding `null` and `turn` within swim style sequences.
    Args:
    - data (pd.DataFrame): Input data.
    - win_len (int): Window length.
    - slide_len (int): Slide step size.
    - sensor_columns (list): Sensor data columns.
    - label_column (str): Column for swim style or transition labels.
    - timestamp_column (str): Column for timestamps.
    - output_dir (str): Output directory for segmented files.
    - max_files (int): Maximum number of files per swim style.
    """
    for label, style_name in LABEL_NAMES.items():
        style_data = data[data[label_column] == label]
        style_windows = [
            style_data.iloc[i : i + win_len] for i in range(0, len(style_data) - win_len + 1, slide_len)
        ]

        style_dir = os.path.join(output_dir, style_name)
        os.makedirs(style_dir, exist_ok=True)

        for i, window in enumerate(style_windows[:max_files]):
            file_path = os.path.join(style_dir, f"{style_name}_{i}.csv")
            # Save both timestamps, sensor values, and ground truth labels
            window[[timestamp_column] + sensor_columns + [label_column]].to_csv(file_path, index=False)

def process_swimmer_data(swimmer_dir, output_dir, win_len, slide_len, sensor_columns, label_column, timestamp_column, max_files):
    """
    Process all CSV files for a swimmer directory, extracting and saving swim style windows.
    """
    swimmer_data = []
    for file in os.listdir(swimmer_dir):
        file_path = os.path.join(swimmer_dir, file)
        if file.endswith(".csv"):
            swimmer_data.append(pd.read_csv(file_path))
    if not swimmer_data:
        return

    swimmer_data = pd.concat(swimmer_data, ignore_index=True)
    extract_and_save_swim_style_windows(swimmer_data, win_len, slide_len, sensor_columns, label_column, timestamp_column, output_dir, max_files)

# Main process
if __name__ == "__main__":
    dataset_root = "data/processed_30hz_relabeled"
    for swimmer_dir in os.listdir(dataset_root):
        swimmer_path = os.path.join(dataset_root, swimmer_dir)
        if os.path.isdir(swimmer_path):
            print(f"Processing swimmer directory: {swimmer_dir}")
            process_swimmer_data(swimmer_path, OUTPUT_DIR, WIN_LEN, SLIDE_LEN, SENSOR_COLUMNS, LABEL_COLUMN, TIMESTAMP_COLUMN, MAX_FILES)
    print(f"GAN dataset with timestamps saved to {OUTPUT_DIR}")
