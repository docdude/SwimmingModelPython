import os
import numpy as np
import pandas as pd
import random

# Constants for simulation
SWIM_STYLES = {
    1: {"name": "freestyle", "spm_range": (50, 70)},
    2: {"name": "breaststroke", "spm_range": (30, 50)},
    3: {"name": "backstroke", "spm_range": (45, 65)},
    4: {"name": "butterfly", "spm_range": (40, 60)},
}
TURN_LABEL = 5
NULL_LABEL = 0
SAMPLING_RATE = 30  # Hz
NUM_SWIMMERS = 50
OUTPUT_DIR = "SyntheticSwimmingData"

real_stats = {
    'ACC': {'mean': [-10.045672, 0.662138, 1.378299],
            'std': [9.670708, 10.485108, 7.143383],
            'min': [-68.464865, -24.947702, -34.048293],
            'max': [16.843439, 78.301846, 21.382489]},
    'GYRO': {'mean': [-0.970774, 0.383655, -1.549766],
             'std': [4.771066, 3.476832, 3.167390],
             'min': [-18.216673, -14.488169, -12.574794],
             'max': [13.667321, 9.098352, 5.382897]},
    'MAG': {'mean': [13.076617, -6.246337, -13.982487],
            'std': [12.735637, 18.240433, 20.276421],
            'min': [-25.647332, -43.825191, -53.185191],
            'max': [53.539872, 44.682488, 55.341928]},
}

def align_statistics(data, real_stats, max_iterations=10, tolerance=0.01):
    """Align synthetic data with real data statistics."""
    data = np.array(data)
    real_mean = np.array(real_stats['mean'])
    real_std = np.array(real_stats['std'])
    real_min = np.array(real_stats['min'])
    real_max = np.array(real_stats['max'])

    for iteration in range(max_iterations):
        # Calculate current stats
        current_mean = np.mean(data, axis=0)
        current_std = np.std(data, axis=0)
        current_min = np.min(data, axis=0)
        current_max = np.max(data, axis=0)

        # Adjust mean and standard deviation
        data -= current_mean
        data /= current_std
        data *= real_std
        data += real_mean

        # Scale to match range
        current_min = np.min(data, axis=0)
        current_max = np.max(data, axis=0)
        range_scale = (real_max - real_min) / (current_max - current_min)
        range_shift = real_min - current_min * range_scale
        data = data * range_scale + range_shift

        # Clamp to real range
        data = np.clip(data, real_min, real_max)

        # Check for convergence
        mean_diff = np.linalg.norm(real_mean - np.mean(data, axis=0))
        std_diff = np.linalg.norm(real_std - np.std(data, axis=0))
        if mean_diff < tolerance and std_diff < tolerance:
            break

    return data

def adjust_synthetic_data(acc_data, gyro_data, mag_data, real_stats):
    """Adjust synthetic data distributions to match real statistics."""
    acc_data = align_statistics(acc_data, real_stats['ACC'])
    gyro_data = align_statistics(gyro_data, real_stats['GYRO'])
    mag_data = align_statistics(mag_data, real_stats['MAG'])
    return acc_data, gyro_data, mag_data

def generate_synthetic_data(swimmer_id, num_files=5):
    swimmer_dir = os.path.join(OUTPUT_DIR, f"{swimmer_id}")
    os.makedirs(swimmer_dir, exist_ok=True)

    for style_id, style_info in SWIM_STYLES.items():
        spm_range = style_info["spm_range"]

        for file_num in range(num_files):
            total_duration = random.randint(60, 180)  # 1–3 minutes
            num_samples = total_duration * SAMPLING_RATE

            # Generate random sensor data
            acc_data = np.random.normal(0, 1, (num_samples, 3))
            gyro_data = np.random.normal(0, 1, (num_samples, 3))
            mag_data = np.random.normal(0, 1, (num_samples, 3))

            # Adjust synthetic data
            acc_data, gyro_data, mag_data = adjust_synthetic_data(acc_data, gyro_data, mag_data, real_stats)

            # Generate labels and strokes
            labels = np.full(num_samples, style_id)
            spm = random.randint(*spm_range)  # Strokes per minute
            stroke_interval = int(SAMPLING_RATE * (60 / spm))
            stroke_indices = np.arange(0, num_samples, stroke_interval)
            stroke_counts = np.zeros(num_samples)
            stroke_counts[stroke_indices] = 1

            # Add null and turn segments
            null_start_duration = random.randint(2, 5) * SAMPLING_RATE
            labels[:null_start_duration] = NULL_LABEL
            acc_data[:null_start_duration] = np.random.normal(real_stats['ACC']['mean'], 
                                                              0.1 * np.array(real_stats['ACC']['std']), 
                                                              (null_start_duration, 3))
            gyro_data[:null_start_duration] = np.random.normal(real_stats['GYRO']['mean'], 
                                                               0.1 * np.array(real_stats['GYRO']['std']), 
                                                               (null_start_duration, 3))
            mag_data[:null_start_duration] = np.random.normal(real_stats['MAG']['mean'], 
                                                              0.1 * np.array(real_stats['MAG']['std']), 
                                                              (null_start_duration, 3))
            stroke_counts[:null_start_duration] = 0

            turn_interval = random.randint(25, 35) * SAMPLING_RATE
            for i in range(turn_interval, num_samples, turn_interval):
                turn_duration = random.randint(2, 5) * SAMPLING_RATE
                turn_end = min(i + turn_duration, num_samples)
                labels[i:turn_end] = TURN_LABEL
                acc_data[i:turn_end] = np.random.normal(real_stats['ACC']['mean'], 
                                                        0.2 * np.array(real_stats['ACC']['std']), 
                                                        (turn_end - i, 3))
                gyro_data[i:turn_end] = np.random.normal(real_stats['GYRO']['mean'], 
                                                         0.2 * np.array(real_stats['GYRO']['std']), 
                                                         (turn_end - i, 3))
                mag_data[i:turn_end] = np.random.normal(real_stats['MAG']['mean'], 
                                                        0.2 * np.array(real_stats['MAG']['std']), 
                                                        (turn_end - i, 3))
                stroke_counts[i:turn_end] = 0

            # Compile data
            timestamps = np.arange(0, num_samples) * (1e9 / SAMPLING_RATE)
            df = pd.DataFrame({
                "timestamp": timestamps.astype(np.int64),
                "ACC_0": acc_data[:, 0],
                "ACC_1": acc_data[:, 1],
                "ACC_2": acc_data[:, 2],
                "GYRO_0": gyro_data[:, 0],
                "GYRO_1": gyro_data[:, 1],
                "GYRO_2": gyro_data[:, 2],
                "MAG_0": mag_data[:, 0],
                "MAG_1": mag_data[:, 1],
                "MAG_2": mag_data[:, 2],
                "label": labels,
                "stroke_count": np.cumsum(stroke_counts * (labels != NULL_LABEL) * (labels != TURN_LABEL)),
            })

        style_name = style_info["name"]
        file_path = os.path.join(swimmer_dir, f"{style_name}.csv")
        df.to_csv(file_path, index=False)

# Generate data for all swimmers
for swimmer_id in range(NUM_SWIMMERS):
    generate_synthetic_data(swimmer_id)

print(f"Data generated in {OUTPUT_DIR}")

