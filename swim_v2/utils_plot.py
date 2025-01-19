import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def visualize_batch_and_mask(batch_data, batch_outputs, batch_sample_weights):
    """
    # Visualize stroke labels and mask for a specific window
    """
    example_index = 0  # Choose a batch sample index
    time_axis = np.arange(batch_data.shape[1])  # Length of the time axis

    plt.figure(figsize=(12, 6))

    # Plot sensor data (e.g., the first feature)
    plt.plot(time_axis, batch_data[example_index, :, 0], label="Sensor Feature 0")

    # Plot stroke labels
    plt.plot(
        time_axis,
        batch_outputs['stroke_label_output'][example_index, :, 0],
        label="Stroke Label Output",
        linestyle="--",
        linewidth=2,
    )

    # Plot stroke mask
    plt.plot(
        time_axis,
        batch_sample_weights['stroke_label_output'][example_index, :, 0],
        label="Stroke Mask",
        linestyle=":",
        linewidth=2,
    )

    plt.legend()
    plt.title(f"Debugging Sample {example_index}")
    plt.xlabel("Time Step")
    plt.ylabel("Values")
    plt.grid()
    plt.show()

def visualize_batch(batch_data, batch_outputs, batch_sample_weights):
    """
    # Visualize a sample batch
    """
    plt.figure(figsize=(10, 5))
    plt.plot(batch_data[0, :, 0], label='Feature 0')
    plt.plot(batch_outputs['stroke_label_output'][0, :, 0], label='Stroke Labels')
    plt.legend()
    plt.title('Sample Window from Generator')
    plt.show()

    # Visualize the stroke mask
    plt.figure(figsize=(10, 5))
    plt.plot(batch_sample_weights['stroke_label_output'][0, :, 0], label='Stroke Mask')
    plt.legend()
    plt.title('Stroke Mask for Sample Window')
    plt.show()


def plot_style_specific_axes(swimming_data):
    groups = swimming_data.data_dict.keys()  # ['original', 'time_scaled_0.9', 'time_scaled_1.1']
    users = swimming_data.users  # List of users

    # Map styles to specific axes or computations
    style_axes = {
        1: {"label": "Freestyle", "axis": "GYRO_2", "transform": lambda x: x, "negative": True},  # Gyro Z, negative
        2: {"label": "Breaststroke", "axis": "ACC_MAG", "transform": lambda x: np.sqrt(np.sum(x**2, axis=1))},  # Accel Magnitude
        3: {"label": "Backstroke", "axis": "GYRO_1", "transform": lambda x: x, "negative": True},  # Gyro Y, negative
        4: {"label": "Butterfly", "axis": "GYRO_1", "transform": lambda x: x, "negative": True},  # Gyro Y, negative
    }

    for user in users:
        for rec in swimming_data.data_dict['original'][user].keys():
            plt.figure(figsize=(18, 8))
            plt.title(f"User: {user}, Recording: {rec}")
            plt.xlabel("Index")
            plt.ylabel("Sensor Data Values")

            # Determine the labels in the recording
            labels = swimming_data.data_dict['original'][user][rec]['label'].values
            unique_labels = np.unique(labels)

            # Skip mixed recordings for simplicity
            if len(np.intersect1d(unique_labels, [1, 2, 3, 4])) > 1:
                print(f"Skipping mixed recording for User {user}, Recording {rec}")
                continue

            # Plot for each group
            for group in groups:#['original','time_scaled_1.1']:

                sensor_data = swimming_data.data_dict[group][user][rec][swimming_data.data_columns].values
                stroke_data = swimming_data.data_dict[group][user][rec][swimming_data.stroke_labels].values

                for style, info in style_axes.items():
                    if style in unique_labels:
                        if info["axis"] == "ACC_MAG":
                            # Compute magnitude for accelerometer
                            accel_data = sensor_data[:, :3]  # Assuming ACC_0, ACC_1, ACC_2
                            style_data = info["transform"](accel_data)
                        else:
                            # Extract the specific axis, accounting for "negative" flag
                            axis_idx = swimming_data.data_columns.index(info["axis"])
                            style_data = info["transform"](sensor_data[:, axis_idx])
                            if info.get("negative", False):  # Flip the values if the "negative" key is True
                                style_data = -style_data

                        # Plot the relevant data
                        plt.plot(style_data, label=f"{group} - {info['label']}")

                        # Overlay stroke labels with scatter plot and annotate indices
                        stroke_indices = np.where(stroke_data == 1)[0]
                        plt.scatter(stroke_indices, style_data[stroke_indices], color='purple', label="Stroke Label (1)", zorder=5)
                        for idx in stroke_indices:
                            plt.text(
                                idx, style_data[idx], f"{idx}",
                                color='black', fontsize=8, ha='center', va='bottom'
                            )

            plt.legend(loc="upper right")
            plt.grid()
            plt.show()


def plot_label_distribution(train_probs, val_probs):
    labels = sorted(set(train_probs.keys()).union(set(val_probs.keys())))
    train_values = [train_probs.get(label, 0) for label in labels]
    val_values = [val_probs.get(label, 0) for label in labels]

    # Plot
    plt.figure(figsize=(10, 6))
    bar_width = 0.35
    x = range(len(labels))

    plt.bar(x, train_values, bar_width, label="Training Set", alpha=0.7)
    plt.bar([p + bar_width for p in x], val_values, bar_width, label="Validation Set", alpha=0.7)

    plt.xlabel("Stroke Labels")
    plt.ylabel("Proportion")
    plt.title("Stroke Label Distribution in Training and Validation Sets")
    plt.xticks([p + bar_width / 2 for p in x], labels)
    plt.legend()
    plt.tight_layout()
    plt.show()

def analyze_feature_distributions(label_user_dict, swimming_data, feature_indices, data_type="training", exclude_label=None):
    """
    Analyze positive (stroke_label=1) and negative (stroke_label=0) class distributions across selected features.

    Args:
        label_user_dict (dict): A dictionary where keys are swim styles and values are lists of user data.
        swimming_data (object): The swimming_data instance containing the get_windows_dict method.
        feature_indices (list): Indices of features to analyze.
        data_type (str): "training" or "validation" to indicate the data source.
        exclude_label (int or None): Swim style label to exclude, if any.

    Returns:
        None: Plots distributions of selected features for stroke_label = 0 and 1.
    """
    if exclude_label is not None:
        label_user_dict = {label: users for label, users in label_user_dict.items() if label != exclude_label}

    try:
        x_val, _, _, y_stroke_val = swimming_data.get_windows_dict(label_user_dict)
    except Exception as e:
        print(f"Error extracting data for {data_type} set: {e}")
        return

    # Determine positive and negative windows based on the stroke threshold
    y_stroke_window = y_stroke_val

    # Reduce x_val for each window to summary stats (e.g., mean or max along axis 1)
    x_val_window = np.mean(x_val, axis=1)  # Shape (num_windows, num_features)

    # Separate positive and negative stroke windows
    pos_indices = np.where(y_stroke_window == 1)[0]
    neg_indices = np.where(y_stroke_window == 0)[0]

    # Validate the separation
    print(f"Number of positive windows: {len(pos_indices)}")
    print(f"Number of negative windows: {len(neg_indices)}")

    # Plot feature distributions
    for feature_index in feature_indices:
        plt.figure(figsize=(10, 6))
        plt.hist(
            x_val_window[neg_indices, feature_index],
            bins=50, alpha=0.5, label="Stroke Label 0 (Negative)", density=True
        )
        plt.hist(
            x_val_window[pos_indices, feature_index],
            bins=50, alpha=0.5, label="Stroke Label 1 (Positive)", density=True
        )
        plt.title(f"Feature {feature_index} Distribution by Stroke Label")
        plt.xlabel(f"Feature {feature_index}")
        plt.ylabel("Density")
        plt.legend()
        plt.show()


def analyze_feature_distribution(label_user_dict, swimming_data, feature_indices, data_type="training", exclude_label=None):
    """
    Analyze positive (stroke_label=1) and negative (stroke_label=0) class distributions across selected features.

    Args:
        label_user_dict (dict): A dictionary where keys are swim styles and values are lists of user data.
        swimming_data (object): The swimming_data instance containing the get_windows_dict method.
        feature_indices (list): Indices of features to analyze.
        data_type (str): "training" or "validation" to indicate the data source.
        exclude_label (int or None): Swim style label to exclude, if any.

    Returns:
        None: Plots distributions of selected features for stroke_label = 0 and 1.
    """
    if exclude_label is not None:
        label_user_dict = {label: users for label, users in label_user_dict.items() if label != exclude_label}

    try:
        x_val, _,  _, y_stroke_val = swimming_data.get_windows_dict(label_user_dict)
    except Exception as e:
        print(f"Error extracting data for {data_type} set: {e}")
        return

    # Determine positive and negative windows based on the stroke threshold
    y_stroke_window = y_stroke_val#.flatten()

    # Reduce x_val for each window to summary stats (e.g., mean or max along axis 1)
    #x_val_window = np.mean(x_val, axis=1)  # Shape (num_windows, num_features)
    x_val_window = x_val.reshape(x_val.shape[0], -1)  # Shape becomes (x, 180 * 6)

    # Separate positive and negative stroke windows
    pos_indices = np.where(y_stroke_window == 1)[0]
    neg_indices = np.where(y_stroke_window == 0)[0]

    # Validate the separation
    print(f"Number of positive windows: {len(pos_indices)}")
    print(f"Number of negative windows: {len(neg_indices)}")

    # Loop through pairs of selected features for joint distribution analysis
    for i, feature_index_x in enumerate(feature_indices):
        for feature_index_y in feature_indices[i + 1:]:
            # Extract positive and negative stroke windows
            pos_data = x_val_window[pos_indices][:, [feature_index_x, feature_index_y]]
            neg_data = x_val_window[neg_indices][:, [feature_index_x, feature_index_y]]

            # Create DataFrames for plotting
            pos_df = pd.DataFrame(pos_data, columns=[f"Feature {feature_index_x}", f"Feature {feature_index_y}"])
            neg_df = pd.DataFrame(neg_data, columns=[f"Feature {feature_index_x}", f"Feature {feature_index_y}"])

            # Create the first jointplot for positive data
            #plt.figure(figsize=(16, 8))
            x=f"Feature {feature_index_x}"
            y=f"Feature {feature_index_y}"
           # plt.subplot(1, 2, 1)
            sns.jointplot(
                x=pos_df[x], 
                y=pos_df[y],
                kind="hex"

            )
            plt.suptitle(f"Positive Distribution: Feature {feature_index_x} vs Feature {feature_index_y}", y=1.02)

            # Create the second jointplot for negative data
            sns.jointplot(
                x=neg_df[x],
                y=neg_df[y],
                kind="hex"

            )
            _ = plt.suptitle(f"Negative Distribution: Feature {feature_index_x} vs Feature {feature_index_y}", y=1.02)

            plt.show()


def plot_batch_and_mask(x_batch, y_stroke, mask, sample_idx=0, sensors=['ACC_X', 'ACC_Y', 'ACC_Z', 'GYRO_X', 'GYRO_Y', 'GYRO_Z']):
    """
    Visualize sensor data, stroke labels, and mask for a single sample in the batch
    
    Args:
        x_batch: shape (batch, timesteps, features, 1)
        y_stroke: shape (batch, timesteps, 1)
        mask: shape (batch, timesteps, 1)
        sample_idx: which sample in the batch to plot
        sensors: list of sensor names for legend
    """
    # Get the data for the specified sample
    sample_data = x_batch[sample_idx, :, :, 0]  # (timesteps, features)
    sample_stroke = y_stroke[sample_idx, :, 0]   # (timesteps,)
    sample_mask = mask[sample_idx, :, 0]         # (timesteps,)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[3, 1])
    
    # Plot sensor data
    timesteps = range(len(sample_data))
    for i in range(sample_data.shape[1]):
        ax1.plot(timesteps, sample_data[:, i], label=sensors[i])
    
    # Add stroke labels as vertical lines
    stroke_locations = np.where(sample_stroke == 1)[0]
    for loc in stroke_locations:
        ax1.axvline(x=loc, color='r', alpha=0.3, linestyle='--')
    
    # Plot masked regions as background shading
    masked_regions = np.where(sample_mask == 1)[0]
    if len(masked_regions) > 0:
        ax1.fill_between(timesteps, ax1.get_ylim()[0], ax1.get_ylim()[1],
                        where=sample_mask == 1,
                        color='gray', alpha=0.2, label='Masked Region')
    
    ax1.set_title('Sensor Data with Stroke Labels and Mask')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True)
    
    # Plot stroke labels and mask
    ax2.plot(timesteps, sample_stroke, 'r-', label='Stroke Label', linewidth=2)
    ax2.plot(timesteps, sample_mask, 'b--', label='Mask', alpha=0.5)
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_title('Stroke Labels and Mask')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True)
    
    # Add common x-label
    fig.text(0.5, 0.04, 'Timesteps', ha='center')
    
    plt.tight_layout()
    plt.show()

    # Print statistics for this sample
    print(f"\nSample {sample_idx} Statistics:")
    print(f"Total timesteps: {len(sample_stroke)}")
    print(f"Number of strokes: {np.sum(sample_stroke == 1)}")
    print(f"Masked timesteps: {np.sum(sample_mask == 1)}")
    print(f"Strokes in masked regions: {np.sum(sample_stroke[sample_mask == 1] == 1)}")
    print(f"Strokes in unmasked regions: {np.sum(sample_stroke[sample_mask == 0] == 1)}")


