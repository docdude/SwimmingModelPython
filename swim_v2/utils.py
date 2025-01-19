import numpy as np
import pandas as pd
import os
import csv
import scipy.interpolate
import scipy.stats
import constants
from scipy.signal import find_peaks
from sklearn.utils import class_weight


def write_latex_confmat(cm, labels, is_integer=False):
    """
    Write confusion matrix into latex table
    :param cm: Two-dimensional confusion matrix
    :param labels: The labels in the confusion matrix
    :param is_integer: A boolean set to True if the values are integers
    :return: The confusion matrix in latex form
    """
    header = '\\begin{tabular}{' + 'c'*(len(labels)+1) + '}\n'
    footer = '\\end{tabular}'
    text = header
    text = text + '&' + '&'.join(label for label in labels) + '\\\\\n'
    for (i, label) in enumerate(labels):
        if is_integer:
            new_line = label + '&' + '&'.join([str(int(v)) for v in cm[i, :]]) + '\\\\\n'
        else:
            new_line = label + '&' + '&'.join(['%.1f' % v for v in cm[i, :]]) + '\\\\\n'
        text = text + new_line
    text = text + footer
    return text


def downsample_dataframe(df, skipstep):
    """
    Down-sample a pandas dataframe. Mainly used for plots.
    :param df: The pandas dataframe
    :param skipstep: The down-sampling factor
    :return:
    """
    df_new = df.iloc[::skipstep]
    x = np.arange(len(df['label'].values))
    x_new = x[::skipstep]
    for (i, col) in enumerate(df.columns):
        y = df[col].values
        if col == 'timestamp':
            continue
        if col == 'label':
            kind = 'nearest'
        else:
            kind = 'cubic'
        y_new = resample(x, y, x_new, kind=kind)
        df_new[col] = y_new
    return df_new


def start_stop(x):
    """
    Start-stop samples where the values in x change
    :param x: An array of values
    :return: Two numpy arrays containing the start and stop sample locations.
    """
    ix = np.where(x == 1)[0]
    if len(ix) == 0:
        starts = np.array([])
        stops = np.array([])
        return starts, stops
    ix_diff = ix[1:] - ix[:-1]
    ix_diff_jump = np.where(ix_diff > 1)[0]
    starts = np.append(ix[0], ix[ix_diff_jump + 1])
    stops = np.append(ix[ix_diff_jump] + 1, ix[-1] + 1)
    return starts, stops


def unclose(x, open_size=100):
    """
    Opening operation
    :param x: An array of binary values
    :param open_size: Opening threshold
    :return: An array with the values in x after opening
    """
    y = np.copy(x)
    for i in range(len(x)):
        ix_1 = i
        ix_2 = np.min([len(x)+1, i+open_size+1])
        xwin = x[ix_1:ix_2]
        if x[ix_1] == 0:
            ix_uno = np.where(xwin == 1)[0]
            if len(ix_uno) > 0:
                if 0 in xwin[ix_uno[0]:]:
                    ix_end = ix_1 + ix_uno[0] + ix_uno[-1]
                    y[ix_1:ix_end] = 0
    return y


def close(x, close_size=100):
    """
    Opening operation
    :param x: An array of binary values
    :param close_size: Closing threshold
    :return: An array with the values in x after closing
    """
    y = np.copy(x)
    for i in range(len(x)):
        ix_1 = i
        ix_2 = np.min([len(x)+1, i+close_size+1])
        xwin = x[ix_1:ix_2]
        if x[ix_1] == 1:
            ix_null = np.where(xwin == 0)[0]
            if len(ix_null) == 0:
                y[i] = x[i]
            elif 1 in xwin[ix_null[0]:]:
                ix_uno = np.where(xwin[ix_null[0]:] == 1)[0]
                ix_end = ix_1 + ix_null[0] + ix_uno[-1]
                y[ix_1:ix_end] = 1
            else:
                y[i] = x[i]
    return y


def write_confusion_matrix(cm, labels):
    """
    Write confusion matrix to text. Use to get a cleaner output
    :param cm: Two-dimensional confusion matrix
    :param labels: A list of labels
    :return: The confusion matrix in text format
    """
    m = 15
    o = " ".ljust(m)
    for label in labels:
        o = o + constants.LABEL_NAMES[label].ljust(m)
    o = o + "\n"
    for (i, label) in enumerate(labels):
        o = o + constants.LABEL_NAMES[label].ljust(m)
        for j in range(len(labels)):
            val = cm[i, j]
            if val - int(val) == 0:
                val_str = str(int(val))
            else:
                val_str = "%.1f" % cm[i, j]
            o = o + val_str.ljust(m)
        o = o + "\n"
    return o

def write_confusion_matrix_stroke(cm, labels):
    """
    Write confusion matrix to text. Use to get a cleaner output
    :param cm: Two-dimensional confusion matrix
    :param labels: A list of labels
    :return: The confusion matrix in text format
    """
    m = 15
    o = " ".ljust(m)
    for label in labels:
        o = o + constants.STROKE_LABELS[label].ljust(m)
    o = o + "\n"
    for (i, label) in enumerate(labels):
        o = o + constants.STROKE_LABELS[label].ljust(m)
        for j in range(len(labels)):
            val = cm[i, j]
            if val - int(val) == 0:
                val_str = str(int(val))
            else:
                val_str = "%.1f" % cm[i, j]
            o = o + val_str.ljust(m)
        o = o + "\n"
    return o

def normalize_confusion_matrix(cm):
    """
    Normalize confusion matrix w.r.t. the class size
    :param cm: Two-dimensional confusion matrix
    :return:
    """
    cmn = np.zeros(cm.shape)
    label_count = np.sum(cm, axis=1)
    for (i, c) in enumerate(label_count):
        if c != 0:
            cmn[i, :] = cm[i, :] / c
    return cmn


def normalize_range(x):
    """
    Normalize range, i.e. to  0, 1
    :param x: An array of values
    :return: x normalized
    """
    max_val = np.max(x)
    min_val = np.min(x)
    return (x-min_val) / (max_val-min_val)


def detrend(x, window_length=600, return_trend=False):
    """
    Remove trend form an array
    :param x: An array of values
    :param window_length: The length used to compute the moving average trend
    :param return_trend: Boolean indicating whether to return the trend or x detrended
    :return: x detrended or the trend
    """
    tail_length = np.floor(window_length / 2)
    nose_length = np.ceil(window_length / 2)
    trend = np.zeros(len(x))
    for i in range(len(x)):
        if i < tail_length:
            first_ix = 0
        else:
            first_ix = int(i - tail_length)
        if i > len(x) - nose_length:
            last_ix = len(x)
        else:
            last_ix = int(i + nose_length)
        trend[i] = np.mean(x[first_ix: last_ix])
    if not return_trend:
        return x - trend
    else:
        return trend


def diff(x):
    """
    The derivative on an array
    :param x:   An array of values
    :return:    An array for the derivative of x
    """
    return x[1:] - x[:-1]


def dirs_in_path(p):
    """
    List all directories in a path
    :param p: A full path
    :return: A list of strings of names of all directories in the path
    """
    return [d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))]


def load_recording(file_path, category='raw'):
    """
    Load a recording into a Pandas DataFrame
    :param file_path:   Path to .csv file containing swimming data
    :param category:    = 'raw', 'labeled' or 'processed'
    :return: Recording data in a Pandas DataFrame. If category is 'raw' or 'labeled', the header and footer are also
             returned
    """
    if category == 'raw' or category == 'labeled':
        df = pd.read_csv(file_path, sep='; ', header=None, skiprows=[0], skipfooter=1)
        df.columns = constants.LABELED_COL_NAMES[0: len(df.columns)]
        header = list(pd.read_csv(file_path, sep='; ', nrows=1).columns)
        with open(file_path, 'r') as f:
            footer = list(csv.reader(f))[-1]
            footer = footer[0].split("; ")
        return df, header, footer
    if category == 'processed':
        df = pd.read_csv(file_path)
        return df


def resample(x, y, x_new, kind='cubic'):
    """
    A simple wrapper for interp1d
    :param x: Original timestamps
    :param y: Original values
    :param x_new: New timestamps
    :param kind: interpolation type
    :return: The values in y evaluated at x_new
    """
    f = scipy.interpolate.interp1d(x, y, kind=kind, bounds_error=False, fill_value=np.nan)
    y_new = f(x_new)
    return y_new

def resample_binary_predictions(x, y_binary, x_target):
    """
    Resample binary predictions to match new timestamps.
    :param x: Original timestamps
    :param y_binary: Binary predictions corresponding to original timestamps
    :param x_target: Target timestamps
    :return: Resampled binary predictions for the target timestamps
    """
    binary_values = np.zeros_like(x_target)
    for original_idx, value in enumerate(y_binary):
        # Ensure `value` is binary (scalar 0 or 1)
        if np.any(value == 1):  # Handles arrays
            # Find the closest timestamp in the new x_target
            closest_idx = np.abs(x_target - x[original_idx]).argmin()
            binary_values[closest_idx] = 1
    return binary_values

def map_window_predictions_to_raw_indices(win_starts, win_stops, y_pred_windows_strokes, threshold=0.1):
    """
    Map window-level predictions to raw data indices.
    :param win_starts: Start indices of the sliding windows.
    :param win_stops: Stop indices of the sliding windows.
    :param y_pred_windows_strokes: Predicted stroke probabilities (shape: num_windows x win_len).
    :param threshold: Threshold to classify strokes.
    :return: List of raw indices where strokes were detected.
    """
    raw_stroke_indices = []

    for win_idx, (start, stop) in enumerate(zip(win_starts, win_stops)):
        window_predictions = y_pred_windows_strokes[win_idx]
        #stroke_indices_in_window = np.where(window_predictions >= 0.019)[0]
        stroke_indices_in_window = np.argmax(window_predictions)[0] >=threshold

        raw_indices = start + stroke_indices_in_window  # Map to raw indices
        raw_stroke_indices.extend(raw_indices)

    return np.array(raw_stroke_indices)

def find_adaptive_stroke_indices(win_starts, win_stops, y_pred_windows_strokes, global_percentile=99.9, local_margin=0.001):
    """
    Find stroke indices adaptively based on window-specific and global thresholds.
    :param win_starts: Start indices of the sliding windows.
    :param win_stops: Stop indices of the sliding windows.
    :param y_pred_windows_strokes: Predicted stroke probabilities (shape: num_windows x win_len).
    :param global_percentile: Percentile to determine the global threshold.
    :param local_margin: Margin to adjust the local threshold based on the max value in the window.
    :return: Array of raw indices where strokes were prominently detected.
    """
    raw_stroke_indices = []
    
    # Compute max prediction values for each window
    max_values_per_window = np.max(y_pred_windows_strokes, axis=1)
    
    # Set a global threshold based on the 99.9th percentile of max values across all windows
    global_threshold = np.percentile(max_values_per_window, global_percentile)
    print(f"Global threshold for stroke detection: {global_threshold:.4f}")
    unique_indices = set()
    for win_idx, (start, stop) in enumerate(zip(win_starts, win_stops)):
        window_predictions = y_pred_windows_strokes[win_idx]  # Predictions for the current window
        
        # Find the index and value of the maximum prediction in the window
        max_idx = np.argmax(window_predictions)
        max_value = window_predictions[max_idx]
        
        # Compute a local threshold based on the max value in this window
        local_threshold = max(global_threshold, max_value - local_margin)
        
        # Validate the max prediction against the local threshold
        if max_value >= local_threshold:
            # Add the raw index of the detected stroke to the list
            raw_index = start + max_idx
            if raw_index not in unique_indices:
                unique_indices.add(raw_index)
                raw_stroke_indices.append(raw_index)

    return np.array(raw_stroke_indices)

def time_scale_dataframe(df, factor, time_col, label_col, static_cols=None, binary_cols=None):
    """
    Time-scale a dataframe with special handling for binary columns like `stroke_labels`.
    :param df: A pandas dataframe
    :param factor: Time-scaling factor
    :param time_col: The column name of timestamps
    :param label_col: The column name of activity labels
    :param static_cols: List of column names that should not be interpolated
    :param binary_cols: List of column names that contain binary data
    :return: Time-scaled dataframe
    """
    if static_cols is None:
        static_cols = []
    if binary_cols is None:
        binary_cols = []
    
    # Identify columns for interpolation
    data_cols = [col for col in df.columns if col not in [time_col, label_col] + static_cols + binary_cols]
    df_new = pd.DataFrame(columns=df.columns)
    
    # Adjust timestamp scaling
    dt = df[time_col].values[1] - df[time_col].values[0]
    dts = dt / factor
    t = df[time_col].values
    ts = t[0] + np.arange(len(t)) * dts
    t_target = np.arange(t[0], ts[-1], dt)
    df_new[time_col] = t_target
    
    # Interpolate continuous data columns
    for col in data_cols:
        y = df[col].values
        df_new[col] = resample(ts, y, t_target, 'cubic')
    
    # Interpolate label column using nearest neighbor
    df_new[label_col] = resample(ts, df[label_col].values, t_target, 'nearest')
    
    # Handle static columns
    for col in static_cols:
        if col in df:
            df_new[col] = resample(t, df[col].values, t_target, 'nearest')
        else:
            df_new[col] = np.nan
    
    # Handle binary columns
    for col in binary_cols:
        if col in df:
            binary_values = np.zeros(len(t_target), dtype=int)
            original_indices = np.where(df[col].values == 1)[0]  # Get original stroke indices
            scaled_indices = np.round(original_indices / factor).astype(int)  # Scale indices
            
            # Assign stroke labels to the closest indices in the new time-scaled data
            for scaled_idx in scaled_indices:
                if 0 <= scaled_idx < len(binary_values):
                    binary_values[scaled_idx] = 1
            df_new[col] = binary_values
        else:
            df_new[col] = np.nan  # Handle missing binary columns
    
    return df_new



def get_sample_weights_new(y_cat):
    """
    Compute sample weights based on class size
    :param y_cat: Labels in categorical form
    :return: The sample weights
    """
    class_weights = np.zeros(y_cat.shape[1])
    for i in range(y_cat.shape[1]):
        if np.sum(y_cat[:, i]) == 0:
            class_weights[i] = 0
        else:
            class_weights[i] = 1/np.sum(y_cat[:, i])
    y_sample_weights = np.sum(y_cat * class_weights, axis=1)/np.sum(y_cat, axis=1)
    y_sample_weights = y_sample_weights/np.sum(y_sample_weights)*len(y_sample_weights)
    return y_sample_weights


def from_categorical(y_cat):
    """
    From categorical to normal labeling
    :param y_cat: Two dimensional array of categorical labels
    :return: An array with normal labeling
    """
    y = np.argmax(y_cat, axis=1)
    return y.astype(int)


def folders_in_path(p):
    """
    Get folders in path
    :param p: Path
    :return: List of folder names in the path
    """
    return [d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))]

def analyze_class_imbalance(y_stroke_val):
    y_stroke_val_flat = y_stroke_val.flatten()
    class_counts = np.bincount(y_stroke_val_flat)
    class_weights = len(y_stroke_val_flat) / (len(np.unique(y_stroke_val_flat)) * class_counts)
    
    print("  Class Counts:", class_counts)
    print("  Computed Class Weights:", class_weights)
    
    # Entropy of label distribution
    y_stroke_val_probs = class_counts / len(y_stroke_val_flat)
    entropy = -np.sum(y_stroke_val_probs * np.log2(y_stroke_val_probs + 1e-10))
    print("  Label Distribution Entropy:", entropy)
    return class_weights

def create_balanced_sampler(y_stroke_val, method='inverse_freq'):
    """
    Create sample weights with multiple methods
    """
    y_stroke_val_flat = y_stroke_val.flatten()
    if method == 'inverse_freq':
        # Your current method
        class_counts = np.bincount(y_stroke_val_flat)
        weights = 1. / class_counts
        sample_weights = weights[y_stroke_val_flat]
        sample_weights /= sample_weights.sum()
        sample_weights = sample_weights.reshape(y_stroke_val.shape)
        return sample_weights
    
    elif method == 'sklearn':
        # Scikit-learn method
        from sklearn.utils.class_weight import compute_sample_weight
        
        sample_weights = compute_sample_weight('balanced', y_stroke_val_flat)
        sample_weights = sample_weights.reshape(y_stroke_val.shape)
        return sample_weights
    
    else:
        raise ValueError("Invalid method. Choose 'inverse_freq' or 'sklearn'")


def calculate_stroke_label_distribution(label_user_dict, swimming_data, data_type="training", exclude_label=0):
    """
    Calculate stroke label distributions in the provided label-user dictionary using get_windows_dict,
    excluding a specified class label from the label-user dictionary. Also, calculate the mean of stroke_label = 1
    and the initial bias for the stroke_label_output layer.

    Args:
        label_user_dict (dict): A dictionary where keys are swim styles (labels) and values are lists of user data.
        swimming_data (object): The swimming_data instance containing the get_windows_dict method.
        data_type (str): "training" or "validation" to indicate the data source.
        exclude_label (int): Label to exclude from the label-user dictionary.

    Returns:
        dict: Stroke label proportions for the provided data dictionary.
        float: Mean of stroke_label = 1.
        float: Initial bias for the stroke_label_output layer.
    """
    from collections import Counter

    # Filter out the excluded label from the label_user_dict
    filtered_dict = {label: users for label, users in label_user_dict.items() if label != exclude_label}

    # Extract stroke labels using the get_windows_dict method
    try:
        _, _, _, y_stroke_val = swimming_data.get_windows_dict(filtered_dict)
    except Exception as e:
        print(f"Error extracting stroke labels for {data_type} data: {e}")
        return {}, None, None

    # Flatten the stroke labels for counting
    stroke_labels_flat = y_stroke_val.flatten()

    # Count occurrences of each stroke label
    label_counter = Counter(stroke_labels_flat)
    total_strokes = len(stroke_labels_flat)

    # Calculate probabilities
    probabilities = {label: count / total_strokes for label, count in label_counter.items()}

    # Calculate the mean of stroke_label = 1
    stroke_label_1_mean = stroke_labels_flat.mean()

    # Calculate the initial bias for the stroke_label_output layer
    pos_count = label_counter.get(1.0, 0)
    neg_count = label_counter.get(0.0, 0)
    pos_prob = pos_count / total_strokes if total_strokes > 0 else 0
    if pos_prob > 0 and pos_prob < 1:
        initial_bias = np.log([pos_prob / (1 - pos_prob)])
    else:
        initial_bias = None  # Can't compute log odds for probabilities of 0 or 1

    # Print the distribution, mean, and initial bias
    print(f"\n{data_type.capitalize()} Stroke Label Distribution (Excluding Swim Style Label {exclude_label}):")
    for label, prob in probabilities.items():
        print(f"  Stroke Label {label}: {prob:.4f}")

    print(f"  Mean of Stroke Label = 1 for {data_type} set: {stroke_label_1_mean:.4f}")
    if initial_bias is not None:
        print(f"  Initial bias for stroke_label_output: {initial_bias[0]:.4f}")
    else:
        print("  Initial bias could not be calculated (check label distribution).")

    class_weights = analyze_class_imbalance(y_stroke_val)
    print("  Class Weights:", class_weights)
    weights = class_weight.compute_class_weight('balanced', classes=np.unique(stroke_labels_flat.astype(int)), y=stroke_labels_flat.astype(int))
    sklearn_class_weights = dict(enumerate(weights))
    print("  Sklearn Class Weights:", sklearn_class_weights)
    sample_weights = create_balanced_sampler(y_stroke_val, method='sklearn')
    print("  Sample Weights:", sample_weights)

    return probabilities, stroke_label_1_mean, initial_bias, class_weights


def verify_masking(y_stroke, mask, name="Dataset", batch_size=64):
    """
    Verifies masking strategy with batch-wise analysis
    
    Args:
        y_stroke: shape (samples, timesteps, 1)
        mask: shape (samples, timesteps, 1)
        name: string identifier for the dataset
        batch_size: size of batches to analyze (default 64)
    """
    print(f"\n{name} Masking Analysis:")
    print(f"Total shape - Stroke labels: {y_stroke.shape}, Mask: {mask.shape}")
    
    # Calculate number of complete batches
    num_batches = len(y_stroke) // batch_size
    
    # Batch-wise statistics
    batch_stats = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        
        # Get batch data
        batch_y_stroke = y_stroke[start_idx:end_idx]
        batch_mask = mask[start_idx:end_idx]
        
        # Analyze batch
        total_timesteps = np.prod(batch_mask.shape[:-1])
        active_timesteps = np.sum(batch_mask == 1)
        ignored_timesteps = np.sum(batch_mask == 0)
        
        # Analyze strokes
        strokes_in_active = batch_y_stroke[batch_mask == 1]
        strokes_in_ignored = batch_y_stroke[batch_mask == 0]
        stroke_count_ignored = np.sum(strokes_in_ignored == 1)
        total_strokes = np.sum(batch_y_stroke == 1)
        
        batch_stats.append({
            'active_ratio': active_timesteps/total_timesteps,
            'ignored_ratio': ignored_timesteps/total_timesteps,
            'strokes_ignored_ratio': stroke_count_ignored/total_strokes if total_strokes > 0 else 0
        })
    
    # Print average batch statistics
    avg_stats = {k: np.mean([b[k] for b in batch_stats]) for k in batch_stats[0].keys()}
    std_stats = {k: np.std([b[k] for b in batch_stats]) for k in batch_stats[0].keys()}
    
    print(f"\nAverage Batch Statistics (over {num_batches} batches):")
    print(f"Active timesteps: {avg_stats['active_ratio']*100:.2f}% ± {std_stats['active_ratio']*100:.2f}%")
    print(f"Ignored timesteps: {avg_stats['ignored_ratio']*100:.2f}% ± {std_stats['ignored_ratio']*100:.2f}%")
    print(f"Strokes in ignored regions: {avg_stats['strokes_ignored_ratio']*100:.2f}% ± {std_stats['strokes_ignored_ratio']*100:.2f}%")
    
    # Also show full dataset statistics for reference
    print(f"\nFull Dataset Statistics:")
    total_timesteps = np.prod(mask.shape[:-1])
    print(f"Total timesteps: {total_timesteps}")
    print(f"Active timesteps: {np.sum(mask == 1)/total_timesteps*100:.2f}%")
    print(f"Ignored timesteps: {np.sum(mask == 0)/total_timesteps*100:.2f}%")
    
    total_strokes = np.sum(y_stroke == 1)
    strokes_ignored = np.sum((y_stroke == 1) & (mask == 0))
    print(f"Total strokes: {total_strokes}")
    print(f"Strokes in ignored regions: {strokes_ignored/total_strokes*100:.2f}%")

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

def detect_boundary_peaks(
    current_window_predictions, 
    next_window_predictions, 
    height=0.4, 
    distance=5,
    width=4,
    prominence=0.5
):
    """
    Detect peaks at window boundaries
    
    Parameters:
    -----------
    current_window_predictions : ndarray
        Predictions for the current window (180 samples)
    next_window_predictions : ndarray or None
        Predictions for the next window (180 samples)
    threshold : float, optional
        Minimum prediction value to consider
    min_distance : int, optional
        Minimum distance between peaks
    
    Returns:
    --------
    list: Boundary peaks with global indices
    """
    boundary_peaks = []

    
    # Check start of the next window if it exists
    if next_window_predictions is not None:
        transition_boundary_predictions = np.concatenate([current_window_predictions[-10:], next_window_predictions[:10]])
        dynamic_prominence = calculate_dynamic_prominence(transition_boundary_predictions)
        insights, _, _ = analyze_peak_characteristics(transition_boundary_predictions)

        transition_boundary_peaks, _ = find_peaks(
            transition_boundary_predictions, 
            height=height,
            wlen=20,
            distance=distance,
            width=width,
            prominence=dynamic_prominence
        )
        # Adjust indices to be within the full window (last 5 samples start at index 175)
        boundary_peaks = [p + 170 for p in transition_boundary_peaks]
    
    # Remove duplicates
    #boundary_peaks = sorted(set(boundary_peaks))
    
    return boundary_peaks

def analyze_peak_characteristics(predictions):
    """
    Analyze peak characteristics to inform detection parameters
    
    Args:
        predictions: Model's stroke probability predictions
    
    Returns:
        Dictionary of peak detection insights
    """
    # Detect initial peaks with minimal constraints
    peaks, properties = find_peaks(
        predictions, 
        prominence=0.1,  # Minimal prominence
        width=1,         # Minimal width
        height=0.3,       # Prediction threshold
        rel_height=0.6   # Relative height
    )
    if len(peaks) == 0:  
        return {}, [], {}
    # Analyze peak widths
    peak_widths = properties['widths']
    peak_width_heights = properties['width_heights']
    peak_heights = properties['peak_heights']
    peak_prominences = properties['prominences']
    
    # Comprehensive insights
    insights = {
        # Peak Count Metrics
        'peak_count': len(peaks),
        
        # Width Characteristics
        'avg_peak_width': np.mean(peak_widths),
        'median_peak_width': np.median(peak_widths),
        'peak_width_std': np.std(peak_widths),
        'min_peak_width': np.min(peak_widths),
        'max_peak_width': np.max(peak_widths),

                # Width Characteristics
        'avg_peak_width_height': np.mean(peak_width_heights),
        'median_peak_width_height': np.median(peak_width_heights),
        'peak_width_height_std': np.std(peak_width_heights),
        'min_peak_width_height': np.min(peak_width_heights),
        'max_peak_width_height': np.max(peak_width_heights),
        
        # Height Characteristics
        'avg_peak_height': np.mean(peak_heights),
        'median_peak_height': np.median(peak_heights),
        'peak_height_std': np.std(peak_heights),
        'min_peak_height': np.min(peak_heights),
        'max_peak_height': np.max(peak_heights),
        
        # Prominence Characteristics
        'avg_peak_prominence': np.mean(peak_prominences),
        'median_peak_prominence': np.median(peak_prominences),
        'peak_prominence_std': np.std(peak_prominences),
        'min_peak_prominence': np.min(peak_prominences),
        'max_peak_prominence': np.max(peak_prominences)
    }
    
    # Detailed Printing
    print("\n--- Peak Detection Insights ---")
    print(f"Total Peaks Detected: {insights['peak_count']}")
    
    print("\nWidth Characteristics:")
    print(f"  Average Width:    {insights['avg_peak_width']:.2f}")
    print(f"  Median Width:     {insights['median_peak_width']:.2f}")
    print(f"  Width Std Dev:    {insights['peak_width_std']:.2f}")
    print(f"  Min Width:        {insights['min_peak_width']:.2f}")
    print(f"  Max Width:        {insights['max_peak_width']:.2f}")

    print("\nWidth_Height Characteristics:")
    print(f"  Average Width_Height:    {insights['avg_peak_width_height']:.2f}")
    print(f"  Median Width_Height:     {insights['median_peak_width_height']:.2f}")
    print(f"  Width_Height Std Dev:    {insights['peak_width_height_std']:.2f}")
    print(f"  Min Width_Height:        {insights['min_peak_width_height']:.2f}")
    print(f"  Max Width_Height:        {insights['max_peak_width_height']:.2f}")
    
    print("\nHeight Characteristics:")
    print(f"  Average Height:   {insights['avg_peak_height']:.2f}")
    print(f"  Median Height:    {insights['median_peak_height']:.2f}")
    print(f"  Height Std Dev:   {insights['peak_height_std']:.2f}")
    print(f"  Min Height:       {insights['min_peak_height']:.2f}")
    print(f"  Max Height:       {insights['max_peak_height']:.2f}")
    
    print("\nProminence Characteristics:")
    print(f"  Average Prominence: {insights['avg_peak_prominence']:.2f}")
    print(f"  Median Prominence:  {insights['median_peak_prominence']:.2f}")
    print(f"  Prominence Std Dev: {insights['peak_prominence_std']:.2f}")
    print(f"  Min Prominence:     {insights['min_peak_prominence']:.2f}")
    print(f"  Max Prominence:     {insights['max_peak_prominence']:.2f}")
    

    
    return insights, peaks, properties


def detect_strokes_dynamic_prominence(
    predictions, 
    height=0.2, 
    width=4,
    min_distance=8, 
    dynamic_prominence=0.5
):
    """
    Detect strokes in predictions with dynamic prominence calculation and boundary checks
    """
    # Analyze peak characteristics within the window
    #insights, peaks, properties = analyze_peak_characteristics(predictions)
    # Find peaks within the window
    window_peaks, _ = find_peaks(
        predictions, 
        height=height,
        width=width,
        distance=min_distance,
        prominence=(dynamic_prominence, None)
    )
    
    # Combine and sort peaks
    all_peaks = sorted(set(list(window_peaks) ))
    
    return all_peaks

def analyze_predictions(y_pred_windows_strokes, win_starts, win_stops):
    """
    Analyze multiple prediction windows
    
    Parameters:
    - predictions_list: List of prediction arrays
    
    Returns:
    - Detected stroke information
    """
    stroke_windows = []
    overall_insights = {}
    
    for win_idx, (start, stop) in enumerate(zip(win_starts, win_stops)):
        curr_window_predictions = y_pred_windows_strokes[win_idx]  # Predictions for the current window
        
        print(f"\n--- Window {win_idx} ---")
                # Determine next window start (if exists)
        next_win_start = start + 180 if (start + 180)  < win_stops.max() else None
       
        # Skip boundary detection for the last window if no next window exists
        if next_win_start is None:
            next_window_predictions = None
            next_window_start = None
        else:
            next_window_predictions = y_pred_windows_strokes[win_idx + 1]
            next_window_start = next_win_start

        # Dynamically calculate prominence
        dynamic_prominence = calculate_dynamic_prominence(curr_window_predictions)
        insights, _, _ = analyze_peak_characteristics(curr_window_predictions)

        window_peaks, _ = find_peaks(
            curr_window_predictions, 
            height=0.2,
            distance=8,
            #width=4,
            prominence=dynamic_prominence
        )

        # Boundary peak detection 
        boundary_peaks = []
        if next_window_predictions is not None and next_window_start is not None:
            boundary_peaks = detect_boundary_peaks(
                curr_window_predictions,
                next_window_predictions, 
                height=0.3, 
                distance=8,
                prominence=dynamic_prominence

            )    
        # Detect strokes within the current window
 #       insights, peaks = detect_strokes_dynamic_prominence(
  #          np.array(window_predictions), 
   #         threshold=0.1,
    #        min_distance=5,
     #       dynamic_prominence=dynamic_prominence,
      #      next_window_predictions=next_window_predictions,
       #     next_window_start=next_window_start
      #  )
        # Combine and sort peaks
        all_peaks = sorted(set(list(window_peaks) + list(boundary_peaks)))

        # Accumulate insights 
        if not overall_insights:
            overall_insights = insights
        
        if len(all_peaks) > 0:
            stroke_windows.append({
                'window_index': win_idx,
                'global_row_index': start,
                'peaks': all_peaks,  # Combined peaks
                'peak_indices': [p + start for p in all_peaks]  # Adjust to global indices
            })

    return overall_insights, stroke_windows


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def get_peaks(arr, window):
    maxss = np.argmax(rolling_window(arr, window), axis=1)
    return np.where(maxss == 0)[0]

def main():
    print("Main")


if __name__ == '__main__':
    main()
