import os
import pickle
import numpy as np
import tensorflow as tf
import learning_data
import utils
import scipy.signal  # For peak detection
import matplotlib.pyplot as plt


def extract_window_data_and_predictions(swimming_data, model, data_parameters, user, rec):
    """
    Extract window data, predictions, and sensor values
    
    Parameters:
    -----------
    swimming_data : LearningData object
        The loaded swimming data
    model : tensorflow.keras.Model
        The trained CNN model
    user : str
        User identifier
    rec : str
        Recording identifier
    
    Returns:
    --------
    dict containing window information
    """
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

def detect_peaks_with_dynamic_threshold(sensor_data, predicted_style, base_prominence=0.5):
    """
    Detect peaks with dynamic thresholding and style-specific rules
    
    Parameters:
    -----------
    sensor_data : np.ndarray
        Sensor magnitude data
    predicted_style : int
        Predicted swimming style
    base_prominence : float
        Base prominence threshold for peak detection
    
    Returns:
    --------
    int : Number of filtered peaks
    """
    # Calculate dynamic threshold based on signal characteristics
    dynamic_prominence = base_prominence * (np.std(sensor_data) / np.mean(sensor_data))
    
    # Style-specific adjustments
    if predicted_style == 0:  # Null/Turn style
        return 0  # No strokes for null/turn
    elif predicted_style == 1:  # Front Crawl
        dynamic_prominence *= 1.2  # Increase threshold for front crawl
    elif predicted_style == 2:  # Breaststroke
        dynamic_prominence *= 0.8  # Decrease threshold for breaststroke
    elif predicted_style == 3:  # Backstroke
        dynamic_prominence *= 1.0  # Keep base threshold for backstroke
    elif predicted_style == 4:  # Butterfly
        dynamic_prominence *= 1.5  # Increase threshold for butterfly
    
    # Detect peaks
    peaks, properties = scipy.signal.find_peaks(sensor_data, prominence=dynamic_prominence)
    
    # Limit maximum peaks per window based on style
    max_peaks = 3 if predicted_style in [1, 4] else 2  # Allow more peaks for styles with rapid strokes
    if len(peaks) > max_peaks:
        sorted_peak_indices = np.argsort(properties['prominences'])[-max_peaks:]
        peaks = peaks[sorted_peak_indices]
    
    return len(peaks)

def count_strokes_by_style(sensor_windows, predicted_styles, base_prominence=0.5):
    """
    Count strokes by detecting peaks in accelerometer and gyrometer data, filtered by swimming style
    
    Parameters:
    -----------
    sensor_windows : np.ndarray
        Array of sensor windows
    predicted_styles : np.ndarray
        Predicted swimming styles for each window
    base_prominence : float
        Base threshold for peak detection
    
    Returns:
    --------
    dict : Stroke counts for each swimming style
    """
    # Initialize stroke counts with zeros
    stroke_counts = {style: 0 for style in range(predicted_styles.shape[1])}
    
    for i, window in enumerate(sensor_windows):
        # Compute combined magnitude for accelerometer and gyrometer
        acc_magnitude = np.sqrt(np.sum(window[:, :3]**2, axis=1))  # Assuming ACC_0, ACC_1, ACC_2
        gyro_magnitude = np.sqrt(np.sum(window[:, 3:6]**2, axis=1))  # Assuming GYRO_0, GYRO_1, GYRO_2
        
        # Determine the most likely swimming style
        predicted_style = np.argmax(predicted_styles[i])
        
        # Detect peaks for accelerometer and gyrometer
        acc_peaks = detect_peaks_with_dynamic_threshold(acc_magnitude, predicted_style, 
                                                        base_prominence=base_prominence)
        gyro_peaks = detect_peaks_with_dynamic_threshold(gyro_magnitude, predicted_style, 
                                                         base_prominence=base_prominence)
        
        # Combine peaks with additional filtering
        if predicted_style != 0:  # Exclude null/turn style
            # Use the maximum of acc or gyro peaks
            combined_peaks = max(acc_peaks, gyro_peaks)
            
            # Only count peaks if the style confidence is high
            style_confidence = np.max(predicted_styles[i])
            if style_confidence > 0.7:  # Adjust threshold as needed
                stroke_counts[predicted_style] += combined_peaks
    
    return stroke_counts

def visualize_peaks(sensor_windows, predicted_styles, base_prominence=0.5):
    """
    Visualize sensor data and detected peaks for debugging
    
    Parameters:
    -----------
    sensor_windows : np.ndarray
        Array of sensor windows
    predicted_styles : np.ndarray
        Predicted swimming styles for each window
    base_prominence : float
        Base threshold for peak detection
    """
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    
    for i, window in enumerate(sensor_windows):
        # Compute magnitudes
        acc_magnitude = np.sqrt(np.sum(window[:, :3]**2, axis=1))  # Assuming ACC_0, ACC_1, ACC_2
        gyro_magnitude = np.sqrt(np.sum(window[:, 3:6]**2, axis=1))  # Assuming GYRO_0, GYRO_1, GYRO_2
        
        # Determine predicted style
        predicted_style = np.argmax(predicted_styles[i])
        style_confidence = np.max(predicted_styles[i])
        
        # Calculate dynamic prominence based on the current window's sensor data
        if np.mean(acc_magnitude) > 0:  # Avoid division by zero
            dynamic_prominence = base_prominence * (np.std(acc_magnitude) / np.mean(acc_magnitude))
        else:
            dynamic_prominence = base_prominence  # Fallback to base if mean is zero
        
        # Style-specific adjustments for dynamic prominence
        if predicted_style == 0:  # Null/Turn style
            dynamic_prominence = 0  # No peaks for null/turn
        elif predicted_style == 1:  # Front Crawl
            dynamic_prominence *= 1.2
        elif predicted_style == 2:  # Breaststroke
            dynamic_prominence *= 0.8
        elif predicted_style == 3:  # Backstroke
            dynamic_prominence *= 1.0
        elif predicted_style == 4:  # Butterfly
            dynamic_prominence *= 1.5
        
        # Detect peaks using the dynamic prominence
        acc_peaks, acc_properties = scipy.signal.find_peaks(acc_magnitude, prominence=dynamic_prominence)
        gyro_peaks, gyro_properties = scipy.signal.find_peaks(gyro_magnitude, prominence=dynamic_prominence)
        
        # Plot
        ax[0].clear()
        ax[0].plot(acc_magnitude, label='Acc Magnitude')
        ax[0].plot(acc_peaks, acc_magnitude[acc_peaks], 'ro', label='Acc Peaks')
        ax[0].set_title(f'Window {i}: Accelerometer (Style: {predicted_style}, Confidence: {style_confidence:.2f})')
        ax[0].legend()
        
        ax[1].clear()
        ax[1].plot(gyro_magnitude, label='Gyro Magnitude')
        ax[1].plot(gyro_peaks, gyro_magnitude[gyro_peaks], 'ro', label='Gyro Peaks')
        ax[1].set_title('Gyrometer Magnitude')
        ax[1].legend()
        
        plt.pause(0.1)
    
    plt.tight_layout()
    plt.show()



def main():
    # Load data and model similar to save_prediction_traces.py
    data_path = 'data/processed_30Hz_relabeled'
    results_path = 'tutorial_save_path_epoch100/'
    
    # Load data parameters
    with open(os.path.join(results_path, '22/data_parameters.pkl'), 'rb') as f:
        data_parameters = pickle.load(f)[0]
    
    users = data_parameters['users']
    
    # Initialize LearningData
    swimming_data = learning_data.LearningData()
    swimming_data.load_data(data_path=data_path, 
                             data_columns=data_parameters['data_columns'],
                             users=users, 
                             labels=data_parameters['labels'])
    
    # Combine labels if needed
    for label in data_parameters['combine_labels'].keys():
        swimming_data.combine_labels(labels=data_parameters['combine_labels'][label], 
                                     new_label=label)
    
    swimming_data.sliding_window_locs(win_len=data_parameters['win_len'], 
                                      slide_len=data_parameters['slide_len'])
    
    # Store results
    all_predictions = {}
    

    # Process each user and recording
    for (i, user) in enumerate(users):
        print(f"Working on {user}. {i+1} of {len(users)}")
        
        # Load model
        model = tf.keras.models.load_model(os.path.join(results_path, user, 'model_best.keras'), compile=False)
        
        # Store user predictions
        user_predictions = {}
        
        # Process each recording
        for rec in swimming_data.data_dict['original'][user].keys():
            print(f"Processing recording: {rec}")
            
            # Extract window data and predictions
            window_data = extract_window_data_and_predictions(swimming_data, model, data_parameters, user, rec)
            
            # Count strokes by style
            stroke_counts = count_strokes_by_style(window_data['sensor_windows'], 
                                                   window_data['predicted_windows'], 
                                                   base_prominence=0.5)
            
            user_predictions[rec] = {
                'window_data': window_data,
                'stroke_counts': stroke_counts
            }

            # Visualize peaks
          #  visualize_peaks(window_data['sensor_windows'], 
           #                 window_data['predicted_windows'], 
            #                base_prominence=0.5)
            
        all_predictions[user] = user_predictions
    
    # Optional: Save results
    ...

    # Print some example results
    for user, recordings in all_predictions.items():
        print(f"\nUser: {user}")
        for rec, data in recordings.items():
            print(f"  Recording: {rec}")
            print("  Stroke Counts by Style:", data['stroke_counts'])

if __name__ == '__main__':
    main()
