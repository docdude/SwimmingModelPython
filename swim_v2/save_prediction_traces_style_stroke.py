# Load models and predict on recordings
# Save traces to a file for easy post-processing implementations
import os
import pickle
import learning_data
import numpy as np
import tensorflow as tf
import utils

# Increase the threshold for truncation
np.set_printoptions(threshold=np.inf)
data_path = '/Users/juanloya/Documents/SwimmingModelPython/swim_v2/data_modified_users/'

# Path to where we want to save the training results

run_name = 'stroke_2_weighted'
base_path = '/Users/juanloya/Documents/SwimmingModelPython/swim_v2'

results_path = os.path.join(base_path, f'run_{run_name}')
save_path = os.path.join(base_path, f'run_{run_name}')

#results_path = '/Users/juanloya/Documents/SwimmingModelPython/swim_v2/tutorial_save_path_epoch_Parallel7_weighted/'
#save_path = '/Users/juanloya/Documents/SwimmingModelPython/swim_v2/tutorial_save_path_epoch_Parallel2_weighted/'


with open(os.path.join(results_path, '2/data_parameters.pkl'), 'rb') as f:
    data_parameters = pickle.load(f)[0]

# Load users
users_all = utils.folders_in_path(data_path)
users = [u for u in users_all]
users.sort(key=int)
users = data_parameters['users']
users = ['2','6','7','11']
swimming_data = learning_data.LearningData()
swimming_data.load_data(
    data_path=data_path,
    data_columns=data_parameters['data_columns'],
    users=users,
    labels=data_parameters['labels'],
    stroke_labels=data_parameters['stroke_labels']
)

for label in data_parameters['combine_labels'].keys():
    swimming_data.combine_labels(labels=data_parameters['combine_labels'][label], new_label=label)

swimming_data.sliding_window_locs(
    win_len=data_parameters['win_len'], slide_len=data_parameters['slide_len']
)

prediction_traces = {user: {} for user in data_parameters['users']}

for (i, user) in enumerate(users):
    print(f"Working on {user}. {i+1} of {len(users)}")
    model = tf.keras.models.load_model(os.path.join(results_path, user, f'model_{user}.keras'), compile=False)
    recs = list(swimming_data.data_dict['original'][user].keys())

    for (ii, rec) in enumerate(recs):
        print(f"Recording {ii+1} of {len(recs)}")
        win_starts = swimming_data.window_locs['original'][user][rec][0]
        win_stops = swimming_data.window_locs['original'][user][rec][1]
      #  windows = np.zeros(
       #     (len(win_starts), swimming_data.win_len, len(swimming_data.data_columns) + len(swimming_data.stroke_labels))
       # )
        windows = np.zeros(
            (len(win_starts), swimming_data.win_len, len(swimming_data.data_columns))
        )
        # Initialize arrays based on label_type
        num_classes = len(data_parameters['labels'])
        if data_parameters['label_type'] == 'sparse':
            y_true_windows_swim_style = np.zeros(len(windows))  # Sparse (integer) labels
        else:
            y_true_windows_swim_style = np.zeros((len(windows), 5))
        
        #y_true_windows_swim_style_maj = np.zeros(len(windows))
        y_true_windows_strokes = np.zeros((len(win_starts), swimming_data.win_len, 1))

        for iii in range(len(win_starts)):
            win_start = win_starts[iii]
            win_stop = win_stops[iii]
            window = swimming_data.data_dict['original'][user][rec][swimming_data.data_columns].values[win_start:win_stop+1, :]
            window_norm = swimming_data.normalize_window(window, norm_type=data_parameters['window_normalization'])
            win_stroke_labels = swimming_data.data_dict['original'][user][rec][swimming_data.stroke_labels].values[win_start:win_stop+1, :]

            windows[iii] = window
            win_labels = swimming_data.data_dict['original'][user][rec]['label'].values[win_start:win_stop+1]
            win_label_cat, majority_label = swimming_data.get_window_label(
                win_labels, label_type=data_parameters['label_type'], majority_thresh=data_parameters['majority_thresh']

            )

            # Store labels according to label_type
            if data_parameters['label_type'] == 'sparse':
                y_true_windows_swim_style[iii] = majority_label
            else:
                y_true_windows_swim_style[iii, :] = win_label_cat
            y_true_windows_strokes[iii, :, :] = win_stroke_labels  # Match dimensions

        # Prepare windows for model prediction
        windows = windows.reshape((windows.shape[0], windows.shape[1], windows.shape[2], 1))
        y_pred_windows = model.predict(windows)
        y_pred_windows_swim_style = y_pred_windows[0]
        y_pred_windows_strokes = y_pred_windows[1]

        # Validate shapes
        print(f"y_pred_windows_swim_style shape: {y_pred_windows_swim_style.shape}")
        print(f"y_pred_windows_strokes shape: {y_pred_windows_strokes.shape}")

        # Process stroke predictions
        y_pred_windows_strokes = y_pred_windows_strokes.squeeze(-1)  # Convert to shape (batch_size, 180)
        y_pred_windows_strokes_flat = y_pred_windows_strokes.flatten()  # Flatten across all windows

        # Dynamically calculate a threshold for strokes
        #stroke_threshold = np.percentile(y_pred_windows_strokes_flat, 99)  # Example: 95th percentile
        stroke_threshold = np.max(y_pred_windows_strokes_flat)  # Example: 95th percentile

        y_pred_windows_strokes_binary = (y_pred_windows_strokes > stroke_threshold-0.005).astype(int)  # Apply threshold

        # Resample to match raw data length
        y_true_raw_swim_style = swimming_data.data_dict['original'][user][rec]['label'].values
        y_true_raw_strokes = swimming_data.data_dict['original'][user][rec]['stroke_labels'].values

        win_mids = win_starts + (win_stops - win_starts) / 2
        x = win_mids
        x_new = np.arange(0, len(y_true_raw_swim_style))
        # Handle swim style predictions based on label_type
        if data_parameters['label_type'] == 'sparse':
            # Convert softmax outputs to class indices for sparse
            y_pred_windows_swim_style = np.argmax(y_pred_windows_swim_style, axis=1)
            y_pred_raw_swim_style = utils.resample(x, y_pred_windows_swim_style, x_new, kind='nearest')
        else:
            # For one-hot, resample the probabilities
            y_pred_raw_swim_style = utils.resample(x, y_pred_windows_swim_style.T, x_new, kind='nearest').T
       # y_pred_raw_swim_style = utils.resample(x, y_pred_windows_swim_style.T, x_new, kind='nearest').T
        #y_pred_raw_strokes = utils.resample_binary_predictions(x, y_pred_windows_strokes_binary, x_new)
        y_pred_raw_strokes_binary = utils.resample(x, y_pred_windows_strokes_binary.T, x_new, kind='linear').T

        prediction_traces[user][rec] = {
            'window': {
                'swim_style': {'true': y_true_windows_swim_style, 'pred': y_pred_windows_swim_style},
                'strokes': {'true': y_true_windows_strokes, 'pred': y_pred_windows_strokes_binary}
            },
            'raw': {
                'swim_style': {'true': y_true_raw_swim_style, 'pred': y_pred_raw_swim_style},
                'strokes': {'true': y_true_raw_strokes, 'pred': y_pred_raw_strokes_binary}
            }
        }
        # Detect prominent stroke indices
        #raw_stroke_indices = utils.find_adaptive_stroke_indices(win_starts, win_stops, y_pred_windows_strokes, global_percentile=80, local_margin=0.001)

        # Output the detected raw stroke indices
        #print("Detected stroke indices:", raw_stroke_indices)

        print(f"x shape: {x.shape}, x_new shape: {x_new.shape}, y_pred_windows_strokes_flat shape: {y_pred_windows_strokes_flat.shape}")
        print(f"stroke_threshold: {stroke_threshold}")
        print(f"y_pred_windows_strokes_binary unique values: {np.unique(y_pred_windows_strokes_binary)}")
        print(f"y_true_windows_swim_style shape: {y_true_windows_swim_style.shape}")
        print(f"y_pred_windows_swim_style shape: {y_pred_windows_swim_style.shape}")

        print(f"y_true_windows_strokes shape: {y_true_windows_strokes.shape}")
        print(f"y_pred_windows_strokes shape: {y_pred_windows_strokes_binary.shape}")


        print(f"y_true_raw_swim_style shape: {y_true_raw_swim_style.shape}")
        print(f"y_pred_raw_swim_style shape: {y_pred_raw_swim_style.shape}")
        print(f"y_true_raw_strokes shape: {y_true_raw_strokes.shape}")
        print(f"y_pred_raw_strokes_binary shape: {y_pred_raw_strokes_binary.shape}")

with open(os.path.join(save_path, 'prediction_traces_best.pkl'), 'wb') as f:
    pickle.dump([prediction_traces], f)
print(f"Saved predictions to {os.path.join(save_path, 'prediction_traces_best.pkl')}")
