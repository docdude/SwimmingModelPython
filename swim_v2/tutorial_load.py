# A basic tutorial in how I load a trained model
import cnn_vanilla
import utils
import learning_data
import os
import random as rn
#import keras
import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt

# A path to re-sampled recordings which are organized into folders by user name.
data_path = r'data/processed_30Hz_relabeled'

# Path to training results
results_path = r'tutorial_save_path_epoch100'

# User whose model we want to load
user = '22'

# Get the data parameters used for loading
with open(os.path.join(results_path, user, 'data_parameters.pkl'), 'rb') as f:
    data_parameters = pickle.load(f)[0]

# Fetch the model
model = tf.keras.models.load_model(os.path.join(results_path, user, 'model_best.keras'))

# Load the recordings
swimming_data = learning_data.LearningData()
swimming_data.load_data(data_path=data_path,
                        data_columns=data_parameters['data_columns'],
                        users=[user],
                        labels=data_parameters['labels'],
                        stroke_labels=data_parameters['stroke_labels'])

if data_parameters['combine_labels'] is not None:
    for label in data_parameters['combine_labels'].keys():
        swimming_data.combine_labels(labels=data_parameters['combine_labels'][label], new_label=label)
swimming_data.sliding_window_locs(win_len=data_parameters['win_len'], slide_len=data_parameters['slide_len'])

recs = list(swimming_data.data_dict['original'][user].keys())

prediction_traces = {rec: None for rec in recs}
for (ii, rec) in enumerate(recs):
    print("Recording %d of %d" % (ii + 1, len(recs)))

    win_starts = swimming_data.window_locs['original'][user][rec][0]

    win_stops = swimming_data.window_locs['original'][user][rec][1]

    windows = np.zeros((len(win_starts), swimming_data.win_len, len(swimming_data.data_columns)))
    y_true_windows = np.zeros((len(windows), 5))
    y_true_windows_maj = np.zeros(len(windows))
    for iii in range(len(win_starts)):
        win_start = win_starts[iii]
        win_stop = win_stops[iii]
        window = swimming_data.data_dict['original'][user][rec][swimming_data.data_columns].values[
                 win_start:win_stop + 1, :]
        #print(swimming_data.data_columns)
        window_norm = swimming_data.normalize_window(window, norm_type=data_parameters['window_normalization'])
        windows[iii] = window_norm
        win_labels = swimming_data.data_dict['original'][user][rec]['label'].values[win_start: win_stop + 1]
        win_label_cat, majority_label = swimming_data.get_window_label(win_labels, label_type='proportional',
                                                                       majority_thresh=0.4)
        win_stroke_labels = swimming_data.data_dict['original'][user][rec]['stroke_labels'].values[win_start: win_stop + 1]
        y_true_windows[iii, :] = win_label_cat
        y_true_windows_maj[iii] = majority_label
    print("preshape")
    print(windows.shape[1])    
    print(windows.shape[2])
    #windows = windows.reshape((windows.shape[0], windows.shape[1], windows.shape[2], 1))
    print("post shape")
    print(windows)
    y_pred_windows = model.predict(windows)
   
    y_true_raw = swimming_data.data_dict['original'][user][rec]['label'].values
    win_mids = win_starts + (win_stops - win_starts) / 2
    x = win_mids
    y = y_pred_windows
    x_new = np.arange(0, len(y_true_raw))
    y_pred_raw = utils.resample(x, y.T, x_new, kind='nearest').T
    prediction_traces[rec] = {'window': {'true': y_true_windows, 'pred': y_pred_windows},
                              'raw': {'true': y_true_raw, 'pred': y_pred_raw}}


for (i, rec) in enumerate(recs):
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    ax[0].plot(swimming_data.data_dict['original'][user][rec]['ACC_0'].values)
    ax[0].plot(swimming_data.data_dict['original'][user][rec]['ACC_1'].values)
    ax[0].plot(swimming_data.data_dict['original'][user][rec]['ACC_2'].values)
    ax[1].plot(prediction_traces[rec]['raw']['true'])
    yp = np.argmax(prediction_traces[rec]['raw']['pred'], axis=1)
    ax[1].plot(yp)
plt.show()
