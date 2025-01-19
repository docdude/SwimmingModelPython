import numpy as np
import pickle
import utils
import os
import matplotlib.pyplot as plt

labels = [0, 1, 2, 3, 4]
label_names = ['Null', 'Front Crawl', 'Breaststroke', 'Backstroke', 'Butterfly']
label_names_abb = ['Null', 'Fr', 'Br', 'Ba', 'Bu']

loso_path = '/Users/juanloya/Documents/SwimmingModelPython/swim_v2/tutorial_save_path_epochBiLSTM1_weighted/' #Path where trained LOSO models are stored
save_path = 'SAVE_PATH'

with open(os.path.join(loso_path, 'prediction_traces_best.pkl'), 'rb') as f:
    prediction_traces = pickle.load(f)[0]
users = list(prediction_traces.keys())
c_thresh = 4 * 30
#print(prediction_traces)
cm_ix = {1: 0, 2: 1, 3: 2, 4: 3, 'mixed': 4}
cm = np.zeros((5, 5))
lane_predictions = {user: None for user in users}
for user in users:
    print("Working on %s" % user)
    for rec in prediction_traces[user].keys():
        #y_true = prediction_traces[user][rec]['raw']['swim_style']['true']
        #y_pred_cat = prediction_traces[user][rec]['raw']['swim_style']['pred']
        y_true = prediction_traces[user][rec]['raw']['true']
        y_pred_cat = prediction_traces[user][rec]['raw']['pred']
        y_pred = utils.from_categorical(y_pred_cat)

        y_true_ns = np.ones(len(y_true))
        y_true_ns[y_true == 0] = 0
        lane_start, lane_stop = utils.start_stop(y_true_ns)
        for i in range(len(lane_start)):
            y_lane_true = y_true[lane_start[i]:lane_stop[i]]
            y_lane_pred = y_pred[lane_start[i]:lane_stop[i]]
            y_lane_true_list, c_true = np.unique(y_lane_true, return_counts=True)
            y_lane_pred_list, c_pred = np.unique(y_lane_pred, return_counts=True)

            if (lane_stop[i] - lane_start[i]) < 22 * 30:
                print("Too short lane")
                continue
            if -1 in y_lane_true_list or 6 in y_lane_true_list:
                print("Skipping lane %d in %s, %s because it contains unknowns or kicks" % (i, user, rec))
                continue

            if 0 in y_lane_pred_list:
                y_lane_pred_list = y_lane_pred_list[1:]
                c_pred = c_pred[1:]
                if len(c_true) == 0:
                    print("Only predicted null for %d in %s, %s" % (i, user, rec))
                    raise ValueError("Possibly need to implement")

            at = np.where(c_true > c_thresh)[0]
            ap = np.where(c_pred > c_thresh)[0]
            y_lane_true_list = [y_lane_true_list[v] for v in at]
            y_lane_pred_list = [y_lane_pred_list[v] for v in ap]
            if len(y_lane_true_list) > 1:
                y_true_label = 'mixed'
            else:
                y_true_label = y_lane_true_list[0]

            if len(y_lane_pred_list) > 1:
                y_pred_label = 'mixed'
            else:
                y_pred_label = y_lane_pred_list[0]

            if (y_true_label != 'mixed') & (y_pred_label != 'mixed'):
                if y_pred_label != y_true_label:
                    print("%s, %s" % (user, rec))
            #print("%s, %s; %s, %s" % (y_true_label, y_pred_label, c_true, c_pred))
            cm[cm_ix[y_true_label], cm_ix[y_pred_label]] = cm[cm_ix[y_true_label], cm_ix[y_pred_label]] + 1

cm_norm = utils.normalize_confusion_matrix(cm)
print(cm)
print(utils.write_confusion_matrix(cm_norm, labels=[1, 2, 3, 4, 0]))
print(utils.write_confusion_matrix(cm, labels=[1, 2, 3, 4, 0]))

print(utils.write_latex_confmat(cm, labels=['Front Crawl', 'Breaststroke', 'Backstroke', 'Butterfly', 'Mixed'], is_integer=True))

print(np.mean(np.diag(cm_norm)))