import numpy as np
import pickle
import os
import utils

# Define labels and paths
labels = [0, 1]  # 0: No Stroke, 1: Stroke
label_names = ['No Stroke', 'Stroke']
label_names_abb = ['NS', 'S']

loso_path = '/Users/juanloya/Documents/SwimmingModelPython/swim_v2/tutorial_save_path_epochtest_weighted/'  # Path where trained LOSO models are stored
save_path = '/Users/juanloya/Documents/SwimmingModelPython/swim_v2/tutorial_save_path_epochtest_weighted/'

# Load prediction traces
with open(os.path.join(loso_path, 'prediction_traces_best.pkl'), 'rb') as f:
    prediction_traces = pickle.load(f)[0]

users = list(prediction_traces.keys())
cm = np.zeros((2, 2))  # Confusion matrix for stroke labels (binary classification)

threshold = 0.1  # Adjust threshold for Stroke classification

for user in users:
    print(f"Working on {user}")
    for rec in prediction_traces[user].keys():
        # Get true and predicted stroke labels
        y_true = prediction_traces[user][rec]['raw']['strokes']['true']
        y_pred = prediction_traces[user][rec]['raw']['strokes']['pred']

        # Filter unexpected values
        valid_indices = (y_true >= 0) & (y_true <= 1)
        y_true = y_true[valid_indices]
        y_pred = y_pred[valid_indices]

        # Apply threshold
        y_pred_binary = (np.array(y_pred) >= threshold).astype(int)

        # Compute confusion matrix for this recording
        for t, p in zip(y_true, y_pred_binary):
            cm[int(t), int(p)] += 1

# Normalize confusion matrix
cm_norm = utils.normalize_confusion_matrix(cm)

# Print and save results
print("Confusion Matrix (Raw Counts):")
print(cm)
print(utils.write_confusion_matrix_stroke(cm, labels=[0,1]))
print("Confusion Matrix (Normalized):")
print(cm_norm)
print(utils.write_confusion_matrix_stroke(cm_norm, labels=[0,1]))
print("Confusion Matrix in LaTeX format:")
print(utils.write_latex_confmat(cm, labels=label_names, is_integer=True))

# Overall accuracy
accuracy = np.trace(cm) / np.sum(cm)
print(f"Overall Accuracy: {accuracy:.4f}")

# Save results to a file
results_path = os.path.join(save_path, 'stroke_label_results.pkl')
with open(results_path, 'wb') as f:
    pickle.dump({'confusion_matrix': cm, 'normalized_confusion_matrix': cm_norm, 'accuracy': accuracy}, f)

print(f"Results saved to {results_path}")
