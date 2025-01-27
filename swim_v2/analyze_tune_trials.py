import os
import tensorflow as tf
from tensorflow.core.util.event_pb2 import Event
from google.protobuf.json_format import MessageToDict
from google.protobuf.json_format import MessageToJson

from tensorboard.plugins.hparams import plugin_data_pb2
import numpy as np
import pickle
from tqdm import tqdm
import shutil

def parse_tensorboard_logs_for_validation(log_base_dir, metric_name):
    """
    Parse TensorBoard logs to extract validation metrics and hyperparameters.
    """
    metrics_by_trial = {}
    hyperparameters_by_trial = {}

    # Sort timestamp directories by creation time (oldest first)
    sorted_timestamp_dirs = sorted(
        [d for d in os.listdir(log_base_dir) if os.path.isdir(os.path.join(log_base_dir, d))],
        key=lambda x: os.path.getctime(os.path.join(log_base_dir, x))
    )

    with tqdm(total=len(sorted_timestamp_dirs), desc="Processing timestamp directories", position=0, colour="green", leave=True) as main_bar:

        for timestamp_dir in sorted_timestamp_dirs:
            timestamp_path = os.path.join(log_base_dir, timestamp_dir)

            # Sort numerical subdirectories within the timestamp directory by creation time
            sorted_subdirs = sorted(
                [d for d in os.listdir(timestamp_path) if os.path.isdir(os.path.join(timestamp_path, d))],
                key=lambda x: os.path.getctime(os.path.join(timestamp_path, x))
            )
            with tqdm(total=len(sorted_subdirs), desc=f"  Processing subdirectories in {timestamp_dir}", position=1, leave=False) as sub_bar:

                for sub_dir in sorted_subdirs:
                    sub_dir_path = os.path.join(timestamp_path, sub_dir)

                    if not os.path.isdir(sub_dir_path):  # Skip non-directories
                        continue

                    execution_dir = os.path.join(sub_dir_path, "execution0")
                    if os.path.isdir(execution_dir):
                        val_dir = os.path.join(execution_dir, "validation")
                        if os.path.isdir(val_dir):
                            metrics_by_trial.update(
                                _extract_metrics_from_tfevents(val_dir, metric_name, trial_key=f"{timestamp_dir}/{sub_dir}")
                            )

                        train_dir = os.path.join(execution_dir, "train")
                        if os.path.isdir(train_dir):
                            hyperparameters_by_trial.update(
                                _extract_hyperparameters_from_tfevents(execution_dir, trial_key=f"{timestamp_dir}/{sub_dir}")
                            )

                    sub_bar.update(1)
            main_bar.update(1)

    return metrics_by_trial, hyperparameters_by_trial


def _extract_metrics_from_tfevents(directory, metric_name, trial_key):
    """
    Extract metrics from TensorBoard event files in a directory.
    """
    metrics = {}
    tfevents_files = [f for f in os.listdir(directory) if f.startswith("events.out.tfevents")]
    for tfevents_file in tfevents_files:
        tfevents_path = os.path.join(directory, tfevents_file)
        try:
            dataset = tf.data.TFRecordDataset(tfevents_path)
            for record in dataset:
                event = Event()
                event.ParseFromString(record.numpy())
                if event.summary.value:
                    for value in event.summary.value:
                        if value.tag == metric_name:
                            if trial_key not in metrics:
                                metrics[trial_key] = []
                            # Try extracting from tensor if simple_value is 0
                            if value.simple_value != 0:
                                metric_value = value.simple_value
                            elif value.HasField("tensor"):
                                # Extract tensor values
                                tensor_proto = value.tensor
                                tensor_value = tf.make_ndarray(tensor_proto)  # Convert tensor to numpy array
                                metric_value = tensor_value.item()  # Get the scalar value from the tensor
                            else:
                                metric_value = None

                            metrics[trial_key].append((event.step, metric_value))
        except Exception as e:
            print(f"Error reading {tfevents_path}: {e}")
    return metrics


def _extract_hyperparameters_from_tfevents(directory, trial_key):
    """
    Extract hyperparameters from TensorBoard event files in a directory.
    """
    hyperparameters = {}
    tfevents_files = [f for f in os.listdir(directory) if f.startswith("events.out.tfevents")]
    for tfevents_file in tfevents_files:
        tfevents_path = os.path.join(directory, tfevents_file)
        try:
            dataset = tf.data.TFRecordDataset(tfevents_path)
            for record in dataset:
                event = Event()
                event.ParseFromString(record.numpy())
               # print(record.numpy())
                if event.summary.value:
                    for value in event.summary.value:
                        if "_hparams_/session_start_info" in value.tag:
                            # Extract hyperparameters from metadata plugin_data
                            plugin_data = value.metadata.plugin_data.content
                            hparams = parse_plugin_data(plugin_data)
                            hyperparameters[trial_key] = hparams
        except Exception as e:
            print(f"Error reading {tfevents_path}: {e}")
    return hyperparameters

def parse_plugin_data(plugin_data_content):
    """
    Parse the plugin data content from hparams metadata.
    
    :param plugin_data_content: Serialized protobuf content in binary format.
    :return: Dictionary of hyperparameters.
    """
    try:
        # Parse the plugin data content as HParamsConfig protobuf
        hparams_plugin_data = plugin_data_pb2.HParamsPluginData()
        hparams_plugin_data.ParseFromString(plugin_data_content)
        print("Parsed Experiment Message (JSON):", MessageToJson(hparams_plugin_data))

        # Convert the protobuf message to a dictionary
        parsed_dict = MessageToDict(hparams_plugin_data, preserving_proto_field_name=True)
        hparams_dict = extract_hparams_and_group(parsed_dict)        
        return hparams_dict
    except Exception as e:
        print(f"Error parsing plugin data: {e}")
        return None

def extract_hparams_and_group(parsed_dict):
    try:
        return {
            'group_name': parsed_dict.get('session_start_info', {}).get('group_name', ''),
            'hparams': parsed_dict.get('session_start_info', {}).get('hparams', {})

        }
    except Exception as e:
        print(f"Error extracting hparams: {e}")
        return None

def analyze_metrics(metrics_by_trial, metric_name="epoch_weighted_f1_score", last_n_epochs=5):
    """
    Analyze extracted metrics for each trial.
    """
    results = {}
    for trial, metrics in metrics_by_trial.items():
        if len(metrics) >= last_n_epochs:
            # Sort metrics by step and compute the mean of the last N epochs
            metrics = sorted(metrics, key=lambda x: x[0])  # Sort by step
            last_n_metrics = [value for step, value in metrics[-last_n_epochs:]]
            mean_f1 = np.mean(last_n_metrics)
            std_f1 = np.std(last_n_metrics)
            peak_f1 = max(last_n_metrics)
            results[trial] = {"mean_f1": mean_f1, "std_f1": std_f1, "peak_f1": peak_f1}

    return results


def select_top_trials(results, hyperparameters_by_trial, top_n=3):
    """
    Select the top N trials based on a custom score and include hyperparameters.
    """
    scored_trials = [
        (trial, stats["mean_f1"] - stats["std_f1"], stats, hyperparameters_by_trial.get(trial, {}))
        for trial, stats in results.items()
    ]
    scored_trials.sort(key=lambda x: x[1], reverse=True)  # Sort by custom score
    return scored_trials[:top_n]


def save_top_trials(top_trials, save_path):
    """
    Save top trials and their hyperparameters as a pickle file.
    """
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)  # Creates all parent directories if they don't exist

    with open(os.path.join(save_path, 'top_trials.pkl'), 'wb') as f:
        pickle.dump(top_trials, f)
    print(f"Top trials saved to {save_path}")


def main(log_dir, metric_name="epoch_weighted_f1_score", top_n=5, last_n_epochs=15, save_path="top_trials.pkl"):
    """
    Main function to extract and analyze TensorBoard logs.
    """
    print("Extracting metrics and hyperparameters from TensorBoard logs...")
    metrics_by_trial, hyperparameters_by_trial = parse_tensorboard_logs_for_validation(log_dir, metric_name)

    print("Analyzing metrics...")
    results = analyze_metrics(metrics_by_trial, metric_name=metric_name, last_n_epochs=last_n_epochs)

    print(f"Selecting the top {top_n} trials...")
    top_trials = select_top_trials(results, hyperparameters_by_trial, top_n=top_n)

    # Print the top trials
    print("\nTop Trials:")
    for rank, (trial, score, stats, hyperparams) in enumerate(top_trials, start=1):
        print(f"Rank {rank}:")
        print(f"  Trial: {trial}")
        print(f"  Mean {metric_name}: {stats['mean_f1']:.4f}")
        print(f"  Std {metric_name}: {stats['std_f1']:.4f}")
        print(f"  Peak {metric_name}: {stats['peak_f1']:.4f}")
        print(f"  Custom Score: {score:.4f}")
        print(f"  Hyperparameters: {hyperparams}")

    print(f"Saving the top {top_n} trials...")
    save_top_trials(top_trials, save_path)


if __name__ == "__main__":
    run_name = "tune_stroke_lstm"
    save_path = f'/Users/juanloya/Documents/SwimmingModelPython/swim_v2/best_hyperparameters/{run_name}'
    log_base_dir = f'/Users/juanloya/Documents/SwimmingModelPython/swim_v2/logs/{run_name}'
    main(log_base_dir, metric_name="epoch_weighted_f1_score", top_n=5, last_n_epochs=15, save_path=save_path)
