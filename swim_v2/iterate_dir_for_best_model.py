import os
import pickle

def load_pickle_files(root_directory):
    history_data = {}
    for subdir, dirs, files in os.walk(root_directory):
        for file in files:
            if file.endswith('history.pkl'):  # Check for .pkl files
                file_path = os.path.join(subdir, file)
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                history_data[file_path] = data
    return history_data

def calculate_score(history):
    # Define how to handle the metrics
    def get_best_value(values, is_loss=True):
        """Return the best value: min for losses, max for accuracies."""
        return min(values) if is_loss else max(values)

    def get_last_value(values):
        """Return the last value in the list."""
        return values[-1]

    def get_average_value(values):
        """Return the average of the values."""
        return sum(values) / len(values)

    # Apply the chosen strategy
    # Here we use get_best_value for demonstration; you can switch to get_last_value or get_average_value
    last_epoch = history[-1]
    train_style_loss = get_best_value(last_epoch['swim_style_output_loss'], is_loss=True)
    train_stroke_loss = get_best_value(last_epoch['stroke_label_output_loss'], is_loss=True)
    train_style_accuracy = get_best_value(last_epoch['swim_style_output_accuracy'], is_loss=False)
    train_stroke_accuracy = get_best_value(last_epoch['stroke_label_output_accuracy'], is_loss=False)
    train_style_weighted_accuracy = get_best_value(last_epoch['swim_style_output_weighted_accuracy'], is_loss=False)
    train_stroke_weighted_accuracy = get_best_value(last_epoch['stroke_label_output_weighted_accuracy'], is_loss=False)

    val_style_loss = get_best_value(last_epoch['val_swim_style_output_loss'], is_loss=True)
    val_stroke_loss = get_best_value(last_epoch['val_stroke_label_output_loss'], is_loss=True)
    val_style_accuracy = get_best_value(last_epoch['val_swim_style_output_accuracy'], is_loss=False)
    val_stroke_accuracy = get_best_value(last_epoch['val_stroke_label_output_accuracy'], is_loss=False)
    val_style_weighted_accuracy = get_best_value(last_epoch['val_swim_style_output_weighted_accuracy'], is_loss=False)
    val_stroke_weighted_accuracy = get_best_value(last_epoch['val_stroke_label_output_weighted_accuracy'], is_loss=False)

    

    # Calculate the score assuming equal weighting
    score = (1 / val_style_loss + val_stroke_loss + val_style_accuracy + + val_stroke_accuracy + val_style_weighted_accuracy + val_stroke_weighted_accuracy + 1 / train_style_loss + train_stroke_loss + train_style_accuracy + train_stroke_accuracy + train_style_weighted_accuracy + train_stroke_weighted_accuracy) / 12
    return score

def calculate_best_score(history):
    # Find the epoch with the minimum validation loss or maximum validation accuracy
    best_epoch = min(history, key=lambda x: x['val_loss'])  # or max(history, key=lambda x: x['val_acc'])

    norm_val_loss = 1 / min(best_epoch['val_loss'])
    norm_train_loss = 1 / min(best_epoch['loss'])
    
    best_epoch = max(history, key=lambda x: x['val_acc'])  # or max(history, key=lambda x: x['val_acc'])
    norm_val_acc = max(best_epoch['val_acc'])
    norm_val_weighted_acc = max(best_epoch['val_weighted_acc'])
    norm_acc = max(best_epoch['acc'])
    norm_weighted_acc = max(best_epoch['weighted_acc'])
    
    # Compute the score
    score = (norm_val_loss + norm_train_loss + norm_val_acc + norm_val_weighted_acc + norm_acc + norm_weighted_acc) / 6
    return score


def find_best_model(history_data):
    best_score = float('-inf')
    best_model_path = None

    for path, history in history_data.items():
        score = calculate_best_score(history)
        print(score)
        if score > best_score:
            best_score = score
            best_model_path = path

    return best_model_path, best_score

def main():
    root_directory = '/Users/juanloya/Documents/SwimmingModelPython/swim_v2/tutorial_save_path_epoch100_mod'  # Path to the directory containing .pkl files
    all_histories = load_pickle_files(root_directory)
    best_model_path, best_score = find_best_model(all_histories)
    print(f"Best model is at {best_model_path} with a score of {best_score}")

if __name__ == "__main__":
    main()
