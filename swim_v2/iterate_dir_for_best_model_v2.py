import os
import pickle
import pandas as pd
import numpy as np

# Paths to directories
DATA_PATH = "/Users/juanloya/Documents/SwimmingModelPython/swim_v2/tutorial_save_path_epoch100_mod"  # Path to the directory containing .pkl files

# Function to load training history
def load_history(history_file):
    with open(history_file, 'rb') as file:
        return pickle.load(file)

def detect_swim_styles_in_history(history):
    """Detect unique swim styles from history metric keys."""
    styles = set()
    for key in history[0].keys():  # Check keys from the first epoch
        if "swim_style_output" in key:
            styles.add(key.split("_output")[0])  # Extract style name
    return styles

def adjust_score_for_styles(history, score):
    """Adjust score based on the number of unique swim styles."""
    unique_styles = detect_swim_styles_in_history(history)
    diversity_factor = 1 + (len(unique_styles) - 1) * 0.1  # Boost score for diversity
    return score * diversity_factor, len(unique_styles)

# Function to calculate a composite score
def calculate_composite_score(history):
    def get_best_value(values, is_loss=True):
        """Return the best value: min for losses, max for accuracies."""
        # Flatten nested lists if necessary
        if isinstance(values[0], list):  
            values = [item for sublist in values for item in sublist]  # Flatten nested lists
        return min(values) if is_loss else max(values)
    
    # Safely extract metrics for all epochs
    try:
        train_style_loss = [epoch.get('swim_style_output_loss', [0])[0] for epoch in history]
        train_stroke_loss = [epoch.get('stroke_label_output_loss', [0])[0] for epoch in history]
        train_style_accuracy = [epoch.get('swim_style_output_accuracy', [0])[0] for epoch in history]
        train_stroke_accuracy = [epoch.get('stroke_label_output_accuracy', [0])[0] for epoch in history]
        train_style_weighted_accuracy = [epoch.get('swim_style_output_weighted_accuracy', [0])[0] for epoch in history]
        train_stroke_weighted_accuracy = [epoch.get('stroke_label_output_weighted_accuracy', [0])[0] for epoch in history]

        val_style_loss = [epoch.get('val_swim_style_output_loss', [0])[0] for epoch in history]
        val_stroke_loss = [epoch.get('val_stroke_label_output_loss', [0])[0] for epoch in history]
        val_style_accuracy = [epoch.get('val_swim_style_output_accuracy', [0])[0] for epoch in history]
        val_stroke_accuracy = [epoch.get('val_stroke_label_output_accuracy', [0])[0] for epoch in history]
        val_style_weighted_accuracy = [epoch.get('val_swim_style_output_weighted_accuracy', [0])[0] for epoch in history]
        val_stroke_weighted_accuracy = [epoch.get('val_stroke_label_output_weighted_accuracy', [0])[0] for epoch in history]

    except (AttributeError, IndexError, TypeError):
        raise ValueError("History structure does not contain expected epoch metrics.")

    # Get best values across all epochs
    best_train_style_loss = get_best_value(train_style_loss, is_loss=True)
    best_train_stroke_loss = get_best_value(train_stroke_loss, is_loss=True)
    best_train_style_accuracy = get_best_value(train_style_accuracy, is_loss=False)
    best_train_stroke_accuracy = get_best_value(train_stroke_accuracy, is_loss=False)
    best_train_style_weighted_accuracy = get_best_value(train_style_weighted_accuracy, is_loss=False)
    best_train_stroke_weighted_accuracy = get_best_value(train_stroke_weighted_accuracy, is_loss=False)

    best_val_style_loss = get_best_value(val_style_loss, is_loss=True)
    best_val_stroke_loss = get_best_value(val_stroke_loss, is_loss=True)
    best_val_style_accuracy = get_best_value(val_style_accuracy, is_loss=False)
    best_val_stroke_accuracy = get_best_value(val_stroke_accuracy, is_loss=False)
    best_val_style_weighted_accuracy = get_best_value(val_style_weighted_accuracy, is_loss=False)
    best_val_stroke_weighted_accuracy = get_best_value(val_stroke_weighted_accuracy, is_loss=False)

    # Normalize and combine metrics into a single score
    norm_val_style_loss = 1 / (best_val_style_loss + 1e-8)
    norm_val_stroke_loss = 1 / (best_val_stroke_loss + 1e-8)
    norm_train_style_loss = 1 / (best_train_style_loss + 1e-8)
    norm_train_stroke_loss = 1 / (best_train_stroke_loss + 1e-8)

    # Calculate composite score
    composite_score = (
        norm_val_style_loss + norm_val_stroke_loss + best_val_style_accuracy + best_val_stroke_accuracy +
        best_val_style_weighted_accuracy + best_val_stroke_weighted_accuracy +
        norm_train_style_loss + norm_train_stroke_loss + best_train_style_accuracy + best_train_stroke_accuracy +
        best_train_style_weighted_accuracy + best_train_stroke_weighted_accuracy
    ) / 12

    return composite_score

def calculate_composite_score_new(history):
    def get_best_value(values, is_loss=True):
        """Return the best value: min for losses, max for accuracies."""
        return min(values) if is_loss else max(values)

    # Access the first dictionary inside the list
    history = history[0]  # Fix here: access the first element of the list

    # Extract metrics from history
    train_loss = history['loss']
    train_style_loss = history['swim_style_output_loss']
    train_stroke_loss = history['stroke_label_output_loss']
    train_style_accuracy = history['swim_style_output_accuracy']
    train_stroke_accuracy = history['stroke_label_output_accuracy']

    val_loss = history['val_loss']
    val_style_loss = history['val_swim_style_output_loss']
    val_stroke_loss = history['val_stroke_label_output_loss']
    val_style_accuracy = history['val_swim_style_output_accuracy']
    val_stroke_accuracy = history['val_stroke_label_output_accuracy']

    # Compute best values across epochs
    best_train_loss = get_best_value(train_loss, is_loss=True)
    best_val_loss = get_best_value(val_loss, is_loss=True)
    best_train_style_accuracy = get_best_value(train_style_accuracy, is_loss=False)
    best_val_style_accuracy = get_best_value(val_style_accuracy, is_loss=False)
    best_train_stroke_accuracy = get_best_value(train_stroke_accuracy, is_loss=False)
    best_val_stroke_accuracy = get_best_value(val_stroke_accuracy, is_loss=False)

    # Normalize losses
    norm_val_loss = 1 / (best_val_loss + 1e-8)
    norm_train_loss = 1 / (best_train_loss + 1e-8)

    # Calculate composite score
    composite_score = (
        norm_val_loss + norm_train_loss +
        best_train_style_accuracy + best_val_style_accuracy +
        best_train_stroke_accuracy + best_val_stroke_accuracy
    ) / 6

    return composite_score


# Iterate through swimmer directories
def analyze_swimmers(data_path):
    results = []
    swimmers = os.listdir(data_path)
    sorted_swimmers = sorted(swimmers, key=int)
    for swimmer in sorted_swimmers:#os.listdir(data_path):
        swimmer_dir = os.path.join(data_path, swimmer)
        history_file = os.path.join(swimmer_dir, f"history.pkl")
        
        if os.path.exists(history_file):
            print(f"Analyzing Swimmer: {swimmer}")
            
            # Load history
            history = load_history(history_file)
            best_score = calculate_composite_score_new(history)
            
            # Append results
            results.append({
                'Swimmer': swimmer,
                'Best_Composite_Score': best_score
            })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Mark the highest score with an asterisk
    max_score = results_df['Best_Composite_Score'].max()
    results_df['Best_Composite_Score'] = results_df['Best_Composite_Score'].apply(
        lambda x: f"*{x}" if x == max_score else f"{x}"
    )
    
    return results_df

if __name__ == "__main__":
    # Update these paths with actual locations
    swimmer_results = analyze_swimmers(DATA_PATH)
    print(swimmer_results)
    swimmer_results.to_csv("model_evaluation_results.csv", index=False)
