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

def detect_swim_styles_in_history(history, num_classes=5):
    """
    Count swim styles based on history metrics and validate against model output classes.
    """
    detected_styles = set()
    
    # Look for swim style metrics in history
    for key in history[0].keys():  # Assuming first epoch structure
        print(key) 
        if "swim_style_output" in key:
            detected_styles.add(key.split("_output")[0])
    
    # Use model output class count as a reference
    swim_style_count = min(len(detected_styles), num_classes)
    print(f"Detected swim styles: {len(detected_styles)} | Total possible classes: {num_classes}")
    
    return swim_style_count


def adjust_score_for_styles(history, score):
    """Adjust score based on the number of unique swim styles."""
    unique_styles = detect_swim_styles_in_history(history, num_classes=5)
    diversity_factor = 1 + (len(unique_styles) - 1) * 0.1  # Boost score for diversity
    return score * diversity_factor, len(unique_styles)

# Function to calculate a composite score
def calculate_composite_score_style(history):
    def get_best_value(values, is_loss=True):
        """Return the best value: min for losses, max for accuracies."""
        if isinstance(values[0], list):
            values = [item for sublist in values for item in sublist]
        return min(values) if is_loss else max(values)
    
    # Extract metrics for all epochs
    train_style_loss = [epoch.get('swim_style_output_loss', 0) for epoch in history]
    train_stroke_loss = [epoch.get('stroke_label_output_loss', 0) for epoch in history]
    val_style_loss = [epoch.get('val_swim_style_output_loss', 0) for epoch in history]
    val_stroke_loss = [epoch.get('val_stroke_label_output_loss', 0) for epoch in history]

    train_style_accuracy = [epoch.get('swim_style_output_accuracy', 0) for epoch in history]
    train_stroke_accuracy = [epoch.get('stroke_label_output_accuracy', 0) for epoch in history]
    val_style_accuracy = [epoch.get('val_swim_style_output_accuracy', 0) for epoch in history]
    val_stroke_accuracy = [epoch.get('val_stroke_label_output_accuracy', 0) for epoch in history]

    # Best values across all epochs
    best_train_style_loss = get_best_value(train_style_loss, is_loss=True)
    best_val_style_loss = get_best_value(val_style_loss, is_loss=True)
    best_train_style_accuracy = get_best_value(train_style_accuracy, is_loss=False)
    best_val_style_accuracy = get_best_value(val_style_accuracy, is_loss=False)

    # Normalize losses
    norm_val_style_loss = 1 / (best_val_style_loss + 1e-8)
    norm_train_style_loss = 1 / (best_train_style_loss + 1e-8)

    # Composite score calculation
    composite_score = (
        norm_val_style_loss + best_val_style_accuracy +
        norm_train_style_loss + best_train_style_accuracy
    ) / 4

    # Adjust for swim style diversity
    adjusted_score, style_count = adjust_score_for_styles(history, composite_score)

    return adjusted_score, style_count


# Iterate through swimmer directories
def analyze_swimmers(data_path):
    results = []
    for swimmer in os.listdir(data_path):
        swimmer_dir = os.path.join(data_path, swimmer)
        history_file = os.path.join(swimmer_dir, f"history.pkl")
        
        if os.path.exists(history_file):
            print(f"Analyzing Swimmer: {swimmer}")
            
            # Load history
            history = load_history(history_file)

            best_score, style_count = calculate_composite_score_style(history)

            # Append results
            results.append({
                'Swimmer': swimmer,
                'Best_Composite_Score': best_score,
                'Swim_Style_Count': style_count
            })
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    # Update these paths with actual locations
    swimmer_results = analyze_swimmers(DATA_PATH)
    print(swimmer_results)
    swimmer_results.to_csv("model_evaluation_results.csv", index=False)
