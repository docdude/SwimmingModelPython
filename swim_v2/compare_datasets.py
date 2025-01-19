import pandas as pd
import numpy as np

def compare_selected_columns(original_df, reconstructed_df, columns_to_compare):
    """
    Compare selected columns of original and reconstructed datasets row by row.

    Parameters:
    -----------
    original_df : pd.DataFrame
        The original dataset.
    reconstructed_df : pd.DataFrame
        The reconstructed dataset.
    columns_to_compare : list
        List of column names to compare.

    Returns:
    --------
    diff_report : pd.DataFrame
        A report showing discrepancies between the two datasets for the selected columns.
    """
    # Filter datasets to only include the specified columns
    original_df_filtered = original_df[columns_to_compare]
    reconstructed_df_filtered = reconstructed_df[columns_to_compare]

    # Ensure both filtered datasets have the same shape
#    if original_df_filtered.shape != reconstructed_df_filtered.shape:
#        raise ValueError("Filtered datasets have different shapes. Check reconstruction.")

    # Compare row-by-row for the selected columns
    differences = []
    for i, (orig_row, recon_row) in enumerate(zip(original_df_filtered.values, reconstructed_df_filtered.values)):
        if not np.array_equal(orig_row, recon_row):
            diff = {
                "Row": i,
                "Original": orig_row,
                "Reconstructed": recon_row,
                "Difference": orig_row - recon_row if orig_row.shape == recon_row.shape else "Shape Mismatch"
            }
            differences.append(diff)

    # Convert differences to a DataFrame
    diff_report = pd.DataFrame(differences)
    return diff_report


# Example usage
# Define the columns to compare
columns_to_compare = [
    "timestamp", "ACC_0", "ACC_1", "ACC_2", "GYRO_0", "GYRO_1", "GYRO_2", "label"
]

# Assuming original and reconstructed datasets are loaded as Pandas DataFrames
original_path = "/Users/juanloya/Documents/SwimmingModelPython/swim_v2/data/processed_30hz_relabeled/0/Breaststroke_1527071936580.csv"
reconstructed_path = "/Users/juanloya/Documents/SwimmingModelPython/swim_v2/data_modified/0/Breaststroke_1527071936580_updated.csv"

original_df = pd.read_csv(original_path)
reconstructed_df = pd.read_csv(reconstructed_path)

# Perform the comparison for the selected columns
diff_report = compare_selected_columns(original_df, reconstructed_df, columns_to_compare)

# Save or print the report
if not diff_report.empty:
    diff_report.to_csv("difference_report.csv", index=False)
    print("Differences found. Report saved to 'difference_report.csv'")
else:
    print("No differences found. Reconstruction is accurate!")

