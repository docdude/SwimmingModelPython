import os
import pandas as pd

# Define the root directory where user subdirectories are stored
root_directory = "data_modified_users"

# Initialize a dictionary to store summary statistics
summary = []

# Iterate through each user directory (0-39)
for user_id in range(40):
    user_directory = os.path.join(root_directory, str(user_id))

    # Check if the user directory exists
    if not os.path.exists(user_directory):
        print(f"User directory {user_id} does not exist.")
        continue

    # Iterate through all CSV files in the user directory
    for filename in os.listdir(user_directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(user_directory, filename)

            # Load the CSV file into a DataFrame
            try:
                data = pd.read_csv(file_path)
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                continue

            # Ensure required columns exist
            required_columns = ["row_index", "timestamp", "ACC_0", "ACC_1", "ACC_2", 
                                "GYRO_0", "GYRO_1", "GYRO_2", "label", "stroke_labels"]
            if not all(col in data.columns for col in required_columns):
                print(f"File {file_path} is missing required columns.")
                continue

            # Group by swim style (label) and count stroke and no-stroke occurrences
            grouped = data.groupby("label")["stroke_labels"].value_counts().unstack(fill_value=0)

            for style, counts in grouped.iterrows():
                strokes = counts.get(1, 0)
                no_strokes = counts.get(0, 0)
                total = strokes + no_strokes

                summary.append({
                    "User": user_id,
                    "Recording": filename,
                    "Style": style,
                    "Strokes": strokes,
                    "No_Strokes": no_strokes,
                    "Total": total,
                    "Stroke_Percentage": (strokes / total * 100) if total > 0 else 0.0
                })

# Convert the summary into a DataFrame
summary_df = pd.DataFrame(summary)

# Save the summary to a CSV file
output_file = "stroke_style_summary.csv"
summary_df.to_csv(output_file, index=False)

print(f"Summary saved to {output_file}")
