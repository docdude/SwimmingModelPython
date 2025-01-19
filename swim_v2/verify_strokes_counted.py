import pandas as pd
import matplotlib.pyplot as plt

# Load the test file
file_path = 'data_modified/20/Freestyle_1526898327587_updated.csv'
df = pd.read_csv(file_path)

# Extract columns
timestamp = df['timestamp']
acc_magnitude = (df[['ACC_0', 'ACC_1', 'ACC_2']] ** 2).sum(axis=1) ** 0.5  # Compute acceleration magnitude
stroke_count = df['stroke_labels']

# Count total strokes
total_strokes = stroke_count.sum()
print(f"Total strokes detected in the dataset: {total_strokes}")

# Get indices where strokes are detected
stroke_indices = df.index[df['stroke_labels'] == 1].tolist()
print(f"Number of rows with detected strokes: {len(stroke_indices)}")

# Plot the acceleration magnitude with stroke markers
plt.figure(figsize=(15, 5))
plt.plot(timestamp, acc_magnitude, label='Acceleration Magnitude', alpha=0.8)
plt.scatter(
    timestamp[stroke_indices],
    acc_magnitude[stroke_indices],
    color='red',
    label='Stroke Detected (1)',
    s=10
)
plt.xlabel('Timestamp')
plt.ylabel('Acceleration Magnitude')
plt.title(f"Acceleration Magnitude with Stroke Count\nTotal Strokes: {total_strokes}")
plt.legend()
plt.grid()
plt.show()

# Debugging the `stroke_count` column
print("Stroke Count Column Value Counts:")
print(df['stroke_labels'].value_counts())

# Save stroke rows to inspect
stroke_rows = df[df['stroke_labels'] == 1]
stroke_rows.to_csv('strokes_debug.csv', index=False)
print("Saved rows with detected strokes to 'strokes_debug.csv'.")
