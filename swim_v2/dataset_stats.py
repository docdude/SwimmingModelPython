import pandas as pd

real_path = '/Users/juanloya/Documents/SwimmingModelPython/swim_v2/data/processed_30Hz_relabeled/0/freestyle_watch.csv'
# Load a real dataset example
real_data = pd.read_csv(real_path)

# Compute statistics for the real data
real_stats = real_data[["ACC_0", "ACC_1", "ACC_2", "GYRO_0", "GYRO_1", "GYRO_2"]].describe()#, "MAG_0", "MAG_1", "MAG_2"]].describe()
synthetic_path = '/Users/juanloya/Documents/SwimmingModelPython/swim_v2/data/processed_30Hz_relabeled/0/freestyle_watch_scaled.csv'

# Load a synthetic dataset example
synthetic_data = pd.read_csv(synthetic_path)

# Compute statistics for the synthetic data
synthetic_stats = synthetic_data[["ACC_0", "ACC_1", "ACC_2", "GYRO_0", "GYRO_1", "GYRO_2"]].describe()#, "MAG_0", "MAG_1", "MAG_2"]].describe()

# Compare the statistics
print("Unscaled Apple Data Statistics:\n", real_stats)
print("Scaled Apple Data Statistics:\n", synthetic_stats)
