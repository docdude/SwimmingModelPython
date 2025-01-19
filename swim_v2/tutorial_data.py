# Basic tutorial in how to load data
import pandas as pd
import utils
import matplotlib.pyplot as plt


p_res = 'data/processed_30hz_relabeled/0/Breaststroke_1527071936580.csv'


# Use this to load a re-sampled recording
df_res = pd.read_csv(p_res)


fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True)
ax.plot(df_res['timestamp'].values, df_res['ACC_0'].values)
ax.plot(df_res['timestamp'].values, df_res['ACC_1'].values)
ax.plot(df_res['timestamp'].values, df_res['ACC_2'].values)
ax.set_title("Resampled recording")
plt.show()







