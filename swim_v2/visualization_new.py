import matplotlib.pyplot as plt
import pandas as pd
import utils
import constants
import os
import numpy as np


def plot_processed_recording(file_path, sensors=None, show=True):
    """
    Plot a processed swimming recording. Meaning data has at least been resampled.
    :param file_path: A path to a saved processed recording or the Pandas DataFrame itself
    :param sensors: A list of sensors that are plotted
    :param show: A boolean value that if true shows the plot
    :return: A plot showing sensor time-series values
    """
    if type(file_path) is str:
        df = pd.read_csv(file_path)
    else:
        df = file_path
    if sensors is None:
        sensors = constants.SENSORS
    fig, ax = plt.subplots(nrows=len(sensors)+1, ncols=1, sharex=True)
    if len(sensors) == 1:
        ax = [ax]
    plot_colors = {'0': 'b', '1': 'r', '2': 'g', '012': 'k'}
    for (i, s) in enumerate(sensors + ['label']):
        ax[i].set_ylabel(s)
        if s in ['PRESS', 'LIGHT', 'label']:
            ax[i].plot(df['timestamp'].values, df[s].values, plot_colors['0'])
        else:
            for ii in [0, 1, 2]:
                ax[i].plot(df['timestamp'].values, df[s+'_'+str(ii)].values, plot_colors[str(ii)])
        # sensor_columns = [c for c in df.columns if c.startswith(s)]
        # for (ii, sc) in enumerate(sensor_columns):
        #     if sc in ['PRESS', 'LIGHT', 'label']:
        #         ax[i].plot(df['timestamp'].values, df[sc].values, plot_colors['0'])
        #     else:
        #         sensor_axis = sc.split('_')[1]
        #         ax[i].plot(df['timestamp'].values, df[sc].values, plot_colors[sensor_axis])
    ax[0].set_title(file_path.split('\\')[-1])
    if show:
        plt.show()
    return fig, ax


def plot_all_raw_recordings(file_paths, sensors=None, show=True):
    """
    Plot all swimming recordings in folder path
    :param file_paths: A list of paths to swimming recordings or a path to a folder containing swimming recordings
    :param sensors: The sensors which are plotted
    :param show: A boolean value that if true shows the plot
    :return: A plot showing sensor time-series values
    """
    if sensors is None:
        sensors = constants.SENSORS
    if type(file_paths) is str:
        file_paths = [os.path.join(file_paths, f) for f in os.listdir(file_paths) if f.endswith('.csv')]
    figs = [None] * len(file_paths)
    axes = [None] * len(file_paths)
    for (i, fp) in enumerate(file_paths):
        figs[i], axes[i] = plot_raw_recording(fp, sensors=sensors, show=False)
        plt.close()
    fig, ax = plt.subplots(nrows=len(sensors), ncols=1, sharex=True)
    if len(sensors) == 1:
        ax = [ax]
    plot_colors = ['b', 'r', 'g']
    x_ticks = [None] * len(file_paths)
    x_tick_labels = [None] * len(file_paths)
    for i in range(len(file_paths)):
        for ii in range(len(axes[i])):
            for iii in range(len(axes[i][ii].lines)):
                xy_data = axes[i][ii].lines[iii].get_xydata()
                ax[ii].plot(xy_data[:, 0], xy_data[:, 1], plot_colors[iii])
        x_ticks[i] = xy_data[0, 0]
        x_tick_labels[i] = axes[i][0].get_title()
    ax[0].set_xticks(x_ticks)
    ax[0].set_xticklabels(x_tick_labels)
    for i in range(len(ax)):
        ax[i].set_ylabel(axes[0][i].get_ylabel())
    if show:
        plt.show(fig)


def plot_raw_recording(file_path, sensors=None, show=True, reverse_axes=False):
    """
    Plot a swimming recording
    :param file_path: Full path to a swimming recording
    :param sensors: The sensors which are plotted
    :param show: A boolean value that if true shows the plot
    :return: A plot showing sensor time-series values
    """
    if sensors is None:
        sensors = constants.SENSORS
    df, header, footer = utils.load_recording(file_path)
    fig, ax = plt.subplots(nrows=len(sensors), ncols=1, sharex=True)
    if len(sensors) == 1:
        ax = [ax]
    for (i, s) in enumerate(sensors):
        ax[i].set_ylabel(s)
        df_s = df[df['sensor'] == s]
        t = df_s['timestamp'].values * 10**-9
        ax[i].plot(t, df_s['value_0'], 'b')
        if s != 'PRESS' and s != 'LIGHT':
            ax[i].plot(t, -df_s['value_1'], 'r')
            ax[i].plot(t, -df_s['value_2'], 'g')
    ax[-1].set_xlabel("Time [s]")
    ax[0].set_title(raw_title(header, footer))
    if show:
        plt.show()
    return fig, ax


def raw_title(header, footer):
    if type(header) is not list:
        header = list(header)
    if type(footer) is not list:
        footer = list(footer)
    title_items = header + footer
    title = "\n".join('{}'.format(item) for item in title_items)
    return title


def main():
    print("Running visualization main")
    user = '0'
    p = os.path.join('../data/processed_30hz_relabeled', user)
    for rec in os.listdir(p):
        df = pd.read_csv(os.path.join(p, rec))
        t = np.arange(len(df['label'].values)) * 1/30
        fig, ax = plt.subplots(nrows=6, ncols=1, sharex=True)
        ax[0].plot(t, df['ACC_0'].values)
        ax[0].plot(t, df['ACC_1'].values)
        ax[0].plot(t, df['ACC_2'].values)
        ax[1].plot(t, df['GYRO_0'].values)
        ax[1].plot(t, df['GYRO_1'].values)
        ax[1].plot(t, df['GYRO_2'].values)
        ax[2].plot(t, df['MAG_0'].values)
        ax[2].plot(t, df['MAG_0'].values)
        ax[2].plot(t, df['MAG_0'].values)
        ax[3].plot(t, df['PRESS'].values)
        ax[4].plot(t, df['LIGHT'].values)
        ax[5].plot(t, df['label'].values)
        ax[5].set_yticks([-1, 0, 1, 2, 3, 4, 5, 6])
        ax[5].set_yticklabels(['Unknown', 'Null', 'Crawl', 'Breaststroke', 'Backstroke', 'Butterfly', 'Turn', 'Kick'])
        ax[-1].set_xlabel("Time [s]")
        ax[0].set_ylabel("Accelerometer")
        ax[1].set_ylabel("Gyroscope")
        ax[2].set_ylabel("Magnetometer")
        ax[3].set_ylabel("Pressure")
        ax[4].set_ylabel("Light")
        ax[5].set_ylabel("Label")
    plt.show()


if __name__ == '__main__':
    main()
