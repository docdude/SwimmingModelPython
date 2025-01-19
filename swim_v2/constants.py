SAMPLING_FREQ = 100
ALL_LABELS = [-1, 0, 1, 2, 3, 4, 5, 6]
LABEL_NAMES = {-1: 'Unknown', 0: 'Null', 1: 'Freestyle', 2: 'Breaststroke', 3: 'Backstroke', 4: 'Butterfly', 5: 'Turn',
               6: 'Kick'}
STROKE_LABELS = {0: 'No Stroke', 1: 'Stroke'}
RAW_COL_NAMES = ['timestamp', 'sensor', 'value_0', 'value_1', 'value_2']
LABELED_COL_NAMES = ['timestamp', 'sensor', 'value_0', 'value_1', 'value_2', 'label']
SENSORS = ['ACC', 'GYRO', 'MAG', 'PRESS', 'LIGHT']
SENSOR_MAX = {'ACC': 96, 'GYRO': 33}
AXIS_MIRROR = ['ACC_0', 'GYRO_1', 'GYRO_2', 'MAG_0']


def main():
    print("Running main")


if __name__ == '__main__':
    main()
