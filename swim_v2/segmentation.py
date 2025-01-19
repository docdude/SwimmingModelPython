import numpy as np
import utils


def sliding_windows_start_stop_continuous(x, win_len, slide_len):
    """
    Start and stop indices for overlapping or non-overlapping windows.
    :param x: A numpy array of time values
    :param win_len: An integer for the window length in number of samples
    :param slide_len: An integer for the slide length in number of samples (0 for no overlap)
    :return: Two lists of window start and stop samples
    """
    if len(x) < win_len:
        return [], []

    if slide_len == 0:
        # Non-overlapping windows: ensure no overlap by setting slide_len = win_len 
        slide_len = win_len

    # Compute the window stops and starts
    win_stops_ix = np.arange(win_len - 1, len(x), slide_len)
    win_starts_ix = np.arange(0, win_stops_ix[-1] - win_len + 2, slide_len)
    win_starts = x[win_starts_ix]
    win_stops = x[win_stops_ix]
    
    return win_starts, win_stops

def label_segments(labels):
    """
    Start and stop indices of segments with uniform labels
    :param labels: A list or numpy array of labels
    :return: 3 lists containing segment start and stop indices and the corresponding labels
    """
    change_ix = np.where(np.abs(utils.diff(labels)) > 0)[0]
    seg_starts = np.append(0, change_ix+1)
    seg_stops = np.append(change_ix+1, len(labels))
    seg_labels = labels[seg_starts]
    return seg_starts, seg_stops, seg_labels


def sliding_windows_start_stop(x, win_len, slide_len, labels=None, labels_ignore=None, keep_pure=False):
    """
    Get window start and stop indices for overlapping windows
    :param x: A numpy array of time values
    :param win_len: An integer for the window length in number of samples
    :param slide_len: An integer for the slide length in number of samples
    :param labels: A list or numpy array of labels for each time value
    :param labels_ignore: A list of labels who are excluded
    :param keep_pure: If True only windows where there is no mix of labels
    :return: Window start and stop indices
    """
    if labels is None:
        win_starts, win_stops = sliding_windows_start_stop_continuous(x, win_len, slide_len)
    else:
        if len(labels) != len(x):
            raise ValueError("labels and x should be of equal length")
        seg_starts, seg_stops, seg_labels = label_segments(labels)
        if labels_ignore is not None:
            ix = [i for i in range(len(seg_labels)) if seg_labels[i] in labels_ignore]
            seg_starts = np.delete(seg_starts, ix)
            seg_stops = np.delete(seg_stops, ix)
        if not keep_pure:
            ix_mid = seg_starts[1:]-seg_stops[:-1] > 1
            ix_starts = np.append(True, ix_mid)
            ix_stops = np.append(ix_mid, True)
            seg_starts = seg_starts[ix_starts]
            seg_stops = seg_stops[ix_stops]
        win_starts = np.array([])
        win_stops = np.array([])
        for i in range(len(seg_starts)):
            x_seg = x[seg_starts[i]:seg_stops[i]]
            seg_win_starts, seg_win_stops = sliding_windows_start_stop_continuous(x_seg, win_len, slide_len)
            win_starts = np.append(win_starts, seg_win_starts)
            win_stops = np.append(win_stops, seg_win_stops)
    return win_starts, win_stops


def main():
    print("Running main")
    win_len = 5
    slide_len = 0
    x = np.arange(20)+10
    labels = np.zeros(20)
    labels[7:13] = 1
    win_starts, win_stops = sliding_windows_start_stop(x, win_len, slide_len)#, labels, labels_ignore=[1])
    print(x)
    print(labels)
    print(win_starts)
    print(win_stops)
    print(win_stops-win_starts+1)


if __name__ == '__main__':
    main()
