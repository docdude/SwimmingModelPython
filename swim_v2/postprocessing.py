import numpy as np
import utils


def inside_pt_threshold(x, p_low, t_low, p_up=None, t_up=None):
    """
    Find where an array satisfies upper/lower probability and time bounds
    :param x: An array of probabilities
    :param p_low: A lower probability bound
    :param t_low: A lower time-interval bound
    :param p_up: An upper probability bound. Default: None
    :param t_up: An upper time-interval bound. Default: None
    :return: is_ns: A boolean 0/1 array
             null_starts: Indices where 1s start
             null_stops: Indices where 1s stop
    """
    if p_up is None:
        p_up = np.inf
    if t_up is None:
        t_up = np.inf
    p_ix = np.where((x < p_up) & (x >= p_low))[0]
    is_ns = np.zeros(len(x))
    if len(p_ix) == 0:
        return is_ns, np.array([]), np.array([])
    is_ns[p_ix] = 1
    null_starts, null_stops = utils.start_stop(is_ns)
    null_within_t = np.where(((null_stops - null_starts) >= t_low) &
                               ((null_stops - null_starts) < t_up))[0]
    if len(null_within_t) == 0:
        return is_ns, np.array([]), np.array([])
    null_starts = null_starts[null_within_t]
    null_stops = null_stops[null_within_t]
    for i in range(len(null_starts)):
        is_ns[null_starts[i]:null_stops[i]] = 1
    return is_ns, null_starts, null_stops


def smooth_nulls(x, p=0.5, t_low=1, close_size_1=3*30, open_size_1=9*30, close_size_2=25*30):
    """
    Find and smooth out long periods of null
    :param x: An array of probabilities for the null class
    :param p: Probability bound
    :param t_low: Time-interval bound
    :param close_size_1: The first closing bound
    :param open_size_1: The first opening bound
    :param close_size_2: The second closing bound
    :return: is_ns_3: A 0/1 array where 1 indicates null periods.
    """
    is_ns_0, _, _ = inside_pt_threshold(x, p_low=p, t_low=t_low)
    is_ns_1 = utils.close(is_ns_0, close_size=close_size_1)
    is_ns_2 = utils.unclose(is_ns_1, open_size=open_size_1)
    is_ns_3 = utils.close(is_ns_2, close_size=close_size_2)
    return is_ns_3


def smooth_turns(x, null_start_init, null_stop_init, min_lane=25*30, max_lane=75*30, pt_pairs=None):
    """
    Find turns between periods of null
    :param x: An array of probabilities for the null class
    :param null_start_init: Starting indices of null periods
    :param null_stop_init: Stopping indices of null periods
    :param min_lane: The minimum length of a lane in samples
    :param max_lane: The maximum length of a lane in samples
    :param pt_pairs: A list of tuples containing pairs of probability and time-interval bounds
    :return: null_start: The same as null_start_init with turns added
             null_stop: The same as null_stop_init with turns added
    """
    if pt_pairs is None:
        pt_pairs = [(0.5, 9), (0.5, 6), (0.25, 9), (0.25, 6), (0.5, 3), (0.25, 3)]
    null_start = np.copy(null_start_init)
    null_stop = np.copy(null_stop_init)
    for i in range(len(null_stop_init) - 1):
        win_prob = x[null_stop_init[i]:null_start_init[i + 1]]
        turn_start = np.array([])
        turn_stop = np.array([])
        for (ii, pt) in enumerate(pt_pairs):
            p_low = pt[0]
            t_low = pt[1] * 30
            if ii == 0:
                t_up = None
                p_up = None
            else:
                # p_up = pt_pairs[ii - 1][0]
                # t_up = pt_pairs[ii - 1][1] * 30
                p_up = None
                t_up = None
            _, turn_start_new, turn_stop_new = inside_pt_threshold(win_prob, p_low=p_low, t_low=t_low, p_up=p_up,
                                                                   t_up=t_up)
            valid_turns = np.where((turn_start_new >= min_lane) & (turn_stop_new < (len(win_prob) - min_lane)))[0]
            # if len(valid_turns) == 0:
            #     continue
            turn_start_new = turn_start_new[valid_turns]
            turn_stop_new = turn_stop_new[valid_turns]
            turn_keep = [True] * len(turn_start_new)
            for iii in range(len(turn_start_new)):
                for iv in range(len(turn_start)):
                    if np.abs(turn_start_new[iii] - turn_start[iv]) < min_lane:
                        # Probably an existing null
                        turn_keep[iii] = False
                    elif np.abs(turn_stop_new[iii] - turn_stop[iv]) < min_lane:
                        # Probably an existing null
                        turn_keep[iii] = False
                    if np.abs(turn_start_new[iii] - turn_stop[iv]) < min_lane:
                        # Starts too close to existing stop
                        turn_keep[iii] = False
                    elif np.abs(turn_stop_new[iii] - turn_start[iv]) < min_lane:
                        # Stops too late close to existing start
                        turn_keep[iii] = False
            turn_start_new = turn_start_new[turn_keep]
            turn_stop_new = turn_stop_new[turn_keep]
            turn_start = np.append(turn_start, turn_start_new).astype(int)
            turn_stop = np.append(turn_stop, turn_stop_new).astype(int)
        turn_start, turn_stop = resolve_clash(win_prob, turn_start, turn_stop, min_lane)
        turn_start = turn_start + null_stop_init[i]
        turn_stop = turn_stop + null_stop_init[i]
        null_start = np.append(null_start, turn_start).astype(int)
        null_stop = np.append(null_stop, turn_stop).astype(int)
    tmp = np.argsort(null_start)
    return null_start[tmp], null_stop[tmp]


def resolve_clash(x, turn_start, turn_stop, min_lane):
    """
    Resolve lanes that clash based on the min_lane threshold
    :param x: An array of probabilities
    :param turn_start: sample locations of turn staring locations
    :param turn_stop: sample locations of turn stopping locations
    :param min_lane: the minimum lane length threshold
    :return: turn_start and turn_stop with clashes resolved
    """
    pt = np.zeros(len(turn_start))
    for i in range(len(turn_start)):
        pt[i] = np.sum(x[turn_start[i]:turn_stop[i]])
    a = np.flip(np.argsort(pt), axis=0)
    turn_start_clean = np.array([])
    turn_stop_clean = np.array([])
    for i in range(len(a)):
        keep_turn = True
        for ii in range(len(turn_start_clean)):
            if np.abs(turn_start[a[i]] - turn_stop_clean[ii]) < min_lane:
                keep_turn = False
            elif np.abs(turn_start_clean[ii] - turn_stop[a[i]]) < min_lane:
                keep_turn = False
        if keep_turn:
            turn_start_clean = np.append(turn_start_clean, turn_start[a[i]]).astype(int)
            turn_stop_clean = np.append(turn_stop_clean, turn_stop[a[i]]).astype(int)
    return turn_start_clean, turn_stop_clean


def main():
    print("Main")


if __name__ == '__main__':
    main()

