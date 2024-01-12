import numpy as np

from src.util import calc_dist


def extract_contact_events(datas, features, params):
    out_features = []
    log_frames = {}
    log_times = {}
    activities = {}
    activities0 = {}

    contact_distance = params['distance']
    frame_range = params['frame_range']

    data0 = datas[0]
    frames = data0.frames

    for datai0, data in enumerate(datas[1:]):
        if data.has_data:
            datai = datai0 + 1
            last_frame = None
            for frame in frames:
                active = frame in data.frames
                if active:
                    frame1 = frame
                else:
                    frame1 = last_frame
                if frame1 is not None:
                    merged = data.data['is_merged'][frame1]
                    dist = calc_dist((data0.data['x'][frame1], data0.data['y'][frame1]), (data.data['x'][frame1], data.data['y'][frame1]))
                    if (not active or merged) and dist < contact_distance:
                        log_frames[datai] = frame1
                        log_times[datai] = frame1 * data.dtime
                        activities[datai] = get_typical_activity(data.activity, frame1, frame_range)
                        activities0[datai] = get_typical_activity(data0.activity, frame1, frame_range)
                        break
                if active:
                    last_frame = frame

    log_frames = np.asarray(sorted(log_frames.items(), key=lambda item: item[1]))
    log_times = np.asarray(sorted(log_times.items(), key=lambda item: item[1]))
    activities0 = [value for key, value in sorted(activities0.items())]
    activities = [value for key, value in sorted(activities.items())]
    n = len(log_times)
    if n > 0:
        times = log_times[:, 1]
    else:
        times = []
    delta_times = []
    last_time = 0
    for time in times:
        delta_times.append(time - last_time)
        last_time = time

    for feature in features:
        if feature == 'n':
            out_features.append(n)
        elif feature == 'time':
            out_features.append(list(times))
        elif feature == 'delta_time':
            out_features.append(list(delta_times))
        elif feature.startswith('activity'):
            parts = feature.split()
            if len(parts) > 1 and parts[1] == '0':
                out_features.append(activities0)
            else:
                out_features.append(activities)

    return out_features


def get_typical_activity(activity, central_frame, frame_range):
    activities = []
    for frame in range(central_frame - frame_range, central_frame + frame_range):
        if frame in activity:
            activity1 = activity[frame]
            if activity1 != '':
                activities.append(activity1)
    if len(activities) > 0:
        return sorted(activities)[len(activities) // 2]
    return ''
