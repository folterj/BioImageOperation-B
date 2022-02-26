import csv
import os
import numpy as np
from tqdm import tqdm

from src.VideoInfo import VideoInfos
from src.BioFeatures import BioFeatures
from src.util import list_to_str, get_bio_base_name, get_input_files, extract_filename_info, calc_dist


def extract_events_all(datas, features, contact_distance):
    output = {}
    data_sets = list(set([get_bio_base_name(data.filename) for data in datas]))
    for data_set in data_sets:
        datas1 = [data for data in datas if data_set in data.filename]
        output[data_set] = extract_events(datas1, features, contact_distance)
    return output


def extract_events(datas, features, contact_distance):
    out_features = []
    log_frames = {}
    log_times = {}
    activities = {}
    activities0 = {}

    data0 = datas[0]
    frames = data0.frames

    for datai0, data in enumerate(datas[1:]):
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
                    activities[datai] = data.activity[frame1]
                    activities0[datai] = data0.activity[frame1]
                    break
            if active:
                last_frame = frame

    log_frames = np.asarray(sorted(log_frames.items(), key=lambda item: item[1]))
    log_times = np.asarray(sorted(log_times.items(), key=lambda item: item[1]))
    activities0 = [value for key, value in sorted(activities0.items())]
    activities = [value for key, value in sorted(activities.items())]
    n = len(log_times)
    times = log_times[:, 1]
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


def run(general_params, params):
    base_dir = general_params['base_dir']

    input_files = get_input_files(general_params, general_params, 'input')
    video_files = get_input_files(general_params, general_params, 'video_input')
    video_infos = VideoInfos(video_files)
    header_start = ['ID', 'Date', 'Time', 'Camera']

    print('Reading input files')
    datas = [BioFeatures(filename) for filename in input_files]

    for feature_set0 in params:
        feature_type = next(iter(feature_set0))
        feature_set = feature_set0[feature_type]
        features = feature_set['features']
        print(f'Extracting {feature_type}')

        if feature_type == 'profiles':
            [data.calc_profiles() for data in tqdm(datas)]
            for feature in features:
                output_filename = os.path.join(base_dir, feature_set['output'].format_map({'feature': feature}))
                header = header_start + list_to_str(datas[0].profiles[feature][1])
                with open(output_filename, 'w', newline='') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow(header)
                    for data in datas:
                        row = list_to_str(data.info) + list_to_str(data.profiles[feature][0])
                        csvwriter.writerow(row)

        elif feature_type == 'features':
            output_filename = os.path.join(base_dir, feature_set['output'])
            header = list(header_start)
            for feature in features:
                header += list(datas[0].features[feature].keys())

            with open(output_filename, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(header)
                for data in datas:
                    row = list_to_str(data.info) + list_to_str(data.features['v_percentiles'].values())
                    csvwriter.writerow(row)

        elif feature_type == 'activity':
            for feature in features:
                [data.classify_activity(output_type=feature) for data in tqdm(datas)]
                output_filename = os.path.join(base_dir, feature_set['output'].format_map({'feature': feature}))

                activity_types = [key for key in datas[0].get_activities_time().keys() if key != '']
                header = list(header_start)
                for activity_type in activity_types:
                    header += [f'{activity_type} [s]', f'{activity_type} [%]']

                with open(output_filename, 'w', newline='') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow(header)
                    for data in datas:
                        video_info = video_infos.find_match(get_bio_base_name(data.filetitle))
                        if video_info is not None:
                            total_frames = video_info.total_frames
                        else:
                            total_frames = None
                        data.classify_activity(output_type=feature)

                        row = list_to_str(data.info)
                        for activity_type in activity_types:
                            row.append(data.get_activity_time(activity_type))
                            row.append(data.get_activity_fraction(activity_type, total_frames))
                        csvwriter.writerow(row)

        elif feature_type == 'events':
            contact_distance = feature_set['contact_distance']
            outputs = extract_events_all(datas, features, contact_distance)
            output_filename = os.path.join(base_dir, feature_set['output'])
            header = ['Date', 'Time', 'Camera'] + list(features)
            with open(output_filename, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(header)
                for output in outputs:
                    info = extract_filename_info(output)
                    row = info[1:] + outputs[output]
                    csvwriter.writerow(row)
