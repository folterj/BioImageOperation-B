import csv
import os
from datetime import timedelta
import numpy as np
from tqdm import tqdm
from sys import exit

from src.VideoInfo import VideoInfos
from src.BioFeatures import BioFeatures
from src.util import list_to_str, get_bio_base_name, get_input_files, calc_dist, \
    find_all_filename_infos, get_input_stats, filter_output_files


def extract_events_all(datas, features, contact_distance, activity_frames_range):
    output = {}
    data_set_infos = list(set(['_'.join(data.info) for data in datas]))
    for data_set_info in tqdm(data_set_infos):
        datas1 = [data for data in datas if data_set_info == '_'.join(data.info)]
        output[data_set_info] = extract_events(datas1, features, contact_distance, activity_frames_range)
    return output


def extract_events(datas, features, contact_distance, activity_frames_range):
    out_features = []
    log_frames = {}
    log_times = {}
    activities = {}
    activities0 = {}

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
                        activities[datai] = get_typical_activity(data.activity, frame1, activity_frames_range)
                        activities0[datai] = get_typical_activity(data0.activity, frame1, activity_frames_range)
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


def get_typical_activity(activity, central_frame, frames_range):
    activities = []
    for frame in range(central_frame - frames_range, central_frame + frames_range):
        if frame in activity:
            activity1 = activity[frame]
            if activity1 != '':
                activities.append(activity1)
    if len(activities) > 0:
        return sorted(activities)[len(activities) // 2]
    return ''


def run(all_params, params):
    general_params = all_params['general']
    base_dir = general_params['base_dir']
    add_missing_data_flag = bool(general_params.get('add_missing', False))

    input_files = get_input_files(general_params, params, 'input')
    print(f'Input files: {len(input_files)}')
    video_files = filter_output_files(get_input_files(general_params, params, 'video_input'), all_params)
    print(f'Video files: {len(video_files)}')
    video_infos = VideoInfos(video_files)
    print(f'Total length: {timedelta(seconds=int(video_infos.total_length))} (frames: {video_infos.total_frames})')
    print(get_input_stats(input_files))

    print('Reading input files')
    datas = [BioFeatures(filename) for filename in tqdm(input_files)]

    if add_missing_data_flag:
        datas = add_missing_data(datas, input_files)
        print(f'Added missing data to total of: {len(datas)}')

    header_start = ['ID']
    nheaders = max([len(data.info) for data in datas])
    for i in range(nheaders):
        header_start += [f'info{i + 1}']

    features_done = []
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
                        row = list_to_str(data.id_info)
                        if data.has_data:
                            row += list_to_str(data.profiles[feature][0])
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
                    row = list_to_str(data.id_info)
                    if data.has_data:
                        row += list_to_str(data.features['v_percentiles'].values())
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
                        row = list_to_str(data.id_info)
                        if data.has_data:
                            video_info = video_infos.find_match(get_bio_base_name(data.filetitle))
                            if video_info is not None:
                                total_frames = video_info.total_frames
                            else:
                                total_frames = None
                            data.classify_activity(output_type=feature)

                            for activity_type in activity_types:
                                row.append(data.get_activity_time(activity_type))
                                row.append(data.get_activity_fraction(activity_type, total_frames))
                        csvwriter.writerow(row)

        elif feature_type == 'events':
            if 'activity' not in features_done:
                print('Feature error: Events requires prior Activity extraction')
                exit(1)
            contact_distance = feature_set['contact_distance']
            activity_frames_range = feature_set['activity_frames_range']
            outputs = extract_events_all(datas, features, contact_distance, activity_frames_range)
            output_filename = os.path.join(base_dir, feature_set['output'])

            nheaders = max([len(data.info) for data in datas])
            header = []
            for i in range(nheaders):
                header += [f'info{i + 1}']
            header += list(features)

            with open(output_filename, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(header)
                for key in outputs:
                    info = key.split('_')
                    row = info + outputs[key]
                    csvwriter.writerow(row)

        features_done.append(feature_type)


def add_missing_data(datas0, files):
    datas = datas0.copy()
    all_infos, all_ids = find_all_filename_infos(files)
    for info in all_infos:
        for id in all_ids:
            id_info = [id] + info
            if not contains_data(datas, id_info):
                datas.append(BioFeatures(info=info, id=id))
    return datas


def contains_data(datas, id_info):
    for data in datas:
        if data.id_info == id_info:
            return True
    return False
