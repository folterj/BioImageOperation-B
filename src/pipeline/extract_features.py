import csv
from datetime import timedelta
import os
from tqdm import tqdm

from src.VideoInfo import VideoInfos
from src.Data import Data, create_datas
from src.pipeline.analyse_contact import extract_contact_events
from src.pipeline.analyse_paths import extract_path_events
from src.util import list_to_str, get_bio_base_name, get_input_files, \
    find_all_filename_infos, get_input_stats, filter_output_files


def run(all_params, params):
    general_params = all_params['general']
    base_dir = general_params['base_dir']
    fps = general_params.get('fps')
    pixel_size = general_params.get('pixel_size')
    window_size = str(general_params.get('window_size'))
    add_missing_data_flag = bool(general_params.get('add_missing', False))

    input_files = get_input_files(general_params, params, 'input')
    if len(input_files) == 0:
        raise ValueError('Missing input files')
    print(f'Input files: {len(input_files)}')
    video_files = filter_output_files(get_input_files(general_params, params, 'video_input'), all_params)
    print(f'Video files: {len(video_files)}')
    video_infos = VideoInfos(video_files)
    print(f'Total length: {timedelta(seconds=int(video_infos.total_length))} (frames: {video_infos.total_frames})')
    print(get_input_stats(input_files))

    print('Reading input files')
    datas = create_datas(input_files, fps=fps, pixel_size=pixel_size, window_size=window_size)
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
            for data in tqdm(datas):
                if data.has_data:
                    data.calc_windows()
                    data.calc_means()
                    data.calc_profiles()

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
                        for feature in features:
                            row += list_to_str(data.features[feature].values())
                    csvwriter.writerow(row)

        elif feature_type == 'activity':
            for feature in features:
                for data in tqdm(datas):
                    if data.has_data:
                        data.classify_activity(output_type=feature)
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
            outputs = extract_events(datas, features, feature_set, fps)
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


def extract_events(datas, features, params, fps):
    event_type = params['type']
    output = {}
    data_set_infos = list(set(['_'.join(data.info) for data in datas]))
    for data_set_info in tqdm(data_set_infos):
        datas1 = [data for data in datas if data_set_info == '_'.join(data.info)]
        if 'contact' in event_type:
            output[data_set_info] = extract_contact_events(datas1, features, params)
        elif 'path' in event_type:
            output[data_set_info] = extract_path_events(datas1, features, params, fps)
    return output


def add_missing_data(datas0, files):
    datas = datas0.copy()
    all_infos, all_ids = find_all_filename_infos(files)
    for info in all_infos:
        for id in all_ids:
            id_info = [id] + info
            if not contains_data(datas, id_info):
                datas.append(Data(info=info, id=id))
    return datas


def contains_data(datas, id_info):
    for data in datas:
        if data.id_info == id_info:
            return True
    return False
