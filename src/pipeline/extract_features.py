import csv
import os
from tqdm import tqdm

from src.VideoInfo import VideoInfos
from src.BioFeatures import BioFeatures
from src.util import list_to_str, get_bio_base_name, get_input_files


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
        print(f'Extracting {feature_type}')

        if feature_type == 'profiles':
            [data.calc_profiles() for data in tqdm(datas)]
            for feature in feature_set['features']:
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
            for feature in feature_set['features']:
                header += list(datas[0].features[feature].keys())

            with open(output_filename, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(header)
                for data in datas:
                    row = list_to_str(data.info) + list_to_str(data.features['v_percentiles'].values())
                    csvwriter.writerow(row)

        elif feature_type == 'activity':
            for feature in feature_set['features']:
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
