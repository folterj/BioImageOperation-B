import glob
import os.path
import cv2 as cv
import numpy as np

from src.AnnotationView import AnnotationView
from src.Data import Data
from src.Features import create_features
from src.VideoInfo import VideoInfos
from src.file.bio import export_tracks
from src.file.generic import import_file
from src.file.plain_csv import export_csv
from src.util import get_bio_base_name, get_input_files, numeric_string_sort, filter_output_files, calc_dist, \
    calc_mean_dist


class Relabeller():
    def __init__(self, params, annotation_filename=''):
        self.method = params['method']
        self.input_pixel_size = params.get('input_pixel_size', 1)
        self.max_relabel_match_distance = params.get('max_relabel_match_distance', 0)
        if annotation_filename != '':
            self.annotations = import_file(annotation_filename, add_position=True)

    def relabel_all(self, data_files, tracks_relabel_dir, video_files):
        video_infos = VideoInfos(video_files)
        data_sets = numeric_string_sort(list(set([get_bio_base_name(data_file) for data_file in data_files])))
        for data_set in data_sets:
            print('Data set:', data_set)
            data_files1 = [data_file for data_file in data_files if data_set in data_file]
            video_info = video_infos.find_match(data_set)
            if self.method == 'annotation':
                self.relabel_annotation(data_files1, tracks_relabel_dir, video_info)
            elif self.method.startswith('sort'):
                self.relabel_sort(data_files1, tracks_relabel_dir, video_info)
            elif self.method == 'gt':
                self.relabel_gt(data_files1, tracks_relabel_dir, video_info)

    def relabel_sort(self, data_files, tracks_relabel_dir, video_info):
        sort_key = self.method.split()[-1]
        datas = [Data(data_file) for data_file in data_files]
        values = [data.get_mean_feature(sort_key) for data in datas]
        datas = [data for value, data in sorted(zip(values, datas), reverse=True)]
        for new_label0, data in enumerate(datas):
            new_label = str(new_label0)
            data.set_new_label(new_label)
            filename, extension = os.path.splitext(os.path.basename(data.filename))
            new_filename = os.path.join(tracks_relabel_dir, filename.rsplit('_', 1)[0] + '_' + new_label + extension)
            export_csv(new_filename, {new_label: data.data})

    def relabel_annotation(self, data_files, tracks_relabel_dir, video_info):
        # Reading labels & find nearest
        datas = []
        for data_file in data_files:
            data = Data(data_file)
            best_label, best_dist = self.get_near_label(data)
            if best_label is not None:
                data.set_new_label(best_label, best_dist)
                datas.append(data)

        self.save_data_files(datas, tracks_relabel_dir, video_info)

    def save_data_files(self, datas, tracks_relabel_dir, video_info):
        for label in self.annotations:
            datas1 = [data for data in datas if data.new_label == label]
            all_frames = []
            for data in datas1:
                all_frames.extend(data.frames)
            all_frames = sorted(set(all_frames))
            datas1.sort(key=lambda data: data.match_dist)
            frames_data = {}
            total_coverage = 0
            for frame in all_frames:
                for data in datas1:
                    frame_data = data.get_frame_data(frame)
                    if frame_data is not None:
                        total_coverage += 1
                        if len(frames_data) == 0:
                            frames_data = frame_data
                        else:
                            for key, value in frame_data.items():
                                frames_data[key] |= value

            if len(frames_data) > 0:
                data1 = datas1[0]
                extension = os.path.splitext(data1.filename)[1]
                new_filename = os.path.join(tracks_relabel_dir, data1.new_title + extension)
                export_csv(new_filename, {label: frames_data})
            if video_info is not None:
                print(f'Label: {label} Coverage: {total_coverage / video_info.total_frames * 100:0.1f}%')
            else:
                print(f'Label: {label} Coverage: {total_coverage} frames')

    def get_near_label(self, data):
        mindist = None
        label = None
        for annotation_id, annotation in self.annotations.items():
            position = annotation['position']
            if isinstance(position, dict):
                position = position[0]
            dist = calc_dist((data.meanx, data.meany), position)
            if mindist is None or dist < mindist:
                mindist = dist
                label = annotation_id
        if mindist is None or 0 < self.max_relabel_match_distance < mindist:
            # closest is too far away; disqualify
            return None, None
        return label, mindist

    def relabel_gt(self, data_files, tracks_relabel_dir, video_info):
        final_matches = {}
        matches = {}
        datas = create_features(data_files)
        data_dict = {data.id: data for data in datas}
        available_tracks = list(data_dict)
        position_factor = 1 / self.input_pixel_size
        if position_factor != 1:
            for data in datas:
                data.position.update((frame, (position[0] * position_factor, position[1] * position_factor))
                                     for frame, position in data.position.items())
        for annotation, gt_values in self.annotations.items():
            distances = {}
            for data in datas:
                distances[data.id] = calc_mean_dist(gt_values['position'], data.position)
            matches[annotation] = dict(sorted(distances.items(), key=lambda item: item[1]))
        matches = dict(sorted(matches.items(), key=lambda item: next(iter(item[1].items()))[1]))

        distances = []
        for gt_id, matches1 in matches.items():
            for label, dist in matches1.items():
                if label in available_tracks:
                    final_matches[gt_id] = label
                    available_tracks.remove(label)
                    distances.append(dist)
                    break

        print(f'#Matches: {len(final_matches)}')
        mean_dist = np.mean(distances)
        print(f'Mean distance: {mean_dist:.1f}')
        match_rate, match_dist = self.get_match_rate(final_matches, data_dict)
        print(f'Match rate: {match_rate:.4f}')
        print(f'Mean match distance: {match_dist:.1f}')

        save_files(final_matches, data_files, tracks_relabel_dir)

    def get_match_rate(self, matches, data_dict):
        distances = []
        nmatches = 0
        total = 0
        for gt_id, track_id in matches.items():
            positions0 = self.annotations[gt_id]['position']
            positions1 = data_dict[track_id].position
            for frame in positions0:
                if frame in positions1:
                    dist = calc_dist(positions0[frame], positions1[frame])
                    if dist <= self.max_relabel_match_distance:
                        nmatches += 1
                    distances.append(dist)
            total += len(positions0)
        match_rate = nmatches / total
        return match_rate, np.mean(distances)


def save_files(matches, data_files, tracks_relabel_dir):
    datas = {}
    single_file = (len(data_files) == 1)
    for i, data_file in enumerate(data_files):
        basename, extension = os.path.splitext(os.path.basename(data_file))
        parts = basename.split('_')
        if not single_file and len(parts) > 0:
            basename = '_'.join(parts[:-1])
        datas |= import_file(data_file)
    for annotation_id, track_id in matches.items():
        data = datas[track_id]
        if 'id' in data:
            data['id'] = annotation_id
        elif 'track_label' in data:
            data['track_label'] = annotation_id
        new_filename = os.path.join(tracks_relabel_dir, basename)
        if len(data_files) > 1:
            new_filename += '_' + annotation_id
        new_filename += extension
        export_tracks(new_filename, data)


def annotate(annotation_image_filename, annotation_filename, annotation_margin):
    image = cv.imread(annotation_image_filename)
    if image is None:
        raise OSError(f'File not found: {annotation_image_filename}')
    print('User action: Review annotations (press ESC when finished)')
    annotator = AnnotationView(image, annotation_filename, annotation_margin)
    annotator.show_loop()
    annotator.close()


def run(all_params, params):
    general_params = all_params['general']
    base_dir = general_params['base_dir']
    method = params['method']
    input_files = get_input_files(general_params, params, 'input')
    if len(input_files) == 0:
        raise ValueError('Missing input files')
    video_files = filter_output_files(get_input_files(general_params, params, 'video_input'), all_params)
    output_dir = os.path.join(base_dir, params['output'])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        [os.remove(file) for file in glob.glob(os.path.join(output_dir, '*'))]

    annotation_filename = params.get('annotation_filename', '')
    if annotation_filename != '':
        annotation_filename = os.path.join(base_dir, annotation_filename)
    annotation_image_filename = params.get('annotation_image', '')
    if annotation_image_filename != '':
        annotation_image_filename = os.path.join(base_dir, annotation_image_filename)
    annotation_margin = params.get('annotation_margin', 0)

    if method.lower() == 'annotation':
        annotate(annotation_image_filename, annotation_filename, annotation_margin)

    relabeller = Relabeller(params, annotation_filename)
    relabeller.relabel_all(input_files, output_dir, video_files)
