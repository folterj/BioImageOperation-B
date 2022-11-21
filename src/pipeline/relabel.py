import glob
import os.path
import shutil
import cv2 as cv
import numpy as np

from src.AnnotationView import AnnotationView
from src.BioData import BioData
from src.BioFeatures import create_biofeatures
from src.VideoInfo import VideoInfos
from src.file.bio import export_tracks
from src.file.plain_csv import import_csv
from src.util import get_bio_base_name, get_input_files, numeric_string_sort, filter_output_files, calc_dist, \
    calc_mean_dist


class Relabeller():
    def __init__(self, method, annotation_filename='', max_relabel_match_distance=0):
        self.method = method
        self.max_relabel_match_distance = max_relabel_match_distance
        if annotation_filename != '':
            self.annotations = import_csv(annotation_filename, add_position=True)

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
        datas = create_biofeatures(data_files)
        values = [data.get_mean_feature(sort_key) for data in datas]
        datas = [data for value, data in sorted(zip(values, datas), reverse=True)]
        for new_label, data in enumerate(datas):
            data.pref_label = new_label
            filename, extension = os.path.splitext(os.path.basename(data.filename))
            new_filename = os.path.join(tracks_relabel_dir, filename.rsplit('_', 1)[0] + '_' + str(new_label) + extension)
            shutil.copy2(data.filename, new_filename)

    def relabel_annotation(self, data_files, tracks_relabel_dir, video_info):
        # Reading labels & find nearest
        datas = []
        for data_file in data_files:
            data = BioData(data_file)
            pref_label, pref_label_dist = self.get_near_label(data)
            if pref_label is not None:
                data.pref_label, data.pref_label_dist = pref_label, pref_label_dist
                datas.append(data)

        self.save_data_files(datas, tracks_relabel_dir, video_info)

    def save_data_files(self, datas, tracks_relabel_dir, video_info):
        for label in self.annotations:
            datas1 = [data for data in datas if data.pref_label == label]
            all_frames = []
            for data in datas1:
                all_frames.extend(data.frames)
            all_frames = sorted(set(all_frames))
            datas1.sort(key=lambda data: data.pref_label_dist)
            lines = {}
            total_coverage = 0
            for frame in all_frames:
                for data in datas1:
                    if frame in data.frames:
                        total_coverage += 1
                        lines[frame] = data.lines[frame]
                        break

            if len(lines) > 0:
                data1 = datas1[0]
                filename = data1.filename
                extension = os.path.splitext(filename)[1]
                new_title = data1.old_title
                if new_title.endswith(data1.old_label):
                    new_title = new_title.rstrip(data1.old_label)
                new_title += label
                new_filename = os.path.join(tracks_relabel_dir, new_title + extension)
                with open(new_filename, 'w') as outfile:
                    outfile.write(data1.header)
                    for line in lines.values():
                        outfile.write(line)
            if video_info is not None:
                print(f'Label: {label} Coverage: {total_coverage / video_info.total_frames * 100:0.1f}%')
            else:
                print(f'Label: {label} Coverage: {total_coverage} frames')

    def get_best_label_from_file(self, filename):
        data = BioData(filename)
        new_label, _ = self.get_near_label(data)
        new_title = data.old_title
        if new_title.endswith(data.old_label):
            new_title = new_title.rstrip(data.old_label)
        new_title += new_label
        return new_label, new_title, data.old_label, data.old_title

    def get_near_label(self, data):
        mindist = None
        label = None
        for annotation, position in self.annotations.items():
            dist = calc_dist((data.meanx, data.meany), position)
            if mindist is None or dist < mindist:
                mindist = dist
                label = annotation
        if mindist is None or 0 < self.max_relabel_match_distance < mindist:
            # closest is too far away; disqualify
            return None, None
        return label, mindist

    def relabel_gt(self, data_files, tracks_relabel_dir, video_info):
        final_matches = {}
        matches = {}
        datas = create_biofeatures(data_files)
        available_tracks = [data.id for data in datas]
        for annotation, gt_values in self.annotations.items():
            distances = {}
            for data in datas:
                distances[data.id] = calc_mean_dist(gt_values['position'], data.position)
            matches[annotation] = dict(sorted(distances.items(), key=lambda item: item[1]))
        matches = dict(sorted(matches.items(), key=lambda item: next(iter(item[1].items()))[1]))

        distances = []
        for gt_id, matches1 in matches.items():
            for label, dist in matches1.items():
                if label in available_tracks:   # and dist <= self.max_relabel_match_distance:
                    final_matches[gt_id] = label
                    available_tracks.remove(label)
                    distances.append(dist)
                    break

        print(f'#Matches: {len(final_matches)}')
        mean_dist = np.mean(distances)
        print(f'Mean distance: {mean_dist:.1f}')
        match_rate, min_dist = self.get_match_rate(final_matches, datas)
        print(f'Match rate: {match_rate:.3f}')
        print(f'Minimum match distance (mean): {min_dist:.1f}')

        save_files(final_matches, data_files, tracks_relabel_dir)

    def get_match_rate(self, matches, datas):
        correct = []
        min_dists = []
        for gt_id, track_id in matches.items():
            positions1 = []
            for data in datas:
                if data.id == track_id:
                    positions1 = data.position
            for frame in positions1:
                min_dist = None
                gt_id1 = None
                for gt_id0, values0 in self.annotations.items():
                    if frame in values0['x']:
                        dist = calc_dist(values0['position'][frame], positions1[frame])
                        if min_dist is None or dist < min_dist:
                            min_dist = dist
                            gt_id1 = gt_id0
                correct.append(gt_id1 == gt_id)
                if min_dist is not None:
                    min_dists.append(min_dist)
        match_rate = np.mean(correct)
        return match_rate, np.mean(min_dists)


def save_files(matches, data_files, tracks_relabel_dir):
    datas = {}
    single_file = (len(data_files) == 1)
    for i, data_file in enumerate(data_files):
        basename, extension = os.path.splitext(os.path.basename(data_file))
        parts = basename.split('_')
        if not single_file and len(parts) > 0:
            basename = '_'.join(parts[:-1])
        datas |= import_csv(data_file)
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

    annotation_filename = os.path.join(base_dir, params.get('annotation_filename', ''))
    annotation_image_filename = os.path.join(base_dir, params.get('annotation_image', ''))
    annotation_margin = params.get('annotation_margin', 0)
    max_relabel_match_distance = params.get('max_relabel_match_distance', 0)

    if method.lower() == 'annotation':
        annotate(annotation_image_filename, annotation_filename, annotation_margin)

    relabeller = Relabeller(method, annotation_filename, max_relabel_match_distance)
    relabeller.relabel_all(input_files, output_dir, video_files)
