import glob
import os.path
import shutil
import cv2 as cv
import numpy as np

from src.AnnotationView import AnnotationView
from src.BioData import BioData
from src.BioFeatures import BioFeatures
from src.VideoInfo import VideoInfos
from src.file.annotations import load_annotations
from src.util import get_bio_base_name, get_input_files


class Relabeller():
    def __init__(self, method, annotation_filename=None, max_relabel_match_distance=None):
        self.method = method
        self.max_relabel_match_distance = max_relabel_match_distance
        if annotation_filename is not None:
            self.annotations = load_annotations(annotation_filename)

    def relabel_all(self, data_files, tracks_relabel_dir, video_files):
        video_infos = VideoInfos(video_files)
        data_sets = list(set([get_bio_base_name(data_file) for data_file in data_files]))
        for data_set in data_sets:
            print('Data set:', data_set)
            data_files1 = [data_file for data_file in data_files if data_set in data_file]
            video_info = video_infos.find_match(data_set)
            if self.method == 'annotation':
                self.relabel_annotation(data_files1, tracks_relabel_dir, video_info)
            elif self.method.startswith('sort'):
                self.relabel_sort(data_files1, tracks_relabel_dir, video_info)

    def relabel_sort(self, data_files, tracks_relabel_dir, video_info):
        sort_key = self.method.split()[-1]
        datas = [BioFeatures(data_file) for data_file in data_files]
        values = [data.get_mean_feature(sort_key) for data in datas]
        datas = [data for value, data in sorted(zip(values, datas), reverse=True)]
        for new_label, data in enumerate(datas):
            data.pref_label = new_label
            filename, extension = os.path.splitext(os.path.basename(data.filename))
            new_filename = os.path.join(tracks_relabel_dir, filename.rsplit('_', 1)[0] + str(new_label) + extension)
            shutil.copy2(data.filename, new_filename)

    def relabel_annotation(self, data_files, tracks_relabel_dir, video_info):
        # Reading labels
        datas = []
        for data_file in data_files:
            data = BioData(data_file)
            pref_label, pref_label_dist = self.get_near_label(data)
            if pref_label is not None:
                data.pref_label, data.pref_label_dist = pref_label, pref_label_dist
                datas.append(data)

        # Selecting labels
        for x, y, label in self.annotations:
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
        mindist = -1
        label = None
        for annotation in self.annotations:
            dist = np.sqrt((annotation[0] - data.meanx) ** 2 + (annotation[1] - data.meany) ** 2)
            if dist < mindist or mindist < 0:
                mindist = dist
                label = annotation[2]
        if mindist > self.max_relabel_match_distance:
            # closest is too far away; disqualify
            return None, None
        return label, mindist


def annotate(annotation_image_filename, annotation_filename, annotation_margin):
    image = cv.imread(annotation_image_filename)
    if image is None:
        raise OSError(f'File not found: {annotation_image_filename}')
    annotator = AnnotationView(image, annotation_filename, annotation_margin)
    annotator.show_loop()
    annotator.close()


def run(general_params, params):
    base_dir = general_params['base_dir']
    method = params['method']
    input_files = get_input_files(general_params, params, 'input')
    video_files = get_input_files(general_params, params, 'video_input')
    output_dir = os.path.join(base_dir, params['output'])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        [os.remove(file) for file in glob.glob(os.path.join(output_dir, '*'))]

    if method.lower() == 'annotation':
        annotation_filename = os.path.join(base_dir, params['annotation_filename'])
        annotation_image_filename = os.path.join(base_dir, params['annotation_image'])
        annotation_margin = params['annotation_margin']
        max_relabel_match_distance = params['max_relabel_match_distance']
        annotate(annotation_image_filename, annotation_filename, annotation_margin)
    else:
        annotation_filename = None
        max_relabel_match_distance = None

    relabeller = Relabeller(method, annotation_filename, max_relabel_match_distance)
    relabeller.relabel_all(input_files, output_dir, video_files)
