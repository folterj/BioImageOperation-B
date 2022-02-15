import glob
import os.path
import cv2 as cv
import numpy as np

from src.AnnotationView import AnnotationView
from src.BioData import BioData
from src.VideoInfo import VideoInfos
from src.file.annotations import load_annotations
from src.file.bio import import_tracks_by_frame
from src.util import get_filetitle, get_filetitle_replace
from src.video import video_info, annotate_videos


class Relabeller():
    def __init__(self, annotation_filename, max_move_distance):
        self.max_move_distance = max_move_distance
        self.annotations = load_annotations(annotation_filename)

    def process_all(self, data_files, tracks_relabel_dir, video_files):
        video_infos = VideoInfos(video_files)
        data_sets = list(set([os.path.basename(data_file).rsplit('_', 1)[0] for data_file in data_files]))
        for data_set in data_sets:
            print('Data set:', data_set)
            data_files1 = [data_file for data_file in data_files if data_set in data_file]
            video_info = video_infos.find_match(data_set)
            self.process(data_files1, tracks_relabel_dir, video_info)

    def process(self, data_files, tracks_relabel_dir, video_info):
        # Reading labels
        datas = []
        for data_file in data_files:
            data = BioData(data_file)
            pref_label, pref_label_dist = self.get_best_label(data)
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
                if not os.path.exists(tracks_relabel_dir):
                    os.makedirs(tracks_relabel_dir)
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
        new_label, _ = self.get_best_label(data)
        new_title = data.old_title
        if new_title.endswith(data.old_label):
            new_title = new_title.rstrip(data.old_label)
        new_title += new_label
        return new_label, new_title, data.old_label, data.old_title

    def get_best_label(self, data):
        mindist = -1
        label = None
        for annotation in self.annotations:
            dist = np.sqrt((annotation[0] - data.meanx) ** 2 + (annotation[1] - data.meany) ** 2)
            if dist < mindist or mindist < 0:
                mindist = dist
                label = annotation[2]
        if mindist > self.max_move_distance:
            # closest is too far away; disqualify
            return None, None
        return label, mindist


def annotate(annotation_image_filename, annotation_filename, max_annotation_distance):
    image = cv.imread(annotation_image_filename)
    annotator = AnnotationView(image, annotation_filename, max_annotation_distance)
    annotator.show_loop()
    annotator.close()


def annotate_merge_videos(input_files, video_files, video_output):
    print('Reading relabelled data')
    all_datas = {}
    for video_file in video_files:
        video_datas = {}
        video_title = get_filetitle_replace(video_file)
        for filename in input_files:
            if video_title in filename:
                title = get_filetitle(filename)
                label = title.rsplit('_')[-1]
                video_datas[label] = import_tracks_by_frame(filename)
        all_datas[video_title] = video_datas
    print('Creating annotated video')
    annotate_videos(video_files, video_output, all_datas, frame_inerval=100)


def relabel(params):
    base_dir = params['base_dir']
    annotation_filename = os.path.join(base_dir, params['label_annotation_filename'])
    annotation_image_filename = os.path.join(base_dir, params['label_annotation_image'])
    tracks_path = os.path.join(base_dir, params['tracks_path'])
    if os.path.isdir(tracks_path):
        tracks_path = os.path.join(tracks_path, '*')
    tracks_relabel_dir = os.path.join(base_dir, params['tracks_relabel_dir'])
    video_input_path = os.path.join(base_dir, params['video_input_path'])

    max_annotation_distance = params['max_annotation_distance']
    max_move_distance = params['max_move_distance']

    if not os.path.exists(annotation_filename):
        annotate(annotation_image_filename, annotation_filename, max_annotation_distance)

    relabeller = Relabeller(annotation_filename, max_move_distance)
    input_files = sorted(glob.glob(tracks_path))
    video_files = sorted(glob.glob(video_input_path))
    relabeller.process_all(input_files, tracks_relabel_dir, video_files)


def relabel_annotate_video(params):
    base_dir = params['base_dir']
    tracks_relabel_path = os.path.join(base_dir, params['tracks_relabel_dir'], '*')
    video_input_path = os.path.join(base_dir, params['video_input_path'])
    video_output_path = os.path.join(base_dir, params['video_output_path'])

    input_files = sorted(glob.glob(tracks_relabel_path))
    video_files = sorted(glob.glob(video_input_path))
    annotate_merge_videos(input_files, video_files, video_output_path)
