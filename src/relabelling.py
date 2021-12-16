import glob
import os.path
import cv2 as cv
import numpy as np

from src.AnnotationView import AnnotationView
from src.BioData import BioData
from src.file.annotations import load_annotations
from src.file.bio import import_tracks_by_frame
from src.parameters import *
from src.util import get_filetitle, get_filetitle_replace
from src.video import video_info, annotate_videos


class Relabeler():
    def __init__(self, annotation_filename):
        self.annotations = load_annotations(annotation_filename)

    def process_all(self, data_files, video_files):
        for video_file in video_files:
            video_title = get_filetitle_replace(video_file)
            print('Video:', video_title)
            _, _, nframes, _ = video_info(video_file)
            data_files1 = [data_file for data_file in data_files if video_title in data_file]
            self.process(data_files1, video_title, nframes)

    def process(self, data_files, video_title, total_frames):
        # Reading labels
        datas = []
        for data_file in data_files:
            data = BioData(data_file, video_title)
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
                new_filename = os.path.join(TRACKS_RELABEL_PATH, new_title + extension)
                if not os.path.exists(TRACKS_RELABEL_PATH):
                    os.makedirs(TRACKS_RELABEL_PATH)
                with open(new_filename, 'w') as outfile:
                    outfile.write(data1.header)
                    for line in lines.values():
                        outfile.write(line)
            print(f'Label: {label} Coverage: {total_coverage / total_frames * 100:0.1f}%')

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
        if mindist > MAX_MOVE_DISTANCE:
            # closest is too far away; disqualify
            return None, None
        return label, mindist


def annotate():
    image = cv.imread(LABEL_ANNOTATION_IMAGE)
    annotator = AnnotationView(image, LABEL_ANNOTATION_FILENAME)
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


if __name__ == '__main__':
    if not os.path.exists(LABEL_ANNOTATION_FILENAME):
        annotate()

    relabeler = Relabeler(LABEL_ANNOTATION_FILENAME)
    input_files = sorted(glob.glob(TRACKS_PATH))
    video_files = sorted(glob.glob(VIDEOS_PATH))
    relabeler.process_all(input_files, video_files)

    relabelled_files = sorted(glob.glob(TRACKS_RELABEL_FILES))
    annotate_merge_videos(relabelled_files, video_files, VIDEOS_OUTPUT)
