import glob
import os.path
import shutil
import cv2 as cv
import numpy as np
from tqdm import tqdm

from src.AnnotationView import AnnotationView
from src.BioData import BioData
from src.file.annotations import load_annotations
from src.parameters import *
from src.util import get_filetitle
from src.video import video_info, annotate_videos


def annotate():
    image = cv.imread(LABEL_ANNOTATION_IMAGE)
    annotator = AnnotationView(image, LABEL_ANNOTATION_FILENAME)
    annotator.show_loop()
    annotator.close()


class Relabeler():
    def __init__(self, annotation_filename):
        self.annotations = load_annotations(annotation_filename)
        self.datas = []
        self.lengths = {}
        self.length = 0

    def process_all(self, data_files, video_files):
        print('Scanning video files')
        for video_file in tqdm(video_files):
            _, _, nframes, fps = video_info(video_file)
            title = get_filetitle(video_file).replace('.', '_')
            self.lengths[title] = nframes
            self.length += nframes

        print('Reading labels')
        datas = []
        for data_file in tqdm(data_files):
            start = self.find_start(data_file)
            data = BioData(data_file, start)
            pref_label, pref_label_dist = self.get_best_label(data)
            if pref_label is not None:
                data.pref_label, data.pref_label_dist = pref_label, pref_label_dist
                datas.append(data)

        print('Selecting labels')
        for x, y, label in self.annotations:
            datas1 = [data for data in datas if data.pref_label == label]
            datas1.sort(key=lambda data: (data.start, data.pref_label_dist))
            total_coverage = 0
            coverage = -1
            for data in datas1:
                if data.start > coverage:
                    self.datas.append(data)
                    coverage = data.end
                    total_coverage += data.length
            print(f'Label: {label} Coverage: {total_coverage / self.length * 100:0.1f}%')

        print('Storing new labels')
        # delete all files
        [os.remove(file) for file in glob.glob(TRACKS_RELABEL_PATH + '*')]
        for data in tqdm(self.datas):
            filename = data.filename
            extension = os.path.splitext(filename)[1]
            new_title = data.old_title
            if new_title.endswith(data.old_label):
                new_title = new_title.rstrip(data.old_label)
            new_title += data.pref_label
            new_filename = os.path.join(TRACKS_RELABEL_PATH, new_title + extension)
            if not os.path.exists(new_filename):
                shutil.copy(filename, new_filename)
            else:
                with open(new_filename, 'a') as outfile:
                    with open(filename) as infile:
                        next(infile)    # skip header
                        for line in infile:
                            outfile.write(line)
        print('Done')

    def find_start(self, data_file):
        start = 0
        for title in self.lengths:
            if title in data_file:
                return start
            start += self.lengths[title]
        return 0

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


if __name__ == '__main__':
    if not os.path.exists(LABEL_ANNOTATION_FILENAME):
        annotate()

    relabeler = Relabeler(LABEL_ANNOTATION_FILENAME)
    input_files = sorted(glob.glob(TRACKS_PATH))
    video_files = sorted(glob.glob(VIDEOS_PATH))
    relabeler.process_all(input_files, video_files)
    print('Creating annotated video')
    annotate_videos(video_files, VIDEOS_OUTPUT, relabeler.datas, outratio=0.01)
