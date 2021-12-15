import glob
import os.path
import shutil
import cv2 as cv
import numpy as np

from src.AnnotationView import AnnotationView
from src.file.annotations import load_annotations
from src.file.bio import import_tracks_by_frame
from src.parameters import *


def annotate():
    image = cv.imread('D:/Video/Living_Earth/Spider Activity/cam1mp4/back mod.png')
    annotator = AnnotationView(image, LABEL_ANNOTATION_FILENAME)
    annotator.show_loop()
    annotator.close()


class Relabeler():
    def __init__(self, annotation_filename):
        self.annotations = load_annotations(annotation_filename)

    def process_all(self, filenames):
        for filename in filenames:
            data = import_tracks_by_frame(filename)
            old_title = os.path.splitext(os.path.basename(filename))[0]
            if 'track_label' in data:
                old_label = str(data['track_label'][0])
            else:
                old_label = old_title.rsplit('_')[-1]


    def get_best_label_from_file(self, filename):
        data = import_tracks_by_frame(filename)
        old_title = os.path.splitext(os.path.basename(filename))[0]
        if 'track_label' in data:
            old_label = str(data['track_label'][0])
        else:
            old_label = old_title.rsplit('_')[-1]
        new_label = self.get_best_label(data)
        new_title = old_title
        if new_title.endswith(old_label):
            new_title = new_title.rstrip(old_label)
        new_title += new_label
        return new_label, new_title, old_label, old_title

    def get_best_label(self, data):
        meanx = np.mean(list(data['x'].values()))
        meany = np.mean(list(data['y'].values()))
        mindist = -1
        label = None
        for annotation in self.annotations:
            dist = np.sqrt((annotation[0] - meanx) ** 2 + (annotation[1] - meany) ** 2)
            if dist < mindist or mindist < 0:
                mindist = dist
                label = annotation[2]
        if mindist > MAX_MOVE_DISTANCE:
            # closest is too far away; disqualify
            return None
        return label


if __name__ == '__main__':
    #annotate()

    relabeler = Relabeler(LABEL_ANNOTATION_FILENAME)
    input_files = sorted(glob.glob(TRACKS_PATH))
    #for input_file in input_files:
    #    new_label, new_title, old_label, old_title = relabeler.get_best_label_from_file(input_file)
    #    print(f'{old_title} -> {new_title}')
    #    extension = os.path.splitext(input_file)[1]
    #    shutil.copy(input_file, os.path.join(TRACKS_RELABEL_PATH, new_title + extension))
    relabeler.process_all(input_files)

