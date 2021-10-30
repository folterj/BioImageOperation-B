import glob
import os.path

import cv2 as cv
import numpy as np

from src.AnnotationView import AnnotationView
from src.file.annotations import load_annotations
from src.file.bio import import_tracks_by_frame
from src.parameters import *


def annotate():
    image = cv.imread('D:/Video/Living_Earth/Spider Activity/cam1mp4/back mod.png')
    annotator = AnnotationView(image, ANNOTATION_FILENAME)
    annotator.show_loop()
    annotator.close()


class Relabeler():
    def __init__(self, annotation_filename):
        self.annotations = load_annotations(annotation_filename)

    def get_label(self, data):
        meanx = np.mean(list(data['x'].values()))
        meany = np.mean(list(data['y'].values()))
        mindist = -1
        label = ''
        for annotation in self.annotations:
            dist = np.sqrt((annotation[0] - meanx) ** 2 + (annotation[1] - meany) ** 2)
            if dist < mindist or mindist < 0:
                mindist = dist
                label = annotation[2]
        if mindist > MAX_MOVE_DISTANCE:
            # closest is too far away; disqualify
            return ''
        return label


if __name__ == '__main__':
    #annotate()
    relabeler = Relabeler(ANNOTATION_FILENAME)
    # test:
    input_files = glob.glob(LIVING_EARTH_PATH + "track*.csv")
    for input_file in input_files:
        data = import_tracks_by_frame(input_file)
        label = relabeler.get_label(data)
        title = os.path.splitext(os.path.basename(input_file))[0]
        print(title, label)
