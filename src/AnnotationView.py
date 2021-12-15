import cv2 as cv
import numpy as np
import tkinter as tk
from tkinter import simpledialog

from src.file.annotations import load_annotations, save_annotations
from src.parameters import *


class AnnotationView(object):
    def __init__(self, image, annotation_filename, maxdist=MAX_ANNOTATION_DISTANCE, window_name='CvView'):
        self.image = image
        self.annotation_filename = annotation_filename
        self.maxdist = maxdist
        self.window_name = window_name
        self.annotations = load_annotations(self.annotation_filename)

        self.tk_root = tk.Tk()
        self.tk_root.overrideredirect(1)
        self.tk_root.withdraw()

        cv.namedWindow(self.window_name, cv.WINDOW_NORMAL)
        cv.setMouseCallback(self.window_name, self.on_mouse)
        self.redraw()

    def redraw(self):
        self.view_image = self.image.copy()
        for annotation in self.annotations:
            self.draw_annotation(annotation)
        self.show()

    def draw_annotation(self, annotation):
        posx, posy, label = annotation
        position = (posx, posy)
        fontface = cv.FONT_HERSHEY_SIMPLEX
        scale = 1
        thickness = 1
        color = (255, 255, 0)
        size = cv.getTextSize(label, fontface, scale, thickness)
        cv.putText(self.view_image, label, position, fontface, scale, color, thickness, cv.LINE_AA)
        cv.drawMarker(self.view_image, position, color)
        return size

    def show(self):
        cv.imshow(self.window_name, self.view_image)
        cv.namedWindow(self.window_name, cv.WINDOW_KEEPRATIO)
        _, _, w, h = cv.getWindowImageRect(self.window_name)
        ih = self.view_image.shape[0]
        iw = self.view_image.shape[1]
        wr = w / iw
        hr = h / ih
        r = min(wr, hr)
        cv.resizeWindow(self.window_name, int(round(iw * r)), int(round(ih * r)))

    def show_loop(self):
        self.show()
        do = True
        while do:
            # the OpenCV window won't display until you call cv2.waitKey()
            key = cv.waitKey(5)
            do = self.key_press(key)

    def close(self):
        cv.destroyAllWindows()

    def on_mouse(self, event, x, y, flags, params):
        if event == cv.EVENT_LBUTTONUP:
            # add
            label = simpledialog.askstring('Annotation', 'Label')
            if label is not None:
                self.annotations.append([x, y, label])
                self.save()
                self.redraw()
        elif event == cv.EVENT_RBUTTONUP:
            # remove
            for annotation in self.annotations:
                dist = np.sqrt((annotation[0] - x) ** 2 + (annotation[1] - y) ** 2)
                if dist < self.maxdist:
                    self.annotations.remove(annotation)
                    self.save()
                    self.redraw()

    def save(self):
        save_annotations(self.annotation_filename, self.annotations)

    def key_press(self, key):
        if key == ord('s'):
            self.save()
        return key != ord('q') and key != 27
