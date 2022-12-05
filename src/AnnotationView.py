import cv2 as cv
import tkinter as tk
from tkinter import simpledialog

from src.file.generic import import_file
from src.file.plain_csv import export_csv
from src.util import calc_dist


class AnnotationView(object):
    def __init__(self, image, annotation_filename, annotation_margin, window_name='CvView'):
        self.image = image
        self.annotation_filename = annotation_filename
        self.annotation_margin = annotation_margin
        self.window_name = window_name
        annotations = import_file(self.annotation_filename)
        self.annotations = {key: (value['x'][0], value['y'][0]) for key, value in annotations.items()}

        self.tk_root = tk.Tk()
        self.tk_root.overrideredirect(1)
        self.tk_root.withdraw()

        cv.namedWindow(self.window_name, cv.WINDOW_NORMAL)
        cv.setMouseCallback(self.window_name, self.on_mouse)
        self.redraw()

    def redraw(self):
        self.view_image = self.image.copy()
        for annotation in self.annotations.items():
            self.draw_annotation(annotation)
        self.show()

    def draw_annotation(self, annotation):
        label, data = annotation
        position = (int(data[0]), int(data[1]))
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
                self.annotations[label] = (x, y)
                self.save()
                self.redraw()
        elif event == cv.EVENT_RBUTTONUP:
            # remove
            min_dist = None
            min_annotation = None
            for annotation, value in self.annotations.items():
                dist = calc_dist((x, y), value)
                if min_dist is None or dist < min_dist:
                    min_dist = dist
                    min_annotation = annotation
            if min_annotation is not None and min_dist < self.annotation_margin:
                self.annotations.pop(min_annotation)
                self.save()
                self.redraw()

    def save(self):
        data = {id: {'x': {0: position[0]}, 'y': {0: position[1]}} for id, position in self.annotations.items()}
        export_csv(self.annotation_filename, data)

    def key_press(self, key):
        if key == ord('s'):
            self.save()
        return key != ord('q') and key != 27
