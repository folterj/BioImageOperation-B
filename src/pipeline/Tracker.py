import inspect
import os
import cv2 as cv
import numpy as np

from src.segmentation import *
from src.util import *


class Tracker:
    def __init__(self, params, base_dir, input_files, output, video_output):
        self.input_files = input_files
        self.output = output
        self.video_output = video_output
        self.frame_interval = params.get('frame_interval', 1)
        self.operations = params.get('operations')

        if 'background' in params:
            self.background = float_image(grayscale_image(imread(os.path.join(base_dir, params['background']))))
        else:
            self.background = None

        if 'mask' in params:
            self.mask = float_image(grayscale_image(imread(os.path.join(base_dir, params['mask']))))
        else:
            self.mask = None

        self.texture_filters = []

    def track(self, input_files=None):
        # TODO: support multiple (sequential) videos files
        # TODO: support image sequence
        if input_files is None:
            input_files = self.input_files

        for input_file in input_files:
            vidcap = cv.VideoCapture(input_file)
            framei = 0
            ok = vidcap.isOpened()
            while ok:
                ok, image = vidcap.read()
                if ok:
                    if framei % self.frame_interval == 0:
                        self.track_frame(image)
                ok = vidcap.isOpened()
                framei += 1
            vidcap.release()

    def track_frame(self, image):
        image = float_image(image)
        original_image = image
        for operation0 in self.operations:
            operation = operation0.rstrip(')').split('(')
            params = []
            if len(operation) > 1:
                for param in operation[1].split(','):
                    value = param.strip()
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                    params.append(value)
            operation = operation[0]
            if operation == 'threshold':
                original_image = image
            if hasattr(self, operation):
                function = getattr(self, operation)
            elif operation in globals():
                function = globals()[operation]
            else:
                raise Exception("Unknown operation: " + operation)

            sig = inspect.signature(function)
            for param in sig.parameters.values():
                if param.name == 'original_image':
                    params.append(original_image)
            image = function(image, *params)

        pass

    def subtract_background(self, image):
        return np.abs(image - self.background)

    def apply_mask(self, image):
        return image * self.mask

    def init_texture_detection(self):
        # This function is designed to produce a set of GaborFilters
        # an even distribution of theta values equally distributed amongst pi rad / 180 degree
        filters = []
        num_filters = 16
        ksize = 35  # The local area to evaluate
        sigma = 3.0  # Larger Values produce more edges
        lambd = 10.0
        gamma = 0.5
        psi = 0  # Offset value - lower generates cleaner results
        for theta in np.arange(0, np.pi, np.pi / num_filters):  # Theta is the orientation for edge detection
            kern = cv.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv.CV_64F)
            kern /= 1.0 * kern.sum()  # Brightness normalization
            filters.append(kern)
        self.texture_filters = filters

    def texture_detection(self, image):
        if not self.texture_filters:
            self.init_texture_detection()
        # apply gabor filer and edge detection
        return edge_detection(gabor_filtering(image, self.texture_filters))
