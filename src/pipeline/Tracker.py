import inspect
import os
import cv2 as cv
import numpy as np

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
        return self.edge_detection(self.gabor_filtering(image))

    def gabor_filtering(self, image):
        # This general function is designed to apply filters to our image
        # First create a numpy array the same size as our input image
        new_image = np.zeros_like(image)
        # Starting with a blank image, we loop through the images and apply our Gabor Filter
        # On each iteration, we take the highest value (super impose), until we have the max value across all filters
        # The final image is returned
        depth = -1  # remain depth same as original image
        for kern in self.texture_filters:  # Loop through the kernels in our GaborFilter
            image_filter = cv.filter2D(image, depth, kern)  # Apply filter to image
            # Using Numpy.maximum to compare our filter and cumulative image, taking the higher value (max)
            np.maximum(new_image, image_filter, new_image)
        return new_image

    def edge_detection(self, image, min_interval=120, max_interval=250):
        return cv.Canny(int_image(image), min_interval, max_interval)

    def segment(self, image, original_image=None):
        #labels1 = dist_watershed_image(image)
        #labels2 = dist_watershed_image2(image)
        contours = get_contours(image)
        min_area = 1
        areas = [get_area(contour) for contour in contours if get_area(contour) > min_area]
        min_area = np.mean(areas) / 2
        areas2 = [get_area(contour) for contour in contours if get_area(contour) > min_area]
        mean_area = np.median(areas2)
        min_area = mean_area / 2
        contours2 = [contour for contour in contours if get_area(contour) > min_area]
        shapes = [get_lengths(contour) for contour in contours2]
        mean_shape = np.median(shapes, 0) / 2

        contours3 = []
        for contour in contours2:
            area = get_area(contour)
            n = int(np.round(area / mean_area))
            if n > 1:
                min_distance = mean_shape[1]
                contours3.extend(self.split_contour_gamma(contour, n, min_distance, original_image))
            else:
                contours3.append(contour)

        annotated_image = color_image(original_image).copy()
        cv.drawContours(annotated_image, contours3, -1, 1, thickness=2)
        show_image(annotated_image)

    def split_contour_gamma(self, contour, n, min_distance, image):
        # TODO: store series of (hierarchical?) contours to be combined with tracking
        final_contours = [contour]
        cropped = grayscale_image(extract_image(image, contour))
        contour_offset = np.min(contour, 0).astype(int)

        thresholds = np.arange(0, 1, 0.05)
        for thres in thresholds:
            contours = get_contours(threshold(cropped, thres))
            if len(contours) >= n:
                final_contours = [contour + contour_offset for contour in contours]
                break

        return final_contours

    def split_contour(self, contour, n, min_distance, image):
        contours = []
        cropped = extract_image(image, contour)
        cropped_shape = cropped.shape[:2]
        contour_offset = np.min(contour, 0).astype(int)
        contour1 = contour - contour_offset

        mask = get_contour_mask(contour1, shape=cropped_shape)
        dist = cv.distanceTransform(int_image(mask), cv.DIST_L2, cv.DIST_MASK_PRECISE)
        local_maxs = peak_local_max(dist, labels=int_image(mask), num_peaks=n, min_distance=int(min_distance))

        if len(local_maxs) < n:
            masks = self.create_divided_image_masks_moments(contour1, cropped_shape, n)
        else:
            markers = np.zeros_like(dist)
            for point in local_maxs:
                markers[tuple(point)] = 1

            markers = label(markers, structure=np.ones((3, 3)))[0]
            labels = watershed(-dist, markers, mask=mask)

            masks = []
            for i in range(n):
                label_mask = np.zeros_like(mask)
                label_mask[labels == (i + 1)] = 1
                masks.append(label_mask)

        for mask in masks:
            masked_image = int_image(mask * cropped[..., -1])
            mask_contours = get_contours(masked_image)
            if len(mask_contours) > 0:
                areas = [cv.contourArea(mask_contour) for mask_contour in mask_contours]
                best_contouri = np.argmax(areas)
                best_area = areas[best_contouri]
                if best_area > 0:
                    best_contour = mask_contours[best_contouri]
                    contours.append(best_contour + contour_offset)

        return contours

    def create_divided_image_masks_moments(self, contour, image_shape, n):
        masks = []
        rotated_rect = cv.minAreaRect(contour)
        center0, size0, angle = rotated_rect
        length_dim = 0  # fixed in X direction
        if np.argmax(size0) == 0:
            size0 = np.flip(size0)
            angle += 90
        move_size = size0[length_dim] / n
        size = np.array(size0)
        size[length_dim] = move_size

        angle_rad = np.deg2rad(angle)
        length = np.array((np.cos(angle_rad), np.sin(angle_rad))) * size0[length_dim]
        start_point = np.array(center0) - length / 2

        step = length / n
        for _ in range(n):
            center = start_point + step / 2
            box = cv.boxPoints((center, size, angle))
            mask = get_contour_mask(box, shape=image_shape)
            masks.append(mask)
            start_point += step
        return masks
