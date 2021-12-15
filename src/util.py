import math
import os

import numpy as np
import cv2 as cv


def pairwise(iterable):
    a = iter(iterable)
    return zip(a, a)


def calc_dist(pos0, pos1=(0, 0)):
    return np.sqrt((pos1[0] - pos0[0]) ** 2 + (pos1[1] - pos0[1]) ** 2)


def get_image_moments(image):
    moments = cv.moments(image.astype(np.float))
    return moments


def get_moments_area(moments):
    return moments['m00']


def get_moments_centre(moments):
    x = moments['m10'] / moments['m00']
    y = moments['m01'] / moments['m00']
    return x, y


def get_moments_angle(moments):
    # returns value between -90 and 90
    return np.rad2deg(0.5 * math.atan2(2 * moments['mu11'], moments['mu20'] - moments['mu02']))


def round_significants(a, significant_digits):
    round_decimals = significant_digits - int(np.floor(np.log10(abs(a)))) - 1
    return round(a, round_decimals)


def get_filetitle(filename):
    return os.path.splitext(os.path.basename(filename))[0]
