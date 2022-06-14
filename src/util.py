import glob
import math
import os

import numpy as np
import cv2 as cv


def list_to_str(lst):
    return [str(x) for x in lst]


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


def get_filetitle_replace(filename):
    return get_filetitle(filename).replace('.', '_')


def get_bio_base_name(filename):
    return os.path.basename(filename).rsplit('_', 1)[0]


def get_input_files(general_params, params, input_name):
    base_dir = general_params['base_dir']
    if input_name in params:
        input_path = params[input_name]
    else:
        input_path = general_params[input_name]
    input_path = os.path.join(base_dir, input_path)
    if os.path.isdir(input_path):
        input_path = os.path.join(input_path, '*')
    return sorted(glob.glob(input_path))


def extract_filename_info(filename):
    filetitle = get_filetitle(filename)
    parts = filetitle.split('_')
    id = parts[-1]
    date = 0
    time = 0
    camera = 0

    i = 0
    if not parts[i][0].isnumeric():
        i += 1
    if len(parts) > 2:
        date = parts[i]
        time = parts[i + 1].replace('-', ':')

    s = filename.lower().find('cam')
    if s >= 0:
        while s < len(filename) and not filename[s].isnumeric():
            s += 1
        e = s
        while e < len(filename) and filename[e].isnumeric():
            e += 1
        if e > s:
            camera = filename[s:e]

    info = [id, date, time, camera]
    return info


def find_all_filename_infos(filenames):
    ids = set()
    infos = set()
    for filename in filenames:
        info0 = extract_filename_info(filename)
        ids.add(info0[0])
        infos.add('_'.join(info0[1:]))
    ids = sorted(list(ids))
    infos = sorted(list(infos))
    infos = [info.split('_') for info in infos]
    return infos, ids
