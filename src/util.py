import glob
import math
import os
import re

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
    return numeric_string_sort(glob.glob(input_path))


def filter_output_files(filenames, all_params):
    base_dir = all_params['general']['base_dir']
    for operation0 in all_params['operations']:
        operation = operation0[next(iter(operation0))]
        if 'video_output' in operation:
            filename = os.path.join(base_dir, operation['video_output'])
            filenames.remove(filename)
    return filenames


def extract_filename_id_info(filename):
    filetitle = get_filetitle(filename)
    parts = filetitle.split('_')
    id = parts[-1]
    info = parts[:-1]

    # find id with non-numeric prefix
    if not id[0].isnumeric():
        if id[-1].isnumeric():
            pos = len(id) - 1
            while id[pos].isnumeric() and pos > 0:
                pos -= 1
            pos += 1
            id0 = id
            id = id0[pos:]
            info += [id0[:pos]]
        else:
            id = ''
            info = parts

    id_info = [id] + info
    return id_info


def find_all_filename_infos(filenames):
    ids = set()
    infos = set()
    for filename in filenames:
        id_info = extract_filename_id_info(filename)
        ids.add(id_info[0])
        infos.add('_'.join(id_info[1:]))
    ids = numeric_string_sort(list(ids))
    infos = numeric_string_sort(list(infos))
    infos = [info.split('_') for info in infos]
    return infos, ids


def get_input_stats(input_files):
    s = ''
    infos, ids = find_all_filename_infos(input_files)
    s += f'#unique video ids: {len(infos)}\n'
    s += f'#unique track ids: {len(ids)}\n'
    for info in infos:
        n = 0
        for input_file in input_files:
            if extract_filename_id_info(input_file)[1:] == info:
                n += 1
        s += f'{info[0:-1]} - #tracks:\t{n}\n'

    for id in ids:
        n = 0
        for input_file in input_files:
            if extract_filename_id_info(input_file)[0] == id:
                n += 1
        s += f'Track id {id} - #videos:\t{n}\n'
    return s


def numeric_string_sort(items):
    return sorted(items, key=lambda item: list(map(int, re.findall(r'\d+', item))))
