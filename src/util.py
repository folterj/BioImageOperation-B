from collections import deque
import colorsys
import cv2 as cv
import glob
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from scipy.ndimage import uniform_filter1d, distance_transform_edt, label
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


mpl.rcParams['figure.dpi'] = 600


def list_to_str(lst):
    return [str(x) for x in lst]


def pairwise(iterable):
    a = iter(iterable)
    return zip(a, a)


def isvalid_position(position):
    return position[0] >= 0 and position[1] >= 0 and np.isfinite(position[0]) and np.isfinite(position[1])


def calc_dist(pos0, pos1=(0, 0)):
    return math.dist(pos0, pos1)


def calc_mean_dist(positions0, positions1):
    distances = []
    for frame in positions0:
        if frame in positions1:
            distances.append(calc_dist(positions0[frame], positions1[frame]))
    return np.mean(distances)


def imread(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(filename)
    return cv.imread(filename)


def grayscale_image(image):
    nchannels = image.shape[2] if image.ndim > 2 else 1
    if nchannels == 4:
        return cv.cvtColor(image, cv.COLOR_RGBA2GRAY)
    elif nchannels > 1:
        return cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    else:
        return image


def color_image(image):
    nchannels = image.shape[2] if len(image.shape) > 2 else 1
    if nchannels == 1:
        return cv.cvtColor(image, cv.COLOR_GRAY2RGB)
    else:
        return image


def float_image(image):
    if image.dtype.kind != 'f':
        maxval = 2 ** (8 * image.dtype.itemsize) - 1
        return image / np.float32(maxval)
    else:
        return image


def int_image(image, dtype=np.dtype(np.uint8)):
    if not (image.dtype.kind == 'i' or image.dtype.kind == 'u') and not dtype.kind == 'f':
        maxval = 2 ** (8 * dtype.itemsize) - 1
        return (image * maxval).astype(dtype)
    else:
        return image


def show_image(image, title='', cmap=None):
    nchannels = image.shape[2] if image.ndim > 2 else 1
    if cmap is None:
        cmap = 'gray' if nchannels == 1 else None
    plt.imshow(image, cmap=cmap)
    if title != '':
        plt.title(title)
    plt.show()


def get_area(data):
    return cv.contourArea(data)


def get_image_moments(image):
    moments = cv.moments(image.astype(np.float32))
    return moments


def get_moments(data):
    moments = cv.moments((np.array(data)).astype(np.float32))    # doesn't work for float64!
    return moments


def get_moments_area(moments):
    return moments['m00']


def get_moments_centre(moments):
    x = moments['m10'] / moments['m00']
    y = moments['m01'] / moments['m00']
    return x, y


def get_lengths(data):
    moments = get_moments(data)
    if moments['m00'] != 0:
        lengths = get_moments_lengths(moments)
    elif len(data) > 1:
        lengths = (math.dist(np.max(data, 0), np.min(data, 0)), 0)
    else:
        lengths = (0, 0)
    return lengths


def get_moments_lengths(moments):
    # https://stackoverflow.com/questions/66309123/find-enclosing-rectangle-of-image-object-using-second-moments
    mu11 = moments['mu11'] / moments['m00']
    mu20 = moments['mu20'] / moments['m00']
    mu02 = moments['mu02'] / moments['m00']
    mu = [[mu02, mu11], [mu11, mu20]]
    l0, _ = np.linalg.eig(mu)
    l = np.array([x for x in l0 if x > 0])
    lengths = np.sqrt(12 * l)
    return tuple(sorted(lengths, reverse=True))


def get_moments_angle(moments):
    # returns value between -90 and 90
    return np.rad2deg(0.5 * math.atan2(2 * moments['mu11'], moments['mu20'] - moments['mu02']))


def round_significants(a, significant_digits):
    round_decimals = significant_digits - int(np.floor(np.log10(abs(a)))) - 1
    return round(a, round_decimals)


def calc_diff(source, multiplier=1):
    dest = {}
    last_value = None
    for frame, value in source.items():
        if last_value is not None:
            dest[frame] = (value - last_value) * multiplier
        last_value = value
    return dest


def create_window0(frames, source, window_size):
    # precise, but much slower
    dest = {}
    frames_window = deque(maxlen=window_size)
    window = deque(maxlen=window_size)
    mean_index = window_size // 2
    for frame in frames:
        frames_window.append(frame)
        window.append(source.get(frame, np.nan))
        if mean_index < len(frames_window):
            mean_frame = frames_window[window_size // 2]
            dest[mean_frame] = np.nanmean(window)
    return dest


def create_window(frames, source_dict, window_size):
    source = [float(source_dict.get(frame, 0)) for frame in frames]
    dest = uniform_filter1d(source, window_size, mode='nearest')
    dest_dict = {frame: data for frame, data in zip(frames, dest) if frame in source_dict}
    return dest_dict


def extract_image(image, polygon):
    polygon_min, polygon_max = np.min(polygon, 0).astype(int), np.ceil(np.max(polygon, 0)).astype(int)
    x0, y0 = polygon_min
    x1, y1 = polygon_max
    w, h = x1 - x0, y1 - y0
    polygon1 = polygon - polygon_min
    cropped = image[y0:y1, x0:x1]
    mask = get_contour_mask(polygon1, shape=(h, w))
    image = color_image(cropped) * np.atleast_3d(mask)
    alpha_image = np.dstack([image, mask])
    return alpha_image


def get_contours(image):
    contours0 = cv.findContours(int_image(image), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = contours0[0] if len(contours0) == 2 else contours0[1]
    contours_squeezed = [contour[:, 0, :] for contour in contours]
    return contours_squeezed


def get_contour_mask(contour, shape):
    image = np.zeros(shape, dtype=np.float32)
    mask = cv.drawContours(image, [np.round(contour).astype(int)], -1, 1, thickness=cv.FILLED)
    return mask


def invert(image):
    return 1 - image


def threshold(image, thres):
    return cv.threshold(image, thres, 1, cv.THRESH_BINARY)[1]


def create_kernel(size):
    kernel_size = int(2 * size + 1)
    return cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))


def erode(image, size):
    return cv.erode(image, create_kernel(size))


def dilate(image, size):
    return cv.dilate(image, create_kernel(size))


def dist_watershed_image(image):
    # https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_watershed/py_watershed.html
    # https://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Watershed_Algorithm_Marker_Based_Segmentation_2.php
    kernel = create_kernel(1)
    opening = cv.morphologyEx(image, cv.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv.dilate(opening, kernel, iterations=3)
    dist_transform = cv.distanceTransform(int_image(opening), cv.DIST_L2, cv.DIST_MASK_PRECISE)
    ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 1, 0)
    unknown = sure_bg - sure_fg
    ret, markers = cv.connectedComponents(int_image(sure_fg))
    markers += 1
    markers[unknown == 1] = 0
    markers = cv.watershed(color_image(int_image(image)), int_image(markers, dtype=np.uint32))
    return markers


def dist_watershed_image2(image):
    # https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_watershed.html
    distance = distance_transform_edt(image)
    coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=int_image(image))
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = label(mask)
    labels = watershed(-distance, markers, mask=image)
    return labels


def ensure_out_path(path):
    if not os.path.isdir(path):
        path = os.path.dirname(path)
    if not os.path.exists(path):
        os.makedirs(path)


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
            if filename in filenames:
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


def get_frames_number(value, fps):
    multipliers = [1, 60, 60, 24]
    nframes = value
    if isinstance(value, str):
        if ':' in value:
            nseconds = 0
            cummultiplier = 1
            for part, multiplier in zip(reversed(value.split(':')), multipliers):
                cummultiplier *= multiplier
                nseconds += int(part) * cummultiplier
            nframes = int(nseconds * fps)
        else:
            nframes = int(value)
    return nframes


def create_colormap(colors, length=256, dtype=np.uint8):
    colormap = np.zeros((length, 1, 3), dtype)     # opencv compatible colormap
    ncolors = len(colors) - 1
    single_range = int(np.round(length / ncolors))
    index = 0
    for colori in range(ncolors):
        color0, color1 = colors[colori], colors[colori + 1]
        for i in range(single_range):
            color = np.multiply(color0, 1 - i / single_range) + np.multiply(color1, i / single_range)
            colormap[index] = (color * 255).astype(dtype)
            index += 1
    return colormap


def create_color_table(n):
    colors = []
    h = 0
    l = 0.5
    for i in range(n):
        colors.append(normalize_lightness(colorsys.hsv_to_rgb(h, 1, 1), l))
        h = math.fmod(h + 251 / 360, 1)
        l -= 0.22
        if l < 0.2:
            l += 0.6
    return colors


def normalize_lightness(color, level):
    r, g, b = color
    level0 = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
    if level0 != 0:
        f = level / level0
        if f > 1:
            r = 1 - (1 - r) / f
            g = 1 - (1 - g) / f
            b = 1 - (1 - b) / f
        else:
            r *= f
            g *= f
            b *= f
    return r, g, b


def color_float_to_cv(rgb):
    return int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255)
