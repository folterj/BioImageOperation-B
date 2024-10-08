import cv2 as cv
from imageio.v3 import imwrite
import math
import numpy as np
import os
from scipy.spatial.distance import cdist
from tqdm import tqdm

from src.file.plain_csv import export_csv_simple
from src.util import ensure_out_path, create_colormap


class PathLink:
    def __init__(self, label, position1, position2, time):
        self.label = label
        self.position1 = position1
        self.position2 = position2
        self.created = time
        self.count = 0
        self.total_use = 0
        self.normal = 0
        self.reverse = 0
        self.used = False

    def update_use(self, time, reversed=False):
        self.count += 1
        self.total_use += time
        if not self.used:
            self.used = True
        if reversed:
            self.reverse += 1
        else:
            self.normal += 1

    def draw(self, image, time, method, power_scale, power_offset, scale=1):
        if 'direction' in method:
            max_value = max(self.normal, self.reverse)
            color_channel = np.clip((self.normal - self.reverse) / max_value, 0, 1)
            magnitude_channel = np.clip(max_value / (time + 1) * 10 ** power_scale, 0, 1)
            color_value = (color_channel, magnitude_channel)
        else:
            if 'time' in method:
                value = self.total_use
            else:
                value = self.count
            color_value = get_log_color_scale(value / (time + 1), power_scale, power_offset)
        position1 = np.round(self.position1 * scale).astype(int)
        position2 = np.round(self.position2 * scale).astype(int)
        cv.line(image, position1, position2, color_value, 1, cv.LINE_AA)

    def to_dict(self):
        return {'label': self.label, 'x1': self.position1[0], 'y1': self.position1[1],
                'x2': self.position2[0], 'y2': self.position2[1],
                'created': self.created, 'total_use': self.total_use}

    def __str__(self):
        return str(self.to_dict())


class Paths:
    def __init__(self):
        self.links = {}
        self.next_label = 0
        self.map = np.zeros((0, 0), dtype=np.float32)
        self.vidwriter = None
        self.colormap_blue_white_red = create_colormap([(0, 0, 1), (1, 1, 1), (1, 0, 0)])

    def run(self, datas, features, params, general_params):
        out_features = []
        self.datas = datas
        self.features = features
        base_dir = general_params['base_dir']
        self.image_size = general_params['image_size']
        self.node_scale = params['node_scale']
        self.method = params.get('method')
        self.draw_power_scale = params.get('draw_power_scale', 3)
        self.draw_power_offset = params.get('draw_power_offset', 0)
        self.output_size = params.get('output_size')

        output = params.get('output')
        if output:
            output = os.path.join(base_dir, output)
            ensure_out_path(output)
        self.output = output

        image_output = params.get('image_output')
        if image_output:
            image_output = os.path.join(base_dir, image_output)
            ensure_out_path(image_output)
        self.image_output = image_output

        raw_image_output = params.get('raw_image_output')
        if raw_image_output:
            raw_image_output = os.path.join(base_dir, raw_image_output)
            ensure_out_path(raw_image_output)
        self.raw_image_output = raw_image_output

        video_output = params.get('video_output')
        if video_output:
            video_output = os.path.join(base_dir, video_output)
            ensure_out_path(video_output)
        self.video_output = video_output

        self.video_output_fps = params.get('video_output_fps')
        self.frame_interval = params.get('frame_interval', 1)

        self.map_size = np.divide(self.image_size, self.node_scale).astype(int)
        self.path_image_size = self.map_size
        self.map = np.zeros(np.flip(self.map_size + 1), dtype=np.float32)

        all_frames0 = set()
        min_frame = None
        max_frame = None
        for data in datas:
            min_frame0 = min(data.frames)
            max_frame0 = max(data.frames)
            if min_frame is None:
                min_frame = min_frame0
            else:
                min_frame = min(min_frame0, min_frame)
            if max_frame is None:
                max_frame = max_frame0
            else:
                max_frame = min(max_frame0, max_frame)
            all_frames0.update(data.frames)
        all_frames0 = sorted(all_frames0)

        last_track_map_position = {}
        for framei, frame in enumerate(tqdm(all_frames0)):
            for tracki, data in enumerate(datas):
                if frame in data.frames and frame in data.position:
                    position = data.position[frame]
                    map_position = np.round(np.divide(position, self.node_scale)).astype(int)
                    if tracki in last_track_map_position and not np.all(map_position == last_track_map_position[tracki]):
                        if 'time' in self.method:
                            i = framei
                        else:
                            i = 1
                        self.map[map_position[1], map_position[0]] += i
                        last_position = last_track_map_position[tracki]
                        self.update_link(last_position, map_position, framei)
                    last_track_map_position[tracki] = map_position

            if framei % self.frame_interval == 0:
                self.save(framei)
                self.draw(framei)

        if self.vidwriter is not None:
            self.vidwriter.release()

        n = 0
        for link in self.links.values():
            if link.used:
                n += 1
        print(f'Final path links (used/total): {n}/{len(self.links)}')

        # TODO: set output features
        return out_features

    def update_link(self, last_position, position, time):
        key = f'{last_position}-{position}'
        link = self.links.get(key)
        reversed = False
        if link is None:
            key_reverse = f'{position}-{last_position}'
            link = self.links.get(key_reverse)
            if link is not None:
                reversed = True
        if link is None:
            self.links[key] = PathLink(self.next_label, last_position, position, time)
            self.next_label += 1
        else:
            link.update_use(time, reversed=reversed)

    def save(self, framei):
        if self.output:
            filename = self.output.format(frame=framei)
            output_data = {}
            for link in self.links.values():
                if link.used:
                    for key, value in link.to_dict().items():
                        if key not in output_data:
                            output_data[key] = []
                        output_data[key].append(value)
            export_csv_simple(filename, output_data)

    def draw(self, framei):
        if self.image_output is not None or self.video_output is not None:
            if 'path' in self.method:
                image = self.draw_paths(framei)
                raw_image = None
            else:
                raw_image, image = self.draw_map(framei)

            if self.image_output is not None:
                image_filename = self.image_output.format(frame=framei)
                rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                imwrite(image_filename, rgb_image)
            if self.raw_image_output is not None and raw_image is not None:
                image_filename = self.raw_image_output.format(frame=framei)
                imwrite(image_filename, raw_image)
            if self.video_output is not None:
                if self.vidwriter is None:
                    image_size = np.flip(image.shape[:2])
                    self.vidwriter = cv.VideoWriter(self.video_output, -1, self.video_output_fps, image_size)
                self.vidwriter.write(image)

    def draw_paths(self, framei):
        if self.output_size is not None:
            size = self.output_size
            scale = np.divide(size, self.path_image_size)
        else:
            size = self.path_image_size
            scale = 1
        shape = list(reversed(size))
        dtype = np.uint8
        if 'direction' in self.method:
            shape += [2]
            dtype = np.float32
        image = np.zeros(shape, dtype=dtype)
        for link in self.links.values():
            if link.used:
                link.draw(image, framei, self.method, self.draw_power_scale, self.draw_power_offset, scale=scale)
        if 'direction' in self.method:
            color_channel = (image[..., 0] * 255).astype(np.uint8)
            magnitude_channel = image[..., -1]
            color_image = cv.LUT(cv.cvtColor(color_channel, cv.COLOR_GRAY2BGR), self.colormap_blue_white_red)
            color_image = (color_image * np.atleast_3d(magnitude_channel)).astype(np.uint8)
        else:
            color_image = cv.applyColorMap(image, cv.COLORMAP_HOT)
        return color_image

    def draw_map(self, framei):
        image0 = self.map[0:self.map_size[1], 0:self.map_size[0]] / (framei + 1)
        image = 1 + (np.log10(image0, where=image0 > 0) + self.draw_power_offset) / self.draw_power_scale  # log: 1(E0) ... 1E-[power]
        image[image0 == 0] = 0
        image = np.round(np.clip(image, 0, 1) * 255).astype(np.uint8)
        color_image = cv.applyColorMap(image, cv.COLORMAP_HOT)
        return image0, color_image


def calc_distance_cdist(target, references):
    distances = cdist([target], references)
    index = distances.argmin()
    distance = distances[0][index]
    return distance, index


def get_log_color_scale(scale, power_scale=3, power_offset=0):
    if scale > 0:
        col_scale = 1 + (math.log10(scale) + power_offset) / power_scale  # log: 1(E0) ... 1E-[power]
    else:
        col_scale = 0
    col_scale = np.clip(col_scale, 0, 1)
    color_value = [int(np.round(col_scale * 255))]
    return color_value
