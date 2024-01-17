import cv2 as cv
from imageio.v3 import imwrite
import math
import numpy as np
import os
from tqdm import tqdm

from src.file.plain_csv import export_csv_simple
from src.util import ensure_out_path


class PathLink:
    def __init__(self, label, position1, position2, created):
        self.label = label
        self.position1 = position1
        self.position2 = position2
        self.created = created
        self.total_use = 0
        self.used = False

    def update_use(self, time):
        self.total_use += time
        if not self.used:
            self.used = True

    def draw(self, image, position1, position2, time):
        power = 3
        scale = self.total_use / (time + 1)
        if scale > 0:
            col_scale = 1 + (math.log10(scale) - 2) / power   # log: 1(E0) ... 1E-[power]
        else:
            col_scale = 0
        col_scale = np.clip(col_scale, 0, 1)
        color_value = [int(np.round(col_scale * 255))]
        cv.line(image, position1, position2, color_value, 1, cv.LINE_AA)

    def to_dict(self):
        return {'x1': self.position1[0], 'y1': self.position1[1],
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

    def run(self, datas, features, params, general_params):
        out_features = []
        self.datas = datas
        self.features = features
        self.image_size = general_params['image_size']
        self.node_distance = params['node_distance']
        node_distance = self.node_distance
        base_dir = general_params['base_dir']
        self.output = os.path.join(base_dir, params['output'])
        ensure_out_path(self.output)
        self.image_output = os.path.join(base_dir, params['image_output'])
        ensure_out_path(self.image_output)
        self.video_output = os.path.join(base_dir, params['video_output'])
        ensure_out_path(self.video_output)
        self.video_output_fps = params['video_output_fps']
        self.frame_interval = params['frame_interval']

        map_size = np.divide(self.image_size, node_distance).astype(int) + 1
        self.map = np.zeros(np.flip(map_size), dtype=np.float32)

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
                if frame in data.frames:
                    position = data.position[frame]
                    map_position = np.round(np.divide(position, node_distance)).astype(int)
                    if tracki in last_track_map_position and not np.all(map_position == last_track_map_position[tracki]):
                        self.map[map_position[1], map_position[0]] += framei
                        last_position = last_track_map_position[tracki]
                        self.update_link(last_position, map_position, framei)
                    last_track_map_position[tracki] = map_position

            if framei % self.frame_interval == 0:
                self.save(framei)
                self.draw(framei)

        if self.vidwriter is not None:
            self.vidwriter.release()

        return out_features

    def update_link(self, last_position, position, time):
        key = str(last_position[0]) + ',' + str(last_position[1]) + '-' + str(position[0]) + ',' + str(position[1])
        link = self.links.get(key)
        if key not in self.links:
            self.links[key] = PathLink(self.next_label, last_position, position, time)
            self.next_label += 1
        else:
            link.update_use(time)

    def save(self, framei):
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
            #for link in self.links:
            #    if link.used:
            #       link.draw(image, framei)

            power = 3
            image0 = self.map / (framei + 1)
            image = 1 + (np.log10(image0, where=image0 > 0) - 2) / power  # log: 1(E0) ... 1E-[power]
            image[image0 == 0] = 0
            image = np.round(np.clip(image, 0, 1) * 255).astype(np.uint8)
            image = cv.applyColorMap(image, cv.COLORMAP_HOT)

            if self.image_output is not None:
                image_filename = self.image_output.format(frame=framei)
                rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                imwrite(image_filename, rgb_image)
            if self.video_output is not None:
                if self.vidwriter is None:
                    self.vidwriter = cv.VideoWriter(self.video_output, -1, self.video_output_fps, self.image_size)
                self.vidwriter.write(image)
