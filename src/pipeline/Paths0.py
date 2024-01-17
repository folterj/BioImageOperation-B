import cv2 as cv
from imageio.v3 import imwrite
import math
import numpy as np
import os
from scipy.spatial.distance import cdist
from sklearn.neighbors import KDTree
from tqdm import tqdm

from src.file.plain_csv import export_csv_simple
from src.util import ensure_out_path


class PathNode:
    def __init__(self, label, position, time):
        self.label = label
        self.position = position
        self.created = time
        self.total_use = 0

    def update_use(self, time):
        self.total_use += time

    def draw(self, image, time):
        power = 3
        scale = self.total_use / (time + 1)
        if scale > 0:
            col_scale = 1 + (math.log10(scale) - 2) / power   # log: 1(E0) ... 1E-[power]
        else:
            col_scale = 0
        col_scale = np.clip(col_scale, 0, 1)
        color_value = [int(np.round(col_scale * 255))]
        position = np.round(self.position).astype(int)
        cv.drawMarker(image, position, color_value, cv.MARKER_CROSS, 2, 1)

    def to_dict(self):
        return {'label': self.label, 'x': self.position[0], 'y': self.position[1],
                'created': self.created, 'total_use': self.total_use}

    def __str__(self):
        return str(self.to_dict())


class PathLink:
    def __init__(self, node1, node2):
        self.node1 = node1
        self.node2 = node2

    def draw(self, image):
        color = (127, 127, 127)
        position1 = np.round(self.node1.position).astype(int)
        position2 = np.round(self.node2.position).astype(int)
        cv.line(image, position1, position2, color, 1, cv.LINE_AA)

    def __str__(self):
        return str(self.node1) + ' - ' + str(self.node2)


class Paths:
    def __init__(self):
        self.nodes = []
        self.node_positions = []
        self.links = []
        self.next_label = 0
        self.vidwriter = None

    def run(self, datas, features, params, general_params):
        out_features = []
        self.datas = datas
        self.features = features
        self.image_size = general_params['image_size']
        self.node_distance = params['node_distance']
        base_dir = general_params['base_dir']
        self.output = os.path.join(base_dir, params['output'])
        ensure_out_path(self.output)
        self.image_output = os.path.join(base_dir, params['image_output'])
        ensure_out_path(self.image_output)
        self.video_output = os.path.join(base_dir, params['video_output'])
        ensure_out_path(self.video_output)
        self.video_output_fps = params['video_output_fps']
        self.frame_interval = params['frame_interval']

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

        last_track_nodes = {}
        for framei, frame in enumerate(tqdm(all_frames0)):
            self.distance_pre_calc()
            for tracki, data in enumerate(datas):
                if frame in data.frames:
                    position = data.position[frame]
                    last_node = last_track_nodes.get(tracki)
                    node = self.get_node(position, framei, last_node)
                    last_track_nodes[tracki] = node
            if framei % self.frame_interval == 0:
                self.save(framei)
                self.draw(framei)

        if self.vidwriter is not None:
            self.vidwriter.release()

        return out_features

    def get_node(self, position, time, last_node):
        node = None
        node_distance = self.node_distance

        if last_node is not None:
            distance = math.dist(last_node.position, position)
            if distance < node_distance / 2:
                node = last_node

        if node is None:
            closest_node, distance = self.find_closest_node(position)
            if closest_node is not None and distance < node_distance:
                node = closest_node
                node.update_use(time)

        if node is None:
            node = self.create_node(position, time, last_node)
            node.update_use(time)
        return node

    def distance_pre_calc(self):
        if self.node_positions:
            self.position_tree = KDTree(self.node_positions)

    def find_closest_node(self, position):
        if self.node_positions:
            #distance, index = calc_distance_cdist(position, self.node_positions)
            distance, index = self.position_tree.query([position])
            distance = distance[0][0]
            index = index[0][0]
            return self.nodes[index], distance
        else:
            return None, None

    def create_node(self, position, time, linked_node=None):
        label = self.next_label
        self.next_label += 1
        node = PathNode(label, position, time)
        self.node_positions.append(position)
        self.nodes.append(node)
        if linked_node is not None:
            self.links.append(PathLink(linked_node, node))
        return node

    def save(self, framei):
        filename = self.output.format(frame=framei)
        output_data = {}
        for node in self.nodes:
            for key, value in node.to_dict().items():
                if key not in output_data:
                    output_data[key] = []
                output_data[key].append(value)
        export_csv_simple(filename, output_data)

    def draw(self, framei):
        if self.image_output is not None or self.video_output is not None:
            shape = list(np.flip(self.image_size))
            image = np.zeros(shape, dtype=np.uint8)
            for node in self.nodes:
                node.draw(image, framei)
            #for link in self.links:
            #    link.draw(image)
            image = cv.applyColorMap(image, cv.COLORMAP_HOT)
            if self.image_output is not None:
                image_filename = self.image_output.format(frame=framei)
                rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                imwrite(image_filename, rgb_image)
            if self.video_output is not None:
                if self.vidwriter is None:
                    self.vidwriter = cv.VideoWriter(self.video_output, -1, self.video_output_fps, self.image_size)
                self.vidwriter.write(image)


def calc_distance_cdist(target, references):
    distances = cdist([target], references)
    index = distances.argmin()
    distance = distances[0][index]
    return distance, index
