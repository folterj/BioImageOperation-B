import math
import numpy as np


class PathNode:
    def __init__(self, position):
        self.position = position
        self.usage = []
        self.age = 0
        self.accum_usage = 1
        self.last_use = 1

    def update_use(self, path_age):
        self.usage.append(path_age)
        self.accum_usage += 1
        self.last_use = 1


class PathLink:
    def __init__(self, node1, node2):
        self.node1 = node1
        self.node2 = node2


class Paths:
    def __init__(self):
        self.nodes = []
        self.links = []

    def create(self, datas, fps, node_distance):
        self.node_distance = node_distance

        last_path_nodes = {}

        # get frames with 1s interval
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
        all_frames = np.round(np.arange(min_frame, max_frame, fps)).astype(int)

        path_age = 0
        last_frame = None
        for frame in all_frames:
            for tracki, data in enumerate(datas):
                if last_frame is not None:
                    if last_frame in data.frames and frame in data.frames:
                        last_position = data.position[last_frame]
                        position = data.position[frame]

                        node = last_path_nodes.get(tracki)
                        if node is None:
                            node = self.get_node(position)
                        node.update_use(path_age)
            last_frame = frame
            path_age += 1

    def analyse(self, features, params):
        out_features = []
        return out_features

    def get_node(self, position):
        for node in self.nodes:
            if math.dist(node.position, position) < self.node_distance:
                return node
        node = PathNode(position)
        self.nodes.append(node)
        return node
