import numpy as np

from src.file.generic import import_file
from src.util import get_filetitle


class BioData:
    def __init__(self, filename):
        self.filename = filename
        data0 = import_file(filename)
        data = data0[next(iter(data0))]
        self.data = data
        self.frames = sorted(data['x'].keys())
        if len(self.frames) > 0:
            frame0 = self.frames[0]
        else:
            frame0 = None
        self.original_title = get_filetitle(filename)
        if 'track_label' in data and frame0 is not None:
            self.original_label = str(data['track_label'][frame0])
        else:
            self.original_label = self.original_title.rsplit('_')[-1]

        self.x = data['x']
        self.y = data['y']
        self.meanx = np.mean(list(data['x'].values()))
        self.meany = np.mean(list(data['y'].values()))
        self.meanarea = np.mean(list(data['area'].values()))
        self.meanlength = np.mean(list(data['length_major'].values()))
        lines = open(filename).readlines()
        self.header = lines[0]
        self.lines = {frame: line for frame, line in zip(self.frames, lines[1:])}
        self.frames = set(self.frames)

        self.new_label = None
        self.match_dist = None

    def get_frame_data(self, frame):
        if frame in self.frames:
            return {key: {frame: values[frame]} for key, values in self.data.items()}
        else:
            return None

    def set_new_label(self, label, match_dist):
        self.new_label = label
        self.match_dist = match_dist
        self.data['track_label'] = {frame: label for frame in self.data['track_label'].keys()}
        new_title = self.original_title
        if new_title.endswith(self.original_label):
            new_title = new_title.rstrip(self.original_label)
        new_title += label
        self.new_title = new_title

    def __str__(self):
        return f'{self.original_title} {self.new_label} ({np.round(self.meanx)},{np.round(self.meany)})'
