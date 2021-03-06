import numpy as np

from src.file.bio import import_tracks_by_frame
from src.util import get_filetitle


class BioData:
    def __init__(self, filename):
        self.filename = filename
        data = import_tracks_by_frame(filename)
        self.frames = sorted(data['area'].keys())
        if len(self.frames) > 0:
            frame0 = self.frames[0]
        else:
            frame0 = None
        self.old_title = get_filetitle(filename)
        if 'track_label' in data and frame0 is not None:
            self.old_label = str(data['track_label'][frame0])
        else:
            self.old_label = self.old_title.rsplit('_')[-1]

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

        self.pref_label = None
        self.pref_label_dist = None

    def __str__(self):
        return f'{self.old_title} {self.pref_label} ({np.round(self.meanx)},{np.round(self.meany)})'
