import numpy as np

from src.file.bio import import_tracks_by_frame
from src.util import get_filetitle


class BioData:
    def __init__(self, filename, start=0):
        self.filename = filename
        data = import_tracks_by_frame(filename)
        self.frames = sorted(data['area'].keys())
        frame0 = self.frames[0]
        self.start = start + frame0
        self.length = self.frames[-1] - self.frames[0]
        self.end = self.start + self.length
        self.old_title = get_filetitle(filename)
        if 'track_label' in data:
            self.old_label = str(data['track_label'][frame0])
        else:
            self.old_label = self.old_title.rsplit('_')[-1]

        self.x = data['x']
        self.y = data['y']
        self.meanx = np.mean(list(data['x'].values()))
        self.meany = np.mean(list(data['y'].values()))
        self.meanarea = np.mean(list(data['area'].values()))
        self.meanlength = np.mean(list(data['length_major'].values()))

        self.pref_label = None
        self.pref_label_dist = None
