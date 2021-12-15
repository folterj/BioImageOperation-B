import os
import numpy as np

from src.file.bio import import_tracks_by_frame
from src.parameters import NBINS, VANGLE_NORM


class BioFeatures:
    def __init__(self, filename):
        self.filename = filename
        self.filetitle = os.path.splitext(os.path.basename(filename))[0].replace("_", " ")
        self.data = import_tracks_by_frame(filename)
        self.extract_filename_info()
        self.calc()

    def extract_filename_info(self):
        parts = self.filetitle.split()
        i = 0
        if not parts[i].isnumeric():
            i += 1
        date = parts[i]
        time = parts[i + 1].replace('-', ':')
        id = parts[-1]

        camera = 0
        s = self.filename.lower().find('cam')
        if s >= 0:
            while s < len(self.filename) and not self.filename[s].isnumeric():
                s += 1
            e = s
            while e < len(self.filename) and self.filename[e].isnumeric():
                e += 1
            if e > s:
                camera = int(self.filename[s:e])

        self.id = id
        self.info = [id, date, time, camera]

    def calc(self):
        self.dtime = np.mean(np.diff(list(self.data['time'].values())))
        frames = self.data['frame']
        self.n = len(frames)
        self.positions = {frame: (x, y) for frame, x, y in zip(frames.values(), self.data['x'].values(), self.data['y'].values())}
        length_major = self.data['length_major1']
        length_minor = self.data['length_minor1']
        self.meanl = np.mean(list(length_major.values()))
        self.meanw = np.mean(list(length_minor.values()))
        v = np.asarray(list(self.data['v'].values()))
        v_angle = np.asarray(list(self.data['v_angle'].values()))
        self.v_norm = v / self.meanl
        self.angle_norm = abs(v_angle) / VANGLE_NORM
        self.v_percentiles = np.percentile(v, [25, 50, 75])
        self.v_hist = self.calc_loghist(self.v_norm, -2, 2)
        self.vangle_hist = self.calc_loghist(self.angle_norm, -3, 1)

    def calc_hist(self, data, range):
        hist, bin_edges = np.histogram(data, bins=NBINS, range=(0, range))
        return hist / len(data)

    def calc_loghist(self, data, power_min, power_max):
        # assume symtric scale - middle bin is 10^0 = 1
        #bin_edges = [10 ** ((i - NBINS / 2) * factor) for i in range(NBINS + 1)]
        bin_edges = np.logspace(power_min, power_max, NBINS + 1)

        # manual histogram, ensuring values at (positive) histogram edges are counted
        hist = np.zeros(NBINS)
        factor = (power_max - power_min) / NBINS
        for x in data:
            if x != 0:
                bin = (np.log10(abs(x)) - power_min) / factor
                if bin >= 0:
                    # discard low values
                    bin = np.clip(int(bin), 0, NBINS)
                    hist[bin] += 1
        hist /= self.n

        #hist, _ = np.histogram(data, bins=bin_edges)

        return hist, bin_edges

    def draw_loghists(self, ax_v, ax_vangle, filetitle_plot):
        self.draw_loghist(self.v_hist, ax_v, 'v ' + filetitle_plot, '#1f77b4')
        self.draw_loghist(self.vangle_hist, ax_vangle, 'vangle ' + filetitle_plot, '#ff7f0e')

    def draw_loghist(self, hist, ax, title='', color='#1f77b4'):
        h, b, _ = ax.hist(hist[0], bins=hist[1], weights=[1 / self.n] * self.n, color=color)
        ax.set_xscale('log')
        ax.title.set_text(title)

    def classify_movement(self, output_type):
        self.movement_type = {}
        if output_type == 'movement_type':
            self.movement_time = {'': 0, 'brownian': 0, 'levi': 0, 'ballistic': 0}
        elif output_type == 'activity_type':
            self.movement_time = {'': 0, 'appendages': 0, 'moving': 0}
        else:
            self.movement_time = {}

        frames = self.data['frame']
        v_all = self.data['v_projection']
        v_angle_all = self.data['v_angle']
        length_major = self.data['length_major1']
        length_minor = self.data['length_minor1']

        lenl0 = self.meanl
        lenw0 = self.meanw
        for frame, v, v_angle, lenl, lenw in zip(frames.values(), v_all.values(), v_angle_all.values(),
                                                 length_major.values(), length_minor.values()):
            v_norm = v / self.meanl
            length_major_delta = abs(lenl - lenl0) / self.meanl
            length_minor_delta = abs(lenw - lenw0) / self.meanw
            v_angle_norm = abs(v_angle) / VANGLE_NORM

            type = ''
            if output_type == 'movement_type':
                if v_norm > 3:
                    type = 'ballistic'
                elif abs(v_norm) > 0.6:
                    if v_norm > 0.6 and v_angle_norm < 0.05:
                        type = 'levi'
                    else:
                        type = 'brownian'
                else:
                    type = ''

            elif output_type == 'activity_type':
                if v_norm > 0.2:
                    type = 'moving'
                elif length_major_delta + length_minor_delta > 0.01:
                    type = 'appendages'
                else:
                    type = ''

            lenl0 = lenl
            lenw0 = lenw

            self.movement_type[frame] = type
            self.movement_time[type] += 1

    def get_movement_time(self, type):
        return self.movement_time[type] * self.dtime

    def get_movement_fraction(self, type):
        return self.movement_time[type] / self.n

    def get_movement_times(self):
        return [self.get_movement_time(type) for type in self.movement_time]

    def get_movement_fractions(self):
        return [self.get_movement_fraction(type) for type in self.movement_time]
