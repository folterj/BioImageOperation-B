import numpy as np

from src.file.generic import import_file
from src.parameters import PROFILE_HIST_BINS, VANGLE_NORM
from src.util import get_filetitle, extract_filename_id_info, isvalid_position


class BioFeatures:
    def __init__(self, data=None, filename=None, info=None, id=None):
        self.data = data
        self.filename = filename
        self.info = info
        self.id = id
        if self.filename is not None:
            self.filetitle = get_filetitle(filename)
            id_info = extract_filename_id_info(self.filename)
            if id is None:
                self.id = id_info[0]
            if info is None:
                self.info = id_info[1:]
        self.id_info = [self.id] + self.info
        self.has_data = (data is not None)
        if self.has_data:
            self.features = {}
            self.profiles = {}
            self.calc_basic()

    def calc_basic(self):
        if not self.has_data:
            return
        self.dtime = np.mean(np.diff(list(self.data['time'].values())))
        if 'frame' in self.data:
            self.frames = list(self.data['frame'].values())
        else:
            self.frames = list(self.data['x'].keys())
        self.frames = np.int0(self.frames)
        self.n = len(self.frames)
        self.position = {frame: (x, y) for frame, x, y in zip(self.frames, self.data['x'].values(), self.data['y'].values())
                         if isvalid_position((x, y))}
        if 'length_major1' in self.data:
            length_major = self.data['length_major1']
            length_minor = self.data['length_minor1']
            self.meanl = np.mean(list(length_major.values()))
            self.meanw = np.mean(list(length_minor.values()))

    def get_mean_feature(self, feature):
        if not self.has_data:
            return 0
        return np.mean(list(self.data[feature].values()))

    def calc_profiles(self):
        if not self.has_data:
            return
        v = np.asarray(list(self.data['v'].values()))
        v_angle = np.asarray(list(self.data['v_angle'].values()))
        self.v_norm = v / self.meanl
        self.angle_norm = abs(v_angle) / VANGLE_NORM
        self.features['v_percentiles'] = {f'v {percentile}% percentile': np.percentile(v, percentile) for percentile in [25, 50, 75]}
        self.profiles['v'] = self.calc_loghist(self.v_norm, -2, 2)
        self.profiles['vangle'] = self.calc_loghist(self.angle_norm, -3, 1)

    def calc_hist(self, data, range):
        hist, bin_edges = np.histogram(data, bins=PROFILE_HIST_BINS, range=(0, range))
        return hist / len(data)

    def calc_loghist(self, data, power_min, power_max):
        # assume symtric scale - middle bin is 10^0 = 1
        #bin_edges = [10 ** ((i - NBINS / 2) * factor) for i in range(NBINS + 1)]
        bin_edges = np.logspace(power_min, power_max, PROFILE_HIST_BINS + 1)

        # manual histogram, ensuring values at (positive) histogram edges are counted
        hist = np.zeros(PROFILE_HIST_BINS)
        factor = (power_max - power_min) / PROFILE_HIST_BINS
        for x in data:
            if x != 0:
                bin = (np.log10(abs(x)) - power_min) / factor
                if bin >= 0:
                    # discard low values
                    bin = np.clip(int(bin), 0, PROFILE_HIST_BINS - 1)
                    hist[bin] += 1
        hist /= self.n

        #hist, _ = np.histogram(data, bins=bin_edges)

        return hist, bin_edges

    def draw_loghists(self, ax_v, ax_vangle, filetitle_plot):
        self.draw_loghist(self.profiles['v'], ax_v, 'v ' + filetitle_plot, '#1f77b4')
        self.draw_loghist(self.profiles['vangle'], ax_vangle, 'vangle ' + filetitle_plot, '#ff7f0e')

    def draw_loghist(self, hist, ax, title='', color='#1f77b4'):
        h, b, _ = ax.hist(hist[0], bins=hist[1], weights=[1 / self.n] * self.n, color=color)
        ax.set_xscale('log')
        ax.title.set_text(title)

    def classify_activity(self, output_type):
        if not self.has_data:
            return {}

        self.activity = {}
        if output_type == 'movement':
            self.nactivity = {'': 0, 'brownian': 0, 'levi': 0, 'ballistic': 0}
        elif output_type == 'activity':
            self.nactivity = {'': 0, 'appendages': 0, 'moving': 0}
        else:
            self.nactivity = {}

        v_all = self.data['v_projection']
        v_angle_all = self.data['v_angle']
        length_major = self.data['length_major1']
        length_minor = self.data['length_minor1']

        lenl0 = self.meanl
        lenw0 = self.meanw
        for frame, v, v_angle, lenl, lenw in zip(self.frames, v_all.values(), v_angle_all.values(),
                                                 length_major.values(), length_minor.values()):
            v_norm = v / self.meanl
            length_major_delta = abs(lenl - lenl0) / self.meanl
            length_minor_delta = abs(lenw - lenw0) / self.meanw
            v_angle_norm = abs(v_angle) / VANGLE_NORM

            type = ''
            if output_type == 'movement':
                if v_norm > 3:
                    type = 'ballistic'
                elif abs(v_norm) > 0.6:
                    if v_norm > 0.6 and v_angle_norm < 0.05:
                        type = 'levi'
                    else:
                        type = 'brownian'
                else:
                    type = ''

            elif output_type == 'activity':
                if v_norm > 0.2:
                    type = 'moving'
                elif length_major_delta + length_minor_delta > 0.01:
                    type = 'appendages'
                else:
                    type = ''

            lenl0 = lenl
            lenw0 = lenw

            self.activity[frame] = type
            self.nactivity[type] += 1
        return self.nactivity

    def get_activities_time(self):
        return {type: self.get_activity_time(type) for type in self.nactivity}

    def get_activity_time(self, type):
        return self.nactivity[type] * self.dtime

    def get_activity_fraction(self, type, total_frames=None):
        if total_frames is not None and total_frames != 0:
            return self.nactivity[type] / total_frames
        return self.nactivity[type]


def create_biofeatures(filenames):
    biofeatures = []
    for filename in filenames:
        data = import_file(filename)
        for id, data1 in data.items():
            biofeatures.append(BioFeatures(data=data1, filename=filename, id=id))
    return biofeatures
