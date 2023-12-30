import math
import numpy as np
import os

from src.file.generic import import_file
from src.file.plain_csv import export_csv
from src.parameters import PROFILE_HIST_BINS, VANGLE_NORM
from src.util import get_filetitle, extract_filename_id_info, isvalid_position, create_window


class Data:
    def __init__(self, data=None, filename=None, info=None, id=None, fps=1, pixel_size=1, window_size='1s'):
        self.data = data
        self.filename = filename
        self.info = info
        self.id = id
        self.fps = fps
        self.pixel_size = pixel_size
        self.window_size = window_size

        if self.filename is not None:
            self.filetitle = get_filetitle(filename)
            self.original_title = self.filetitle
            id_info = extract_filename_id_info(self.filename)
            if id is None:
                self.id = id_info[0]
            if info is None:
                self.info = id_info[1:]
        if self.id is None or self.id == '':
            self.id = str(list(data['track_label'].values())[0])
        self.id_info = [self.id]
        if self.info is not None:
            self.id_info += self.info
        self.original_id = self.id

        self.has_data = (data is not None)
        if self.has_data:
            self.features = {}
            self.profiles = {}
            self.calc_basic()

        self.new_label = None
        self.match_dist = None

    def get_frame_data(self, frame):
        if frame in self.frames:
            return {key: {frame: values[frame]} for key, values in self.data.items()}

    def set_new_label(self, new_label, match_dist=0):
        self.new_label = new_label
        self.match_dist = match_dist
        self.data['track_label'] = {frame: new_label for frame in self.data['track_label'].keys()}
        new_title = self.original_title
        if new_title.endswith(self.original_id):
            new_title = new_title.rstrip(self.original_id)
        new_title += new_label
        self.new_title = new_title

    def calc_basic(self):
        if not self.has_data:
            return
        data = self.data
        self.dtime = np.mean(np.diff(list(data['time'].values())))
        if 'frame' in self.data:
            self.frames = list(data['frame'].values())
        else:
            self.frames = list(data['x'].keys())
        self.frames = np.int0(self.frames)
        self.n = len(self.frames)

        positions = {}
        dist = {}
        if 'position' not in data or 'dist' not in data:
            position = None
            last_position = None
            for frame, x, y in zip(data['x'].keys(), data['x'].values(), data['y'].values()):
                if isvalid_position((x, y)):
                    position = (x, y)
                    positions[frame] = position

                if 'dist' not in data:
                    if position is not None:
                        if last_position is not None:
                            dist[frame] = math.dist(last_position, position)
                        last_position = position
            if 'dist' not in data:
                data['dist'] = dist
        if 'position' in data:
            positions = data['position']
        self.position = positions

        #TODO: calc v, a, angle_v, angle_a, using pixel_size etc

    def calc_windows(self):
        if self.window_size.endswith('s'):
            window_size = int(self.window_size[:-1].strip())
            self.window_frames = int(round(window_size * self.fps))
        else:
            self.window_frames = int(self.window_size)

        self.calc_window('dist', 'dist1')
        self.calc_window('v', 'v1')
        self.calc_window('a', 'a1')
        self.calc_window('angle', 'angle1')
        self.calc_window('v_angle', 'v_angle1')
        self.calc_window('a_angle', 'a_angle1')
        self.calc_window('projection', 'projection1')
        self.calc_window('v_projection', 'v_projection1')
        self.calc_window('area', 'area1')

    def calc_window(self, source, dest):
        if dest not in self.data and source in self.data:
            self.data[dest] = create_window(self.frames, self.data[source], self.window_frames)

    def calc_means(self):
        self.meanx = self.get_mean_feature('x')
        self.meany = self.get_mean_feature('y')
        if 'length_major1' in self.data:
            self.meanl = self.get_mean_feature('length_major1')
            self.meanw = self.get_mean_feature('length_minor1')

    def get_mean_feature(self, feature):
        return np.mean(list(self.data[feature].values()))

    def calc_profiles(self):
        v = np.asarray(list(self.data['v'].values()))
        v_angle = np.asarray(list(self.data['v_angle'].values()))
        self.v_norm = v / self.meanl
        self.angle_norm = abs(v_angle) / VANGLE_NORM
        self.features['v_percentiles'] = {f'v {percentile}% percentile': np.percentile(v, percentile)
                                          for percentile in [25, 50, 75]}
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
        return {activity_type: self.get_activity_time(activity_type) for activity_type in self.nactivity}

    def get_activity_time(self, type):
        return self.nactivity[type] * self.dtime

    def get_activity_fraction(self, type, total_frames=None):
        if total_frames is not None and total_frames != 0:
            return self.nactivity[type] / total_frames
        return self.nactivity[type]

    def __str__(self):
        return f'{self.original_title} {self.new_label}'


def create_data(filenames, fps=1, pixel_size=1, window_size='1s'):
    all_data = []
    for filename in filenames:
        data = import_file(filename)
        for id, data1 in data.items():
            all_data.append(Data(data=data1, filename=filename, id=id,
                                 fps=fps, pixel_size=pixel_size, window_size=window_size))
    return all_data


def read_data_dict(filename, fps=1, pixel_size=1, window_size='1s'):
    data = import_file(filename)
    id = next(iter(data.keys()))
    data_dict = {id: Data(data=data[id], filename=filename, id=id,
                          fps=fps, pixel_size=pixel_size, window_size=window_size)}
    return data_dict


def read_data(filename, fps=1, pixel_size=1, window_size='1s'):
    data0 = import_file(filename)
    id = next(iter(data0.keys()))
    data = Data(data=data0[id], filename=filename, id=id,
                fps=fps, pixel_size=pixel_size, window_size=window_size)
    return data


def write_data(output_folder, all_data):
    for data in all_data:
        filetitle = os.path.basename(data.filetitle) + '.csv'
        filename = os.path.join(output_folder, filetitle)
        export_csv(filename, {data.id: data.data})
