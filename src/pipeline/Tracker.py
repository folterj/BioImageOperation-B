import numpy as np
from sklearn.metrics import euclidean_distances, pairwise_distances
from tqdm import tqdm

from src.file.FeatherFileReader import FeatherFileReader
from src.file.FeatherFileWriter import FeatherFileWriter
from src.file.FeatherStreamReader import FeatherStreamReader
from src.file.CsvStreamWriter import CsvStreamWriter
from src.file.FeatherStreamWriter import FeatherStreamWriter
from src.util import *
from src.video import video_iterator, draw_annotation, video_info


class Tracker:
    def __init__(self, params, base_dir, input_files, video_input, output, video_output, debug_mode=False):
        self.base_dir = base_dir
        self.input_files = input_files
        self.video_input = video_input
        self.output = output
        self.video_output = video_output
        self.debug_mode = debug_mode
        _, _, nframes, fps = video_info(self.video_input[0])
        self.frame_interval = get_frames_number(params.get('frame_interval', 1), fps)
        self.frame_start = get_frames_number(params.get('frame_start', 0), fps)
        self.frame_end = get_frames_number(params.get('frame_end', nframes), fps)
        self.operations = params.get('operations')

        self.id_label = params.get('id_label', 'id')
        self.move_distance = params.get('move_distance', 1)
        self.max_move_distance = params.get('max_move_distance', 1)
        self.min_active = params.get('min_active', 3)
        self.max_inactive = params.get('max_inactive', 3)
        self.tracks = {}
        self.next_id = 0

    def track(self, input_files=None):
        # for each unique frame, collect all ids+values into single dictionary[id] = values
        # assume that order of the data is: frames, ids
        if input_files is None:
            input_files = self.input_files
        try:
            data_reader = FeatherStreamReader(input_files)
        except Exception as e:
            print(f'Warning: unable to open input files as stream ({e})')
            data_reader = FeatherFileReader(input_files)
        data_iterator = data_reader.get_stream_iterator()
        if self.video_input:
            frame_iterator = video_iterator(self.video_input,
                                            start=self.frame_start, end=self.frame_end, interval=self.frame_interval)
        else:
            frame_iterator = iter([])

        if self.output:
            batch_size = 1000
            data_writers = [
                FeatherStreamWriter(self.output + '_stream.feather', batch_size),
                FeatherFileWriter(self.output + '.feather', batch_size),
                CsvStreamWriter(self.output + '.csv', batch_size),
            ]
        else:
            data_writers = []

        if self.debug_mode:
            self.debug_writer = CsvStreamWriter(self.output + '_debug.csv')

        if self.video_output:
            width, height, nframes, fps = video_info(self.video_input[0])
            fourcc = cv.VideoWriter.fourcc(*'avc1')
            vidwriter = cv.VideoWriter(self.video_output, fourcc, fps, (width, height))
            label_color = color_float_to_cv((0, 0, 1))
            inactive_color = color_float_to_cv((0.5, 0.5, 1))
        else:
            width, height = 0, 0
            vidwriter = None
            label_color = None
            inactive_color = None

        frames = range(self.frame_start, self.frame_end, self.frame_interval)
        data = {'frame': -1, 'values': {}}
        for framei in tqdm(frames, total=len(frames)):
            if self.video_output:
                if self.video_input:
                    image = next(frame_iterator)
                else:
                    image = np.zeros((height, width, 3), np.uint8)
            else:
                image = None
            frame_done = False
            frame_values = {}
            while not frame_done:
                data_framei = int(data['frame'])
                if data_framei == framei:
                    track_id = int(data[self.id_label])
                    frame_values[track_id] = self.calc_features(data)
                    frame_values[track_id]['original_values'] = data
                elif data_framei > framei:
                    frame_done = True
                if not frame_done:
                    data = next(data_iterator)

            self.track_frame(framei, frame_values)
            if self.output:
                for track_id, track in self.tracks.items():
                    if track['assigned']:
                        values = track['original_values']
                        values['track_id'] = track_id
                        for data_writer in data_writers:
                            data_writer.write(values)
            if self.video_output:
                for track_id, track in self.tracks.items():
                    if track['assigned'] or self.debug_mode:
                        if track['assigned']:
                            color = label_color
                        else:
                            color = inactive_color
                        draw_annotation(image, str(track_id), track['position'], color=color)
                vidwriter.write(image)

        if self.output:
            for data_writer in data_writers:
                data_writer.close()

        if self.debug_mode:
            self.debug_writer.close()

        if self.video_output:
            vidwriter.release()

    def calc_features(self, values0):
        positions = []
        positions.append((values0['x_head'], values0['y_head']))
        positions.append((values0['x_body'], values0['y_body']))
        positions.append((values0['x_tail'], values0['y_tail']))
        values = {'positions': positions}
        posi = int((len(positions) - 1) / 2)    # find the middle position
        values['position'] = positions[posi]
        last_position = None
        dists = []
        for position1 in positions:
            if not position1 == (None, None):
                if last_position is not None:
                    dists.append(math.dist(last_position, position1))
                last_position = position1
        values['length'] = np.sum(dists)
        return values

    def track_frame(self, framei, ids):
        def length_distance(length, mean_length):
            return abs(length - mean_length)

        id_list = list(ids.keys())
        # update tracks
        for id, track in list(self.tracks.items()):
            if not self.update_track(track, framei):
                self.tracks.pop(id)
        # find matching tracks (greedy matching)
        id_positions = [id['position'] for id in ids.values()]
        tracks_list = list(self.tracks.values())
        track_positions = [track['position'] for track in tracks_list]
        if len(id_positions) > 0 and len(track_positions) > 0:
            distance_matrix = euclidean_distances(id_positions, track_positions)
            #distance_matrix2 = pairwise_distances([id['length'] for id in ids.values()],
            #                                      [track['mean_length'] for track in tracks_list],
            #                                      metric=length_distance)
            #distance_matrix = distance_matrix1 + distance_matrix2
            all_best_indices = []
            best_dists = []
            for id_index in range(len(ids)):
                distances = distance_matrix[id_index]
                best_indices = np.argsort(distances)
                all_best_indices.append(best_indices)
                best_dists.append(distances[best_indices[0]])
            greedy_indices = np.argsort(best_dists)
            for id_index in greedy_indices:
                values = ids[id_list[id_index]]
                position = values['position']
                assigned = False
                for match_track_index in all_best_indices[id_index]:
                    track = tracks_list[match_track_index]
                    distance = math.dist(track['position'], position)
                    if not track['assigned'] and self.check_match(track, distance):
                        if self.debug_mode:
                            self.debug_writer.write({'distance': distance})
                        self.assign_track(track, values, distance, framei)
                        assigned = True
                        break
                if not assigned:
                    # add tracks for any non-assigned ids
                    self.add_track(values, framei)
        else:
            # create tracks for all ids
            for values in ids.values():
                self.add_track(values, framei)

    def check_match(self, track, distance):
        return distance < self.max_move_distance + track['inactive_count'] * self.move_distance

    def add_track(self, values, framei):
        self.tracks[self.next_id] = {'assigned': True,
                                     'active_count': 0, 'inactive_count': 0, 'last_active': framei,
                                     'mean_length': values['length'], 'delta': np.zeros(2)}
        track = self.tracks[self.next_id]
        track.update(values)
        self.next_id += 1

    def assign_track(self, track, values, distance, framei):
        add_factor = 0.1
        active_factor = calc_active_factor(track, self.min_active)
        range_factor = calc_range_factor(track, distance, self.move_distance)
        length = track['length'] if track['length'] is not None else track['mean_length']
        length_factor = calc_length_factor(track, abs(length - values['length']))
        match_factor = range_factor * length_factor * active_factor
        delta = np.array(values['position']) - track['position']
        track['delta'] = delta * add_factor + track['delta'] * (1 - add_factor)
        track.update(values)
        track['assigned'] = True
        track['active_count'] += 1
        track['inactive_count'] = 0
        track['last_active'] = framei
        track['mean_length'] = track['mean_length'] * (1 - match_factor) + length * match_factor

    def update_track(self, track, framei):
        if not track['assigned']:
            track['inactive_count'] += 1
            delta = track['delta']
            l, angle = vector_to_polar(delta)
            if l > self.move_distance:
                delta = np.array(polar_to_vector([self.move_distance, angle]))
            else:
                delta *= 0.95
            track['position'] = track['position'] + delta
            track['delta'] = delta
        track['assigned'] = False
        return framei - track['last_active'] < self.max_inactive


def calc_active_factor(track, min_active):
    active_factor = min((track['active_count'] + 1) / (min_active + 1), 1)
    return active_factor


def calc_range_factor(track, distance, move_distance):
    if track['inactive_count'] == 0:
        range_factor = 1 - distance / move_distance
    else:
        factor_inactive = min(track['inactive_count'] * 0.1, 1)
        range_factor = (1 - distance / (move_distance * factor_inactive)) / factor_inactive
    return range_factor


def calc_length_factor(track, length_dif):
    length_factor = 1
    l = track['mean_length']
    if l == 0 and track['length'] is not None:
        l = track['length']
    if not np.isclose(l, 0):
        length_factor = max(1 - length_dif / l, 0)
    return length_factor
