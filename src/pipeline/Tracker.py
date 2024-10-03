import numpy as np
from sklearn.metrics import euclidean_distances
from tqdm import tqdm

from src.file.streaming import get_stream_total, get_stream_iterator
from src.util import *
from src.video import video_iterator, draw_annotation, video_info


class Tracker:
    def __init__(self, params, base_dir, input_files, video_input, output, video_output):
        self.base_dir = base_dir
        self.input_files = input_files
        self.video_input = video_input
        self.output = output
        self.video_output = video_output
        _, _, _, fps = video_info(self.video_input[0])
        self.frame_interval = get_frames_number(params.get('frame_interval', 1), fps)
        self.frame_start = get_frames_number(params.get('frame_start', 0), fps)
        self.frame_end = get_frames_number(params.get('frame_end'), fps)
        self.operations = params.get('operations')

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
        data_iterator = get_stream_iterator(input_files, self.calc_features)
        if self.video_input:
            frame_iterator = video_iterator(self.video_input,
                                            start=self.frame_start, end=self.frame_end, interval=self.frame_interval)
        else:
            frame_iterator = iter([])

        if self.video_output is not None:
            width, height, nframes, fps = video_info(self.video_input[0])
            vidwriter = cv.VideoWriter(self.video_output, -1, fps, (width, height))
            label_color = color_float_to_cv((0, 0, 1))
        else:
            # avoid warnings
            vidwriter = None
            label_color = None

        frames = range(self.frame_start, self.frame_end, self.frame_interval)
        data = {'frame': -1, 'values': {}}
        for framei in tqdm(frames, total=len(frames)):
            if self.video_output:
                if self.video_input:
                    image = next(frame_iterator)
                else:
                    image = np.zeros((height, width, 3), np.uint8)
            frame_done = False
            frame_values = {}
            while not frame_done:
                data_framei = data['frame']
                if data_framei == framei:
                    id = data['id']
                    values = data['values']
                    frame_values[id] = values
                    frame_values[id]['original_values'] = data['original_values']
                elif data_framei > framei:
                    frame_done = True
                if not frame_done:
                    data = next(data_iterator)

            self.track_frame(framei, frame_values)
            if self.video_output:
                for track_id, track in self.tracks.items():
                    draw_annotation(image, str(track_id), track['position'], color=label_color)
                vidwriter.write(image)

        if vidwriter is not None:
            vidwriter.release()

    def calc_features(self, values0):
        positions = []
        positions.append((values0['x_head'], values0['y_head']))
        positions.append((values0['x_body'], values0['y_body']))
        positions.append((values0['x_tail'], values0['y_tail']))
        values = {'positions': positions}
        posi = int((len(positions) - 1) / 2)
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
                    if not track['assigned'] and distance < self.max_move_distance:
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

    def add_track(self, values, framei):
        self.tracks[self.next_id] = {'assigned': True,
                                     'active_count': 0, 'inactive_count': 0, 'last_active': framei,
                                     'mean_length': values['length']}
        track = self.tracks[self.next_id]
        track.update(values)
        self.next_id += 1

    def assign_track(self, track, values, distance, framei):
        active_factor = calc_active_factor(track, self.min_active)
        range_factor = calc_range_factor(track, distance, self.max_move_distance)
        length_factor = calc_length_factor(track, abs(track['length'] - values['length']))
        match_factor = range_factor * length_factor * active_factor
        track.update(values)
        track['assigned'] = True
        track['active_count'] += 1
        track['inactive_count'] = 0
        track['last_active'] = framei
        track['mean_length'] = track['mean_length'] * (1 - match_factor) + track['length'] * match_factor

    def update_track(self, track, framei):
        if not track['assigned']:
            track['inactive_count'] += 1
        track['assigned'] = False
        return framei - track['last_active'] < self.max_inactive


def calc_active_factor(track, min_active):
    active_factor = min((track['active_count'] + 1) / (min_active + 1), 1)
    return active_factor


def calc_range_factor(track, distance, max_move_distance):
    range_factor = 1
    if max_move_distance > 0:
        if track['inactive_count'] == 0:
            range_factor = 1 - distance / max_move_distance
        else:
            factor_inactive = min(track['inactive_count'] * 0.1, 1)
            range_factor = (1 - distance / (max_move_distance * factor_inactive)) / factor_inactive
    return range_factor


def calc_length_factor(track, length_dif):
    length_factor = 1
    l = track['mean_length']
    if l == 0:
        l = track['length']
    if l != 0:
        length_factor = max(1 - length_dif / l, 0)
    return length_factor
