import pyarrow
from sklearn.metrics import euclidean_distances

from src.util import *


class Tracker:
    def __init__(self, params, base_dir, input_files, video_input, output, video_output):
        self.base_dir = base_dir
        self.input_files = input_files
        self.video_input = video_input
        self.output = output
        self.video_output = video_output
        self.frame_interval = params.get('frame_interval', 1)
        self.operations = params.get('operations')
        self.tracks = {}
        self.next_id = 0

    def track(self, input_files=None):
        # for each unique frame, collect all ids+values into single dictionary[id] = values
        # assume that order is frames, ids
        values = {}
        last_frame = None
        for data in self.data_iterator(input_files):
            frame = data['frame']
            id = data['id']
            if frame != last_frame and last_frame is not None:
                self.track_frame(last_frame, values)
                values = {}
            values[id] = data['values']
            last_frame = frame
        # final frame
        if len(values) > 0:
            self.track_frame(last_frame, values)

    def data_iterator(self, input_files=None):
        if input_files is None:
            input_files = self.input_files

        for input_file in input_files:
            with pyarrow.ipc.open_file(input_file) as reader:
                for batchi in range(reader.num_record_batches):
                    batch = reader.get_batch(batchi)
                    for i in range(batch.num_rows):
                        framei = int(batch['frame'][i].as_py())
                        id = int(batch['track_id'][i].as_py())
                        positions = []
                        positions.append((batch['x_head'][i].as_py(), batch['y_head'][i].as_py()))
                        positions.append((batch['x_body'][i].as_py(), batch['y_body'][i].as_py()))
                        positions.append((batch['x_tail'][i].as_py(), batch['y_tail'][i].as_py()))
                        yield {'frame': framei, 'id': id, 'values': self.calc_features(positions)}

    def calc_features(self, positions):
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
        # find matching tracks (greedy matching)
        id_positions = [id['position'] for id in ids.values()]
        track_positions = [track['position'] for track in self.tracks.values()]
        if len(id_positions) > 0 and len(track_positions) > 0:
            distance_matrix = euclidean_distances(id_positions, track_positions)
            for id_index in range(len(ids)):
                distances = distance_matrix[id_index]
                best_indices = np.argsort(distances)

        # add tracks for any non-assigned ids
        for id in ids.values():
            self.tracks[self.next_id] = id
            self.next_id += 1
