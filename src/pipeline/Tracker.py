import pyarrow
from sklearn.metrics import euclidean_distances
from tqdm import tqdm

from src.util import *
from src.video import video_iterator, draw_annotation, video_info


class Tracker:
    def __init__(self, params, base_dir, input_files, video_input, output, video_output):
        self.base_dir = base_dir
        self.input_files = input_files
        self.video_input = video_input
        self.output = output
        self.video_output = video_output
        self.frame_interval = params.get('frame_interval', 1)
        self.operations = params.get('operations')

        self.max_move_distance = params.get('max_move_distance', 1)
        self.max_inactive = params.get('max_inactive', 3)
        self.tracks = {}
        self.next_id = 0

    def track(self, input_files=None):
        # for each unique frame, collect all ids+values into single dictionary[id] = values
        # assume that order of the data is: frames, ids
        values = {}
        last_frame = None
        if input_files is None:
            input_files = self.input_files
        total_rows = 0
        for input_file in input_files:
            with pyarrow.ipc.open_file(input_file) as reader:
                for batchi in range(reader.num_record_batches):
                    total_rows += reader.get_batch(batchi).num_rows

        if self.video_output is not None:
            width, height, nframes, fps = video_info(self.video_input[0])
            vidwriter = cv.VideoWriter(self.video_output, -1, fps, (width, height))
            label_color = color_float_to_cv((0, 1, 0))
        else:
            # avoid warnings
            vidwriter = None
            label_color = None

        data_iterator = self.data_iterator(input_files)
        if self.video_input:
            image_iterator = video_iterator(self.video_input, frame_interval=self.frame_interval)
        else:
            image_iterator = iter([])

        for data, image in tqdm(zip(data_iterator, image_iterator), total=total_rows):
            frame = data['frame']
            id = data['id']
            if frame != last_frame and last_frame is not None:
                self.track_frame(last_frame, values)
                values = {}
            values[id] = data['values']
            values[id]['original_values'] = data['original_values']
            last_frame = frame
            if self.video_output is not None:
                for id, track in self.tracks.items():
                    draw_annotation(image, str(id), track['position'], color=label_color)
                vidwriter.write(image)
        # final frame
        if len(values) > 0:
            self.track_frame(last_frame, values)

        if vidwriter is not None:
            vidwriter.release()

    def data_iterator(self, input_files=None):
        for input_file in input_files:
            with pyarrow.ipc.open_file(input_file) as reader:
                for batchi in range(reader.num_record_batches):
                    batch = reader.get_batch(batchi)
                    for i in range(batch.num_rows):
                        row = {column_name: batch[column_name][i].as_py() for column_name in batch.column_names}
                        yield {'frame': int(row['frame']),
                               'id': int(row['track_id']),
                               'values': self.calc_features(row),
                               'original_values': row}

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
        # update tracks
        for id, track in list(self.tracks.items()):
            track['assigned'] = False
            if framei - track['last_active'] > self.max_inactive:
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
                values = ids[id_index]
                position = values['position']
                assigned = False
                for match_track_index in all_best_indices[id_index]:
                    track = tracks_list[match_track_index]
                    if not track['assigned'] and math.dist(track['position'], position) < self.max_move_distance:
                        track.update(values)
                        track['last_active'] = framei
                        track['assigned'] = True
                        assigned = True
                        break
                if not assigned:
                    # add tracks for any non-assigned ids
                    self.add_track(values, framei)
        else:
            # create tracks for all ids
            for values in ids.values():
                self.add_track(values, framei)

    def add_track(self, values, framei=-1):
        self.tracks[self.next_id] = {'last_active': framei}
        self.tracks[self.next_id].update(values)
        self.next_id += 1
