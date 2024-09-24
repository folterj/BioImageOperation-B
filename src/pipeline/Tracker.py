import pyarrow

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

    def track(self, input_files=None):
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
                        self.track_frame(framei, id, positions)

    def track_frame(self, framei, id, positions):
        posi = int((len(positions) - 1) / 2)
        position = positions[posi]
        pass
