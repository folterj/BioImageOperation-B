import pyarrow
# TODO: add CSV support


def calc_features_function_dummy(data):
    return data

class StreamReader:
    def __init__(self, input_files):
        self.input_files = input_files
        self.query_stream()

    def query_stream(self):
        total_rows = 0
        batch_size = None
        for input_file in self.input_files:
            with pyarrow.ipc.open_file(input_file) as reader:
                self.schema = reader.schema
                for batchi in range(reader.num_record_batches):
                    batch_rows = reader.get_batch(batchi).num_rows
                    if batch_size is None:
                        batch_size = batch_rows
                    total_rows += batch_rows
        self.total_rows = total_rows
        self.batch_size = batch_size

    def get_stream_iterator(self, id_label='id', calc_features_function=None):
        if calc_features_function is None:
            calc_features_function = calc_features_function_dummy
        for input_file in self.input_files:
            with pyarrow.ipc.open_file(input_file) as reader:
                for batchi in range(reader.num_record_batches):
                    batch = reader.get_batch(batchi)
                    for i in range(batch.num_rows):
                        row = {column_name: batch[column_name][i].as_py() for column_name in batch.column_names}
                        yield {'frame': int(row['frame']),
                               'id': int(row[id_label]),
                               'values': calc_features_function(row),
                               'original_values': row}
