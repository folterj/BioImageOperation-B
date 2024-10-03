import pyarrow
# TODO: add CSV support


def calc_features_function_dummy(data):
    return data


def get_stream_iterator(input_files, calc_features_function=None):
    if calc_features_function is None:
        calc_features_function = calc_features_function_dummy
    for input_file in input_files:
        with pyarrow.ipc.open_file(input_file) as reader:
            for batchi in range(reader.num_record_batches):
                batch = reader.get_batch(batchi)
                for i in range(batch.num_rows):
                    row = {column_name: batch[column_name][i].as_py() for column_name in batch.column_names}
                    yield {'frame': int(row['frame']),
                           'id': int(row['track_id']),
                           'values': calc_features_function(row),
                           'original_values': row}


def get_stream_total(input_files):
    total_rows = 0
    for input_file in input_files:
        with pyarrow.ipc.open_file(input_file) as reader:
            for batchi in range(reader.num_record_batches):
                total_rows += reader.get_batch(batchi).num_rows
    return total_rows
