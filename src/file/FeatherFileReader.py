import pyarrow


class FeatherFileReader:
    def __init__(self, input_files):
        self.input_files = input_files
        self.query_stream()

    def query_stream(self):
        total_rows = 0
        batch_size = None
        for input_file in self.input_files:
            with pyarrow.RecordBatchFileReader(input_file) as reader:
                self.schema = reader.schema
                for batchi in range(reader.num_record_batches):
                    batch_rows = reader.get_batch(batchi).num_rows
                    if batch_size is None:
                        batch_size = batch_rows
                    total_rows += batch_rows
        self.total_rows = total_rows
        self.batch_size = batch_size

    def get_stream_iterator(self):
        for input_file in self.input_files:
            with pyarrow.RecordBatchFileReader(input_file) as reader:
                for batchi in range(reader.num_record_batches):
                    batch = reader.get_batch(batchi)
                    for i in range(batch.num_rows):
                        yield {column_name: batch[column_name][i].as_py() for column_name in batch.column_names}
