import pyarrow


class FeatherFileWriter:
    def __init__(self, filename, batch_size=1000):
        self.filename = filename
        self.batch_size = batch_size
        self.writer = None
        self.create_new_data()

    def create_new_data(self):
        self.data = {}
        self.n = 0

    def write(self, data):
        for key, value in data.items():
            if key not in self.data:
                self.data[key] = []
            self.data[key].append(value)
        self.n += 1
        if self.n >= self.batch_size:
            self.write_batch()

    def write_batch(self):
        batch = pyarrow.record_batch(self.data)
        if self.writer is None:
            self.writer = pyarrow.RecordBatchFileWriter(self.filename, batch.schema)
        self.writer.write_batch(batch)
        self.create_new_data()

    def close(self):
        if self.n > 0:
            self.write_batch()
        self.writer.close()
