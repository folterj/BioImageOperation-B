import pyarrow


class FeatherStreamWriter:
    def __init__(self, filename, schema, batch_size):
        self.schema = schema
        self.batch_size = batch_size
        self.writer = pyarrow.ipc.new_stream(filename, schema)
        self.create_new_data()

    def create_new_data(self):
        self.data = {name: [] for name in self.schema.names}
        self.n = 0

    def write(self, data):
        for key, value in data.items():
            self.data[key].append(value)
        self.n += 1
        if self.n >= self.batch_size:
            self.write_batch()
            self.create_new_data()

    def write_batch(self):
        batch = pyarrow.record_batch(self.data, schema=self.schema)
        self.writer.write_batch(batch)

    def close(self):
        if self.n > 0:
            self.write_batch()
        self.writer.close()
