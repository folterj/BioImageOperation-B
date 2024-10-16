import csv


class CsvStreamWriter:
    def __init__(self, filename, batch_size=1000):
        self.batch_size = batch_size
        self.file = open(filename, 'w', newline='')
        self.writer = None
        self.data = []

    def write(self, data):
        self.data.append(data)
        if len(self.data) >= self.batch_size:
            self.write_batch()
            self.data = []

    def write_batch(self):
        if self.writer is None:
            column_names = list(self.data[0].keys())
            self.writer = csv.DictWriter(self.file, fieldnames=column_names)
            self.writer.writeheader()
        self.writer.writerows(self.data)

    def close(self):
        if len(self.data) > 0:
            self.write_batch()
        self.file.close()
