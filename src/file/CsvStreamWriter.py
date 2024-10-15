import csv


class CsvStreamWriter:
    def __init__(self, filename, columns, batch_size):
        self.batch_size = batch_size
        self.file = open(filename, 'w', newline='')
        self.writer = csv.DictWriter(self.file, fieldnames=columns)
        self.writer.writeheader()
        self.data = []

    def write(self, data):
        self.data.append(data)
        if len(self.data) >= self.batch_size:
            self.write_batch()
            self.data = []

    def write_batch(self):
        self.writer.writerows(self.data)

    def close(self):
        if len(self.data) > 0:
            self.write_batch()
        self.file.close()
