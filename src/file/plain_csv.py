import csv
import numpy as np


def export_csv(infilename, outfilename, headers, data):
    with open(infilename) as infile:
        reader = csv.reader(infile)
        headers0 = next(reader)
        with open(outfilename, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(headers0 + headers)
            for row in np.transpose(data):
                row0 = next(reader)
                writer.writerow(row0 + list(row))
