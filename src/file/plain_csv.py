import csv


def export_csv(infilename, outfilename, headers, data):
    with open(infilename) as infile:
        reader = csv.reader(infile)
        headers0 = next(reader)
        with open(outfilename, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(headers0 + headers)
            for row in reader:
                frame = int(row[0])
                add = []
                for data_row in data:
                    if frame in data_row:
                        add.append(data_row[frame])
                writer.writerow(row + add)
