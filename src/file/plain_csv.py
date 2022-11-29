import csv
import math
import pandas as pd


def import_csv(filename):
    # dict[id][frame]
    # (id/frame can be None)
    data = {}
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        columns0 = next(reader)
        id_cols = []
        frame_cols = []
        for columni, column in enumerate(columns0):
            if column.lower() in ['id', 'track_label']:
                id_cols.append(columni)
            if column.lower() == 'frame':
                frame_cols.append(columni)
        columns = []
        for column0 in columns0:
            column = column0
            # check for duplicate column labels
            if columns0.count(column0) > 1:
                i = 0
                while True:
                    column = f'{column0}_{i}'
                    if column not in columns:
                        break
                    i += 1
            columns.append(column.lower())
    df = pd.read_csv(filename, names=columns, skiprows=1)
    if len(id_cols) > 0:
        ids = set()
        for id_col in id_cols:
            for value in df[columns[id_col]]:
                if not math.isnan(value):
                    ids.add(int(value))
        for id in ids:
            # check all id columns for specified id
            for id_col in id_cols:
                col_name = columns[id_col]
                id_df = df[df[col_name] == id].copy()
                if len(id_df) > 0:
                    parts = col_name.rsplit('_', 1)
                    label_set = parts[-1]
                    if str.isnumeric(label_set):
                        # filter column names
                        for col in list(id_df.columns):
                            parts = col.rsplit('_', 1)
                            if str.isnumeric(parts[-1]):
                                if parts[-1] == label_set:
                                    id_df.columns = id_df.columns.str.replace(col, parts[0])
                                else:
                                    id_df = id_df.drop(columns=col)
                    id_dict = dataframe_to_frame_dict(id_df, columns, frame_cols)
                    if not str(id) in data:
                        data[str(id)] = id_dict
                    else:
                        for key, value in id_dict.items():
                            data[str(id)][key] = dict(sorted((data[str(id)][key] | value).items()))
    else:
        data['0'] = dataframe_to_frame_dict(df, columns, frame_cols)
    return data


def dataframe_to_frame_dict(df, columns, frame_cols=[]):
    if len(frame_cols) > 0:
        frame_col = columns[frame_cols[0]]
    else:
        frame_col = 'frame'
        df.insert(0, frame_col, range(len(df)))
    df.set_index(frame_col, drop=False, inplace=True)
    return df.to_dict()


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
