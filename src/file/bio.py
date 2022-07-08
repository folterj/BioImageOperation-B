import csv
import pandas as pd

from src.util import pairwise


def extract_contour(s):
    contour = []
    for xs, ys in pairwise(s.split()):
        contour.append((float(xs), float(ys)))
    return contour


def import_tracks(filepath, convert_contours=False):
    # pandas automatically converts values
    df = pd.read_csv(filepath)
    data = df.to_dict()
    data_transposed = {}
    for frame_index in data['frame']:
        row = {}
        for column in data.keys():
            if convert_contours and column == "contour":
                contour = extract_contour(data[column][frame_index])
                row[column] = contour
                data[column][frame_index] = contour
            else:
                row[column] = data[column][frame_index]
        data_transposed[frame_index] = row
    return data_transposed, data


def import_tracks_by_frame(filepath, convert_contours=False):
    # pandas automatically converts values
    df = pd.read_csv(filepath, index_col='frame')
    data = df.to_dict()
    if convert_contours:
        for frame_index in data['frame']:
            for column in data.keys():
                if column == "contour":
                    contour = extract_contour(data[column][frame_index])
                    data[column][frame_index] = contour
    return data


def import_tracks0(filepath):
    data = {}
    with open(filepath, mode='r') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            frame_index = row["frame"]
            row["contour"] = extract_contour(row["contour"])
            data[frame_index] = row
    return data
