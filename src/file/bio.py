import os
import pandas as pd

from src.util import pairwise


def import_dataframe(filepath):
    # pandas automatically converts values
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.lower()
    id_col = None
    frame_col = None
    for header in df.columns:
        if header in ['id', 'track_label']:
            id_col = header
        if header == 'frame':
            frame_col = header
    return df, id_col, frame_col


def import_tracks_by_id(filepath):
    df, id_col, frame_col = import_dataframe(filepath)
    has_id = (id_col is not None)
    if has_id:
        data = {}
        for id in set(df[id_col]):
            id_df = df[df[id_col] == id]
            data[str(id)] = id_df.to_dict()
    else:
        data = df.to_dict()
    return data, has_id


def import_tracks_by_frame(filepath, convert_contours=False):
    df, id_col, frame_col = import_dataframe(filepath)
    has_id = (id_col is not None)
    has_frames = (frame_col is not None)
    if has_id:
        data = {}
        for id in set(df[id_col]):
            id_df = df[df[id_col] == id]
            if not has_frames:
                id_df.insert(0, frame_col, range(len(id_df)))
            id_df.set_index(frame_col, drop=False, inplace=True)
            data[str(id)] = id_df.to_dict()
    else:
        data = df.set_index(df.columns[0], drop=False).to_dict()

    if convert_contours:
        for frame_index in data['frame']:
            for column in data.keys():
                if column == "contour":
                    contour = extract_contour(data[column][frame_index])
                    data[column][frame_index] = contour
    return data, has_id


def extract_contour(s):
    contour = []
    for xs, ys in pairwise(s.split()):
        contour.append((float(xs), float(ys)))
    return contour


def export_tracks(filepath, data):
    df = pd.DataFrame.from_dict(data)
    exists = os.path.exists(filepath)
    df.to_csv(filepath, mode='a', index=False, header=not exists)
