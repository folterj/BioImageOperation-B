import os.path

import numpy as np
import pandas as pd


def import_numpy(filename):
    data = {}
    filebase, ext = os.path.splitext(filename)

    num_start = None
    for i in range(len(filebase) - 1, -1, -1):
        if filebase[i].isdigit():
            num_start = i
        else:
            break
    if num_start is not None:
        id = int(filebase[num_start:])
    else:
        id = 0
    data[id] = {}

    npfile = np.load(filename)
    npdict = {key: value for key, value in npfile.items()}
    columns = list(npdict.keys())
    frame_cols = []
    for i, column in enumerate(columns):
        if column.lower() == 'frame':
            frame_cols.append(i)
    has_frame = (len(frame_cols) > 0)

    if has_frame:
        frames = npdict[columns[frame_cols[0]]].astype(int)
    else:
        frames = range(len(npdict[list(npdict.keys())[0]]))

    for column, values in npdict.items():
        data[id][column.lower()] = {frame: value for frame, value in zip(frames, values)}
    return data
