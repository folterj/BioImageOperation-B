import os

from src.file.numpy_format import import_numpy
from src.file.plain_csv import import_csv
from src.util import isvalid_position


def import_file(filename, add_position=False):
    ext = os.path.splitext(filename)[1].lower()
    if ext.startswith('.np'):
        data = import_numpy(filename)
    else:
        data = import_csv(filename)
    if add_position:
        for values in data.values():
            if 'x' in values:
                values['position'] = {frame: (x, y) for frame, x, y in zip(values['x'].keys(), values['x'].values(), values['y'].values())
                                      if isvalid_position((x, y))}
    return data
