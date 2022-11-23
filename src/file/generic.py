import os

from src.file.numpy_format import import_numpy
from src.file.plain_csv import import_csv


def import_file(filename, add_position=False):
    ext = os.path.splitext(filename)[1].lower()
    if ext.startswith('.np'):
        return import_numpy(filename)
    else:
        return import_csv(filename, add_position=add_position)
