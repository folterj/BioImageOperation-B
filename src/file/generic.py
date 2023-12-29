import os

from src.file.numpy_format import import_numpy
from src.file.plain_csv import import_csv


def import_file(filename):
    ext = os.path.splitext(filename)[1].lower()
    if ext.startswith('.np'):
        data = import_numpy(filename)
    else:
        data = import_csv(filename)
    return data
