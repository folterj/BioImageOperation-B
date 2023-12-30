import os

from src.Data import create_data, write_data
from src.util import get_input_files


def import_files(path, window_size='1s'):
    input_files = get_input_files({'base_dir': os.getcwd()}, {'input': path}, 'input')
    features = create_data(input_files, window_size=window_size)
    return features


if __name__ == '__main__':
    input_path = 'D:/Video/2022_06_22_exp7/test/*.csv'
    output_path = 'D:/Video/2022_06_22_exp7/test_output'
    features = import_files(input_path, window_size='7')
    write_data(output_path, features)
