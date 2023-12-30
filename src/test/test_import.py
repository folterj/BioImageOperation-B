import os

from src.Data import create_data, write_data
from src.util import get_input_files


def import_files(path, fps=1, pixel_size=1, window_size='1s'):
    input_files = get_input_files({'base_dir': os.getcwd()}, {'input': path}, 'input')
    datas = create_data(input_files, fps=fps, pixel_size=pixel_size, window_size=window_size)
    return datas


if __name__ == '__main__':
    input_path = 'D:/Video/2022_06_22_exp7/test/*.csv'
    output_path = 'D:/Video/2022_06_22_exp7/test_output'
    datas = import_files(input_path, fps=25)
    for data in datas:
        data.calc_windows()
    write_data(output_path, datas)
