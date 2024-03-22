from src.pipeline.Tracker import Tracker
from src.util import get_input_files


def run(all_params, params):
    general_params = all_params['general']
    base_dir = general_params['base_dir']
    input_files = get_input_files(general_params, params, 'input')
    output = get_input_files(general_params, params, 'output')
    video_output = get_input_files(general_params, params, 'video_output')
    if len(input_files) == 0:
        raise ValueError('Missing input files')

    tracker = Tracker(params, base_dir, input_files, output, video_output)
    tracker.track()
