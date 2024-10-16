from src.pipeline.Tracker import Tracker
from src.util import get_input_files, try_path_join


def run(all_params, params):
    general_params = all_params['general']
    base_dir = general_params['base_dir']
    input_files = get_input_files(general_params, params, 'input')
    video_input = get_input_files(general_params, params, 'video_input')
    output = try_path_join(base_dir, params.get('output'))
    video_output = try_path_join(base_dir, params.get('video_output'))
    if len(input_files) == 0:
        raise ValueError('Missing input files')

    tracker = Tracker(params, base_dir, input_files, video_input, output, video_output)
    tracker.track()
