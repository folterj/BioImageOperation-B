import os

from src.Data import read_data
from src.util import get_filetitle_replace, get_input_files, filter_output_files
from src.video import annotate_videos


def run(all_params, params):
    general_params = all_params['general']
    base_dir = general_params['base_dir']
    input_files = get_input_files(general_params, params, 'input')
    if len(input_files) == 0:
        raise ValueError('Missing input files')
    video_files = filter_output_files(get_input_files(general_params, params, 'video_input'), all_params)
    if len(input_files) == 0:
        raise ValueError('Missing video files')
    video_output_path = os.path.join(base_dir, params['video_output'])
    frame_interval = params.get('frame_interval', 1)
    show_labels = params.get('show_labels', [])
    annotate_merge_videos(input_files, video_files, video_output_path, frame_interval, show_labels)


def annotate_merge_videos(input_files, video_files, video_output, frame_interval=1, show_labels=[]):
    print('Reading relabelled data')
    all_datas = {}
    for video_file in video_files:
        video_title = get_filetitle_replace(video_file)
        datas = {}
        for filename in input_files:
            if video_title in filename or len(input_files) == 1 or len(video_files) == 1:
                datas |= read_data(filename)
        all_datas[video_title] = datas
    print('Creating annotated video')
    annotate_videos(video_files, video_output, all_datas, frame_interval=frame_interval, show_labels=show_labels)
