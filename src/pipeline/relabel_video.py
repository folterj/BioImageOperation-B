import os.path

from src.file.bio import import_tracks_by_frame
from src.util import get_filetitle, get_filetitle_replace, get_input_files
from src.video import annotate_videos


def annotate_merge_videos(input_files, video_files, video_output):
    print('Reading relabelled data')
    all_datas = {}
    for video_file in video_files:
        video_datas = {}
        video_title = get_filetitle_replace(video_file)
        for filename in input_files:
            if video_title in filename:
                title = get_filetitle(filename)
                label = title.rsplit('_')[-1]
                video_datas[label] = import_tracks_by_frame(filename)
        all_datas[video_title] = video_datas
    print('Creating annotated video')
    annotate_videos(video_files, video_output, all_datas, frame_inerval=100)


def run(general_params, params):
    base_dir = general_params['base_dir']
    input_files = get_input_files(general_params, params, 'input')
    video_files = get_input_files(general_params, params, 'video_input')
    video_output_path = os.path.join(base_dir, params['video_output'])
    annotate_merge_videos(input_files, video_files, video_output_path)
