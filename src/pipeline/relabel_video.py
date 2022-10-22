import os.path

from src.file.bio import import_tracks_by_frame
from src.util import get_filetitle, get_filetitle_replace, get_input_files, filter_output_files
from src.video import annotate_videos


def annotate_merge_videos(input_files, video_files, video_output, frame_interval):
    print('Reading relabelled data')
    all_datas = {}
    for video_file in video_files:
        video_datas = {}
        video_title = get_filetitle_replace(video_file)
        for filename in input_files:
            if video_title in filename:
                title = get_filetitle(filename)
                data, has_id = import_tracks_by_frame(filename)
                if has_id:
                    video_datas = data
                else:
                    label = title.rsplit('_')[-1]
                    video_datas[label] = data
        all_datas[video_title] = video_datas
    print('Creating annotated video')
    annotate_videos(video_files, video_output, all_datas, frame_interval=frame_interval)


def run(all_params, params):
    general_params = all_params['general']
    base_dir = general_params['base_dir']
    input_files = get_input_files(general_params, params, 'input')
    video_files = filter_output_files(get_input_files(general_params, params, 'video_input'), all_params)
    video_output_path = os.path.join(base_dir, params['video_output'])
    frame_interval = params.get('frame_interval', 1)
    annotate_merge_videos(input_files, video_files, video_output_path, frame_interval)
