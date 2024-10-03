import cv2 as cv
import os
from tqdm import tqdm

from src.Data import read_data
from src.file.streaming import get_stream_iterator, get_stream_total
from src.util import *
from src.video import annotate_videos, video_iterator, video_info, draw_annotation


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
    stream = params.get('stream', False)
    if stream:
        annotate_stream_video(input_files, video_files, video_output_path, params)
    else:
        annotate_merge_videos(input_files, video_files, video_output_path, params)


def annotate_stream_video(input_files, video_files, video_output, params):
    total_rows = get_stream_total(input_files)
    data_iterator = get_stream_iterator(input_files)
    width, height, nframes, fps = video_info(video_files[0])
    frame_start = get_frames_number(params.get('frame_start', 0), fps)
    frame_end = get_frames_number(params.get('frame_end'), fps)
    frame_interval = get_frames_number(params.get('frame_interval', 1), fps)
    frame_iterator = video_iterator(video_files,
                                    start=frame_start, end=frame_end, interval=frame_interval)

    vidwriter = cv.VideoWriter(video_output, -1, fps, (width, height))
    label_color = color_float_to_cv((1, 0, 0))

    frames = range(frame_start, frame_end, frame_interval)
    data = {'frame': -1, 'values': {}}
    for framei, image in tqdm(zip(frames, frame_iterator), total=len(frames)):
        frame_done = False
        while not frame_done:
            data_framei = data['frame']
            if data_framei == framei:
                id = data['id']
                values = data['values']
                position = (values['x_body'], values['y_body'])
                draw_annotation(image, str(id), position, color=label_color)
            elif data_framei > framei:
                frame_done = True
            if not frame_done:
                data = next(data_iterator)
        vidwriter.write(image)
    vidwriter.release()

def annotate_merge_videos(input_files, video_files, video_output, params):
    print('Reading label data')
    all_datas = {}
    for video_file in video_files:
        video_title = get_filetitle_replace(video_file)
        datas = {}
        for filename in input_files:
            if video_title in filename or len(input_files) == 1 or len(video_files) == 1:
                datas |= read_data(filename)
        all_datas[video_title] = datas
    print('Creating annotated video')
    annotate_videos(video_files, video_output, all_datas, params)
