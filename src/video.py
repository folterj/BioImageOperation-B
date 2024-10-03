import cv2 as cv
import numpy as np
from tqdm import tqdm

from src.util import get_filetitle_replace, create_color_table, color_float_to_cv


def video_iterator(video_infiles, start=0, end=None, interval=1):
    for video_infile in tqdm(video_infiles):
        vidcap = cv.VideoCapture(video_infile)
        if start > 0:
            vidcap.set(cv.CAP_PROP_POS_FRAMES, start)
            framei = start
        else:
            framei = 0
        ok = vidcap.isOpened()
        while ok:
            ok = vidcap.isOpened()
            if ok:
                ok, video_frame = vidcap.read()
                if framei % interval == 0:
                    if ok:
                        yield video_frame
                    else:
                        yield None
            framei += 1
            if framei >= end:
                ok = False
        vidcap.release()


def annotate_videos(video_infiles, video_outfile, datas, params):
    interval = params.get('frame_interval', 1)
    start = params.get('frame_start', 0)
    end = params.get('frame_end', -1)
    show_labels = params.get('show_labels', [])
    width, height, nframes, fps = video_info(video_infiles[0])
    vidwriter = cv.VideoWriter(video_outfile, -1, fps, (width, height))
    colors = create_color_table(1000)

    for video_infile in tqdm(video_infiles):
        title = get_filetitle_replace(video_infile)
        video_datas = datas[title]
        vidcap = cv.VideoCapture(video_infile)
        if start > 0:
            vidcap.set(cv.CAP_PROP_POS_FRAMES, start)
            framei = start
        else:
            framei = 0
        ok = vidcap.isOpened()
        while ok:
            ok = vidcap.isOpened()
            if ok:
                ok, video_frame = vidcap.read()
                if ok:
                    if framei % interval == 0:
                        if 'frame' in show_labels:
                            draw_text_abs(video_frame, str(framei), (width // 2, height // 2), scale=2, thickness=2)
                        for label, data in video_datas.items():
                            if framei in data.position:
                                position = tuple(np.array(data.position[framei]).astype(int))
                                color = color_float_to_cv(colors[int(label) % len(colors)])
                                draw_annotation(video_frame, label, position, color=color)
                        vidwriter.write(video_frame)
            framei += 1
            if framei >= end:
                ok = False
        vidcap.release()
    vidwriter.release()


def annotate_video(video_infile, video_outfile, frames, all_positions, all_headers, all_data):
    width, height, nframes, fps = video_info(video_infile)
    vidcap = cv.VideoCapture(video_infile)
    vidwriter = cv.VideoWriter(video_outfile, -1, fps, (width, height))

    for frame_index in tqdm(frames):
        if vidcap.isOpened():
            success, video_frame = vidcap.read()
            if success:
                for positions, headers, data in zip(all_positions, all_headers, all_data):
                    annotate_frame(video_frame, frame_index, positions, headers, data)
                vidwriter.write(video_frame)

    vidwriter.release()
    vidcap.release()


def annotate_frame(video_frame, frame_index, positions, headers, all_data):
    dh = 20
    if frame_index in positions:
        x, y = map(int, positions[frame_index])
    else:
        x, y = 0, 40
    for header, data in zip(headers, all_data):
        if frame_index in data:
            frame_data = data[frame_index]
            y += dh
            text = header + ': '
            if isinstance(frame_data, int):
                text += str(frame_data)
            elif isinstance(frame_data, float):
                text += f'{frame_data:.2f}'
            else:
                text += frame_data
            size = draw_text_abs(video_frame, text, (x, y), draw_mode=False)
            x1 = max(x - size[0][0] // 2, 0)
            draw_text_abs(video_frame, text, (x1, y))


def draw_annotation(image, label, position, scale=1, thickness=1, color=(127, 127, 127)):
    position = (int(position[0]), int(position[1]))
    cv.drawMarker(image, position, color)
    return draw_text_abs(image, label, position, scale=scale, thickness=thickness, color=color)


def draw_text_abs(image, text, position, scale=1, thickness=1, color=(127, 127, 127), draw_mode=True):
    font_scale = scale * np.sqrt(image.shape[0] / 1000.0 * image.shape[1] / 1000.0) / 2
    position = (int(position[0]), int(position[1]))
    fontface = cv.FONT_HERSHEY_SIMPLEX
    size = cv.getTextSize(text, fontface, font_scale, thickness)
    if draw_mode:
        cv.putText(image, text, position, fontface, font_scale, color, thickness, cv.LINE_AA)
    return size


def video_info(video_filepath: str) -> (int, int, int, float):
    vidcap = cv.VideoCapture(video_filepath)
    width = int(vidcap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv.CAP_PROP_FRAME_HEIGHT))
    nframes = int(vidcap.get(cv.CAP_PROP_FRAME_COUNT))
    fps = vidcap.get(cv.CAP_PROP_FPS)
    return width, height, nframes, fps
