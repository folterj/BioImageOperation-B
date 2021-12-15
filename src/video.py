import cv2 as cv
from tqdm import tqdm


def annotate_videos(video_infiles, video_outfile, datas, outratio=1):
    width, height, nframes, fps = video_info(video_infiles[0])
    interval = int(round(1 / outratio))
    vidwriter = cv.VideoWriter(video_outfile, -1, fps, (width, height))

    for video_infile in tqdm(video_infiles):
        vidcap = cv.VideoCapture(video_infile)
        framei = 0
        ok = vidcap.isOpened()
        while ok:
            ok = vidcap.isOpened()
            if ok:
                ok, video_frame = vidcap.read()
                if ok:
                    if framei % interval == 0:
                        for data in datas:
                            if data.start <= framei <= data.end and framei in data.frames:
                                position = (int(round(data.x[framei])), int(round(data.y[framei])))
                                draw_annotation(video_frame, data.pref_label, position)
                        vidwriter.write(video_frame)
            framei += 1
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


def draw_annotation(image, label, position, scale=0.5, thickness=1, color=(255, 255, 0)):
    position = (int(position[0]), int(position[1]))
    cv.drawMarker(image, position, color)
    return draw_text_abs(image, label, position, scale=scale, thickness=thickness, color=color)


def draw_text_abs(image, text, position, scale=0.5, thickness=1, color=(255, 255, 0), draw_mode=True):
    position = (int(position[0]), int(position[1]))
    fontface = cv.FONT_HERSHEY_SIMPLEX
    size = cv.getTextSize(text, fontface, scale, thickness)
    if draw_mode:
        cv.putText(image, text, position, fontface, scale, color, thickness, cv.LINE_AA)
    return size


def video_info(video_filepath: str) -> (int, int, int, float):
    vidcap = cv.VideoCapture(video_filepath)
    width = int(vidcap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv.CAP_PROP_FRAME_HEIGHT))
    nframes = int(vidcap.get(cv.CAP_PROP_FRAME_COUNT))
    fps = vidcap.get(cv.CAP_PROP_FPS)
    return width, height, nframes, fps
