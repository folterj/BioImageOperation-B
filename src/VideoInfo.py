from src.util import get_filetitle_replace
from src.video import video_info


class VideoInfo:
    def __init__(self, filename):
        self.filename = filename
        self.video_title = get_filetitle_replace(filename)
        self.width, self.height, self.total_frames, self.fps = video_info(filename)


class VideoInfos(dict):
    def __init__(self, filenames):
        super().__init__()
        self.total_frames = 0
        self.total_length = 0
        for filename in filenames:
            video_title = get_filetitle_replace(filename)
            video_info = VideoInfo(filename)
            self.total_frames += video_info.total_frames
            self.total_length += video_info.total_frames / video_info.fps
            self[video_title] = video_info

    def find_match(self, target):
        for key, item in self.items():
            if key in target:
                return item
        return None
