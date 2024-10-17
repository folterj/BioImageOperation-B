import cv2 as cv
import multiprocessing as mp
from tqdm import tqdm


def simple_reader(filename):
    vidcap = cv.VideoCapture(filename)
    #vidcap.set(cv.CAP_PROP_POS_FRAMES, start)
    nframes = int(vidcap.get(cv.CAP_PROP_FRAME_COUNT))
    for framei in tqdm(range(nframes)):
        ok = vidcap.isOpened()
        if ok:
            ok, video_frame = vidcap.read()
            if ok:
                shape = video_frame.shape
    vidcap.release()


def capture_frames(filename):
    capture = cv.VideoCapture(filename)
    capture.set(cv.CAP_PROP_BUFFERSIZE, 2)

    # FPS = 1/X, X = desired FPS
    FPS = 1 / 120
    FPS_MS = int(FPS * 1000)

    framei = 0
    while True:
        # Ensure camera is connected
        if capture.isOpened():
            (status, frame) = capture.read()
            # Ensure valid frame
            if status:
                shape = frame.shape
                print(framei)
            else:
                break
            framei += 1
            #time.sleep(FPS)

    capture.release()


if __name__ == '__main__':
    filename = 'D:/Video/Cooperative_digging/2024-08-29_16-11-00_SV11.mp4'
    simple_reader(filename)
    #capture_process = mp.Process(target=capture_frames, args=[filename])
    #capture_process.start()
