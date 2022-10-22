import csv
import os


def load_annotations(filename, multiple_frames=False):
    rerun = False
    annotations = {}
    if os.path.exists(filename):
        with open(filename) as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)
            labelpos = list(range(len(header)))
            id_col = None
            frame_col = None
            for i, label in enumerate(header):
                if 'id' in label.lower():
                    id_col = i
                    labelpos.remove(id_col)
                if 'frame' in label.lower():
                    frame_col = i
                    labelpos.remove(frame_col)
                    multiple_frames = True
            if id_col is None:
                id_col = 0
            for row in reader:
                id = row[id_col]
                frame_id = None
                if frame_col is not None:
                    frame_id = row[frame_col]
                position = float(row[labelpos[0]]), float(row[labelpos[1]])
                if multiple_frames:
                    if id not in annotations:
                        annotations[id] = {}
                    if frame_id is None:
                        frame_id = len(annotations[id])
                    annotations[id][frame_id] = position
                elif id in annotations:
                    #id already in annotations; multiple frames
                    multiple_frames = True
                    rerun = True
                    break
                else:
                    annotations[id] = position
    if rerun:
        return load_annotations(filename, multiple_frames)
    else:
        return annotations


def save_annotations(filename, annotations):
    header = ['id', 'x', 'y']
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for id, position in annotations.items():
            row = list(position)
            row.insert(0, id)
            writer.writerow(row)
