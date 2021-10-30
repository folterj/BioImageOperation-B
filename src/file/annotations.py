import csv
import os


def load_annotations(filename):
    annotations = []
    if os.path.exists(filename):
        with open(filename) as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                annotations.append([int(row[0]), int(row[1]), row[2]])
    return annotations


def save_annotations(filename, annotations):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(annotations)
