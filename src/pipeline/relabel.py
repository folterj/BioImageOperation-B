import glob
import os
import cv2 as cv

from src.AnnotationView import AnnotationView
from src.pipeline.Relabeller import Relabeller
from src.util import get_input_files, filter_output_files


def run(all_params, params):
    general_params = all_params['general']
    base_dir = general_params['base_dir']
    method = params['method']
    input_files = get_input_files(general_params, params, 'input')
    if len(input_files) == 0:
        raise ValueError('Missing input files')
    video_files = filter_output_files(get_input_files(general_params, params, 'video_input'), all_params)
    output_dir = os.path.join(base_dir, params['output'])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        [os.remove(file) for file in glob.glob(os.path.join(output_dir, '*'))]

    annotation_filename = params.get('annotation_filename', '')
    if annotation_filename != '':
        annotation_filename = os.path.join(base_dir, annotation_filename)
    annotation_image_filename = params.get('annotation_image', '')
    if annotation_image_filename != '':
        annotation_image_filename = os.path.join(base_dir, annotation_image_filename)
    annotation_margin = params.get('annotation_margin', 0)

    if method.lower() == 'annotation':
        annotate(annotation_image_filename, annotation_filename, annotation_margin)

    relabeller = Relabeller(params, annotation_filename)
    relabeller.relabel_all(input_files, output_dir, video_files)


def annotate(annotation_image_filename, annotation_filename, annotation_margin):
    image = cv.imread(annotation_image_filename)
    if image is None:
        raise OSError(f'File not found: {annotation_image_filename}')
    print('User action: Review annotations (press ESC when finished)')
    annotator = AnnotationView(image, annotation_filename, annotation_margin)
    annotator.show_loop()
    annotator.close()
