DEFAULT_PARAMETER_FILENAME = 'resources/params.yml'

BIO_TRACKING_PATH = '../../flylab-behavioural/resources/bio/'
BIO_TRACKING_FILE = BIO_TRACKING_PATH + 'tracking_5.csv'
OUTPUT_PATH = '../output/'

#LIVING_EARTH_PATH = 'D:/Video/Living_Earth/Foraging/'
#LIVING_EARTH_INFILE = LIVING_EARTH_PATH + 'tracks1b/tracks_0.csv'
#LIVING_EARTH_OUTFILE = LIVING_EARTH_PATH + 'tracks1b/movement.csv'

#LIVING_EARTH_PATH = 'D:/Video/Living_Earth/Spider Activity/tracks/selected/'
#LIVING_EARTH_PATH = 'D:/Video/Living_Earth/Spider Activity/tracks full2/'
LIVING_EARTH_PATH = 'D:/Video/Living_Earth/Spider Activity/Activity Tracking July 2021/Cam 3/'
VIDEOS_PATH = LIVING_EARTH_PATH + '*.avi.mp4'
VIDEOS_OUTPUT = LIVING_EARTH_PATH + 'annotated.mp4'
TRACKS_PATH = LIVING_EARTH_PATH + 'tracks/*track*.csv'
TRACKS_RELABEL_PATH = LIVING_EARTH_PATH + 'tracks_relabel/'
TRACKS_RELABEL_FILES = TRACKS_RELABEL_PATH + '*track*.csv'
LABEL_ANNOTATION_IMAGE = LIVING_EARTH_PATH + 'back.png'
LABEL_ANNOTATION_FILENAME = LIVING_EARTH_PATH + 'annotations.csv'

OUTPUT_DATAFRAME = LIVING_EARTH_PATH + 'activity_dataframe.csv'
OUTPUT_PROFILE_V = LIVING_EARTH_PATH + 'profile_v.csv'
OUTPUT_PROFILE_VANGLE = LIVING_EARTH_PATH + 'profile_vangle.csv'

#LIVING_EARTH_VIDEO_INFILE = LIVING_EARTH_PATH + 'trim1.mp4'
#LIVING_EARTH_VIDEO_INFILE = LIVING_EARTH_PATH + 'trim1.mp4'
#LIVING_EARTH_VIDEO_OUTFILE = LIVING_EARTH_PATH + 'trim1_annotated.mp4'
LIVING_EARTH_VIDEO_INFILE = 'D:/Video/Living_Earth/Spider Activity/09-28-21_16-10-32.281.avi'
LIVING_EARTH_VIDEO_OUTFILE = 'D:/Video/Living_Earth/Spider Activity/09-28-21_16-10-32.281_annotated.mp4'

NBINS = 20
VANGLE_NORM = 360

PLOT_DPI = 300

MAX_ANNOTATION_DISTANCE = 10
MAX_MOVE_DISTANCE = 100
