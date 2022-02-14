import argparse
import yaml

from src.parameters import *
from src.process_features import extract_activity_features


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Extract activity features")
    parser.add_argument('--params',
                        help='The location of the parameters file',
                        default=DEFAULT_PARAMETER_FILENAME)
    args = parser.parse_args()
    with open(args.params, 'r') as file:
        params = yaml.safe_load(file)

    extract_activity_features(params)
