import argparse
import yaml

from src.parameters import *
from src.relabelling import relabel


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Relabelling")
    parser.add_argument('--params',
                        help='The location of the parameters file',
                        default=DEFAULT_PARAMETER_FILENAME)
    args = parser.parse_args()
    with open(args.params, 'r') as file:
        params = yaml.safe_load(file)

    relabel(params)
