import argparse
import yaml

from src.process_features import extract_response_features

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Extract response features")
    parser.add_argument('--params',
                        required=True,
                        help='The location of the parameters file')
    args = parser.parse_args()
    with open(args.params, 'r') as file:
        params = yaml.safe_load(file)

    extract_response_features(params)
